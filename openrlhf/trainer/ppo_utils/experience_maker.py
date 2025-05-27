from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import math
from openrlhf.models.actor import Actor
from openrlhf.models.utils import compute_approx_kl, masked_mean
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.verifier import verifier, extract_boxed_content
import torch
import torch.nn as nn
from tqdm import tqdm
logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
	if isinstance(tensor, list):
		return [to(t, device) for t in tensor]
	return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
	if isinstance(tensor, list):
		return [pin_memory(t) for t in tensor]
	return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience:
	"""Experience is a batch of data.

	These data should have the the sequence length and number of actions. Left
	padding for sequences is applied.

	Shapes of each tensor:
	sequences: (B, S)
	action_log_probs: (B, A)
	attention_mask: (B, S)
	action_mask: (B, A)
	kl: (B, A)

	"A" is the number of actions.
	"""

	sequences: torch.Tensor
	action_log_probs: torch.Tensor
	advantages: Optional[torch.Tensor]
	attention_mask: Optional[torch.LongTensor]
	action_mask: Optional[torch.BoolTensor]
	info: Optional[dict]
	kl: Optional[torch.Tensor] = None

	@torch.no_grad()
	def to_device(self, device: torch.device):
		self.sequences = to(self.sequences, device)
		self.action_log_probs = to(self.action_log_probs, device)
		self.advantages = to(self.advantages, device)
		self.attention_mask = to(self.attention_mask, device)
		self.action_mask = to(self.action_mask, device)
		self.kl = to(self.kl, device)
		self.info = {key: to(value, device) for key, value in self.info.items()}
		return self

	def pin_memory(self):
		self.sequences = pin_memory(self.sequences)
		self.action_log_probs = pin_memory(self.action_log_probs)
		self.attention_mask = pin_memory(self.attention_mask)
		self.action_mask = pin_memory(self.action_mask)
		self.kl = pin_memory(self.kl)
		self.info = {key: pin_memory(value) for key, value in self.info.items()}
		return self


@dataclass
class Samples:
	"""Samples is a batch of data.

	There can be 2 formats to store the samples, batched or packed. The batched
	format means padding is applied to the sequences, while the packed format will
	concatenate the prompt and response without padding.

	Shapes of each tensor, when 2 shapes are shown, the first one is for batched
	format
			and the second one is for packed format:
	sequences: (B, S) or (1, total_length), the tokens of both prompt and
	response.
	attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
	action_mask: (B, A) or None, the action (response) mask to show which part of
	the
			sequence is the response. When the samples are packed, this is None.
	num_actions: int or (B,), the number of actions (tokens) in the response.
			When the samples are not packed, we will use action_mask, so this is an
			int to
			show the size of action_mask. Otherwise, this is a tensor to show the
			number of
			actions for each sample.
	packed_seq_lens: None or (B,), the length of each sample in the packed
	samples.
	response_length: (B,), the number of tokens in the response.
	total_length: (B,), the total number of tokens in the sequences.
	"""

	sequences: torch.Tensor
	attention_mask: Optional[torch.LongTensor]
	action_mask: Optional[torch.BoolTensor]
	num_actions: Union[int, torch.Tensor]
	answers: Optional[list[str]]
	packed_seq_lens: Optional[torch.Tensor]
	response_length: torch.Tensor
	total_length: torch.Tensor


class NaiveExperienceMaker(ABC):
	"""Naive experience maker."""

	def __init__(
			self,
			actor: Actor,
			critic: nn.Module,
			reward_model: nn.Module,
			initial_model: Actor,
			tokenizer,
			prompt_max_len: int,
			kl_controller,
			strategy=None,
			remote_rm_url: str = None,
			reward_fn=None,
	) -> None:
		super().__init__()
		self.actor = actor
		self.critic = critic
		self.reward_model = reward_model
		self.remote_rm_url = remote_rm_url
		self.initial_model = initial_model
		self.tokenizer = tokenizer
		self.prompt_max_len = prompt_max_len
		self.kl_ctl = kl_controller
		self.strategy = strategy
		self.reward_fn = reward_fn
		self.perf_stats = None
	# tokenizer
	def tokenize_fn(self, texts, max_length, padding=True, device=None):
		if not padding:
			# when padding is False, return tokenized texts as list
			return self.tokenizer(
					texts,
					add_special_tokens=False,
					max_length=max_length,
					truncation=True,
			)
		batch = self.tokenizer(
				texts,
				return_tensors="pt",
				add_special_tokens=False,
				max_length=max_length,
				padding=True,
				truncation=True,
		)
		return {k: v.to(device) for k, v in batch.items()}

	@torch.no_grad()
	def make_experience_list(
			self, all_prompts: Union[str, List[str]], **generate_kwargs
	) -> List[Experience]:
		"""Make a list of experience with the micro_rollout_batch_size.

		This method will first calculate the response sequences and rewards for the
		given prompts.
		Then, if we need certain processing for the rewards or do certain filtering,
		we can process the rollout as a whole.
		After that, we will calculate the advantages and returns for each
		experience.
		"""
		args = self.strategy.args
		# generate responses
		samples_list = self.generate_samples(all_prompts, **generate_kwargs)
		torch.distributed.barrier()

		experiences = []
		answers = [sample.answers for sample in samples_list]
		answers = sum(answers, [])
		answers = [answers[i] for i in range(0, len(answers), args.n_samples_per_prompt)]
		completions = sum([self.tokenizer.batch_decode(sample.sequences[:, -sample.num_actions:].cpu(), skip_special_tokens=True) for sample in samples_list], [])
		all_online_candidates = [extract_boxed_content(completion) for completion in completions]
		all_online_candidates = [all_online_candidates[i:i+args.n_samples_per_prompt] for i in range(0, len(all_online_candidates), args.n_samples_per_prompt)]

		online_candidates = []
		for idx, item in enumerate(all_online_candidates):
			online_candidate_i = []
			all_online_candidate_i = list(set(item))
			gt_contained = False
			if all_online_candidate_i:
				for candidate in all_online_candidate_i:
					if len(candidate) <= 20:
						online_candidate_i.append(candidate)
					if verifier(candidate, answers[idx]):
						gt_contained = True
			if not gt_contained:
				online_candidate_i.append(answers[idx])

			online_candidates += [online_candidate_i] * args.n_samples_per_prompt
		for idx_list, samples in tqdm(
				enumerate(samples_list),
				desc="make_experience",
				disable=not self.strategy.is_rank_0(),
		):
			minibatch_online_candidates = online_candidates[idx_list * args.micro_rollout_batch_size: (idx_list + 1) * args.micro_rollout_batch_size]
			experiences.append(
					self.make_experience(samples, minibatch_online_candidates, **generate_kwargs).to_device("cpu")
			)

		experiences, posterior_qs, outcome_rewards = self.process_experiences(experiences)

		for experience, posterior_q, outcome_reward in zip(
				experiences, posterior_qs, outcome_rewards
		):
			experience = experience.to_device("cuda")
			posterior_q = posterior_q.to(device="cuda")
			outcome_reward = outcome_reward.to(device="cuda")

			if experience.action_mask is not None:
				posterior_q = experience.action_mask * posterior_q
			experience.advantages = posterior_q.detach()
			
			posterior_q_mean = (
					posterior_q[experience.action_mask]
					.mean()
					.unsqueeze(-1)
					.expand(posterior_q.size(0), 1)
			)
			outcome_reward_mean = (
					outcome_reward
					.mean()
					.unsqueeze(-1)
					.expand(outcome_reward.size(0), 1)
			)
			experience.info["posterior_q"] = posterior_q_mean
			experience.info["outcome_reward"] = outcome_reward_mean
			experience.kl = None
			del experience.info["num_actions"]
			experience.to_device("cpu")
		return experiences

	@torch.no_grad()
	def generate_samples(
			self, all_prompts_answers: List[str], **generate_kwargs
	) -> List[Samples]:
		"""Generate samples and return in batches."""
		assert not getattr(self, "packing_samples", False)
		args = self.strategy.args
		self.actor.eval()
		# sample multiple response
		all_prompts = sum(
				[
						[prompt_answer[0]] * args.n_samples_per_prompt
						for prompt_answer in all_prompts_answers
				],
				[],
		)
		all_answers = sum(
				[
						[prompt_answer[1]] * args.n_samples_per_prompt
						for prompt_answer in all_prompts_answers
				],
				[],
		)
		samples_list = []
		for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
			prompts = all_prompts[i : i + args.micro_rollout_batch_size]
			inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
			sequences, attention_mask, action_mask = self.actor.generate(
					**inputs, **generate_kwargs
			)
			samples = Samples(
					sequences=sequences,
					attention_mask=attention_mask,
					action_mask=action_mask,
					num_actions=action_mask.size(1),
					answers=all_answers[i : i + args.micro_rollout_batch_size],
					packed_seq_lens=None,
					response_length=action_mask.float().sum(dim=-1),
					total_length=attention_mask.float().sum(dim=-1),
			)
			samples_list.append(samples)
		return samples_list

	@torch.no_grad()
	def make_experience(self, samples: Samples, online_candidates=None, **generate_kwargs) -> Experience:
		"""Turn samples into experience by calculating logprobs, values, rewards, and kl divergence."""
		self.actor.eval()
		self.initial_model.eval()
		if self.reward_model is not None:
			self.reward_model.eval()
		if self.critic is not None:
			self.critic.eval()

		# extract values from samples
		sequences = samples.sequences
		attention_mask = samples.attention_mask
		action_mask = samples.action_mask
		num_actions = samples.num_actions
		answers = samples.answers
		# log probs
		action_log_probs = self.actor(sequences, num_actions, attention_mask)

		# init log probs
		base_action_log_probs = self.initial_model(
				sequences, num_actions, attention_mask
		)

		completions = self.tokenizer.batch_decode(sequences[:, -num_actions:].cpu(), skip_special_tokens=True)
		outcome_reward = torch.tensor([
				[float(verifier(completions[i], answers[i]))] * num_actions
				for i in range(len(completions))
		])
		posterior_q = self.posterior_value(sequences, num_actions, action_mask, answers, outcome_reward, online_candidates)
		kl = compute_approx_kl(
				action_log_probs,
				base_action_log_probs,
				action_mask=action_mask,
				use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
		)

		info = {
				"kl": masked_mean(kl, action_mask, dim=-1),
				"posterior_q": posterior_q,
				"outcome_reward": outcome_reward,
				"response_length": samples.response_length,
				"total_length": samples.total_length,
				"num_actions": num_actions,
		}
		# reset model state
		self.actor.train()

		return Experience(
				sequences,
				action_log_probs,
				None,
				attention_mask,
				action_mask,
				info,
				kl
		)

	@torch.no_grad()
	def posterior_value(self, sequences, num_actions, action_mask, answers, outcome_reward, candidates):
		posterior_q = torch.zeros(sequences.size(0), num_actions, device="cpu")
		batch_size, seq_len = sequences.size()
		last_content_indices = []
		start_idx = seq_len - num_actions - 1
		for i in range(batch_size):
			valid_len = (
				action_mask[i, -num_actions:].nonzero()[-1].item() + 1
				if action_mask[i, -num_actions:].any() else 0
			)
			last_content_indices.append(start_idx + valid_len)

		max_position = max(last_content_indices)
		chunk_size = self.strategy.args.step_size
		all_positions = list(range(start_idx, max_position + 1, chunk_size))
		if not all_positions or all_positions[-1] != max_position:
			all_positions.append(max_position)

		all_weighted_q = [[] for _ in range(batch_size)]
		all_candidate_tokens = []
		candidates_modified = []
		for i in range(batch_size):
			seq_candidates_tokens = []
			candidates_modified_i = []
			gt_answer = None
			for candidate in candidates[i]:
				answer_ids = self.tokenizer("{" + candidate + "}.",  add_special_tokens=False)["input_ids"]
				answer_tensor = torch.tensor(answer_ids, dtype=sequences[0].dtype, device=sequences[0].device)
				candidates_modified_i.append("{" + candidate + "}.")
				seq_candidates_tokens.append(answer_tensor)
				if verifier(candidate, answers[i]):
					gt_answer = candidate
			if gt_answer is None:
				gt_answer = answers[i]
				gt_ids = self.tokenizer("{" + gt_answer + "}.",  add_special_tokens=False)["input_ids"]
				gt_tensor = torch.tensor(gt_ids, dtype=sequences[0].dtype, device=sequences[0].device)
				seq_candidates_tokens.append(gt_tensor)
				candidates_modified_i.append("{" + gt_answer + "}.")
			candidates_modified_i.append("{" + gt_answer + "}.")
			gt_ids = self.tokenizer("{" + gt_answer + "}.",  add_special_tokens=False)["input_ids"]
			gt_tensor = torch.tensor(gt_ids, dtype=sequences[0].dtype, device=sequences[0].device)
			seq_candidates_tokens.append(gt_tensor)
			candidates_modified.append(candidates_modified_i)
			all_candidate_tokens.append(seq_candidates_tokens)
		for seq_idx in range(batch_size):
			seq = sequences[seq_idx]
			weight = torch.ones(len(candidates_modified[seq_idx])-1, device=seq.device)
			last_candidate_probs = None
			last_answer_prob = None
			weight_positions = []
			candidate_probs_positions = []
			for position_idx, position in enumerate(all_positions):
				if position > last_content_indices[seq_idx]:
					continue
				prompt = self.tokenizer.decode(seq[:position], skip_special_tokens=False)
				context_ids = self.tokenizer(prompt + '\nBased on the above reasoning, the final answer is \\boxed',  add_special_tokens=False)["input_ids"]
				context_length = len(context_ids)
				candidate_inputs = []
				for i, cand_tokens in enumerate(all_candidate_tokens[seq_idx]):
					context_ids = self.tokenizer(prompt + '\nBased on the above reasoning, the final answer is \\boxed' + candidates_modified[seq_idx][i], add_special_tokens=False)["input_ids"]
					context_tensor = torch.tensor(context_ids, dtype=seq.dtype, device=seq.device)
					candidate_inputs.append(context_tensor)
				candidate_inputs_padded = torch.nn.utils.rnn.pad_sequence(
					candidate_inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side="right"
				)

				attention_mask = (candidate_inputs_padded != self.tokenizer.pad_token_id).long()
				outputs = self.actor.model(input_ids=candidate_inputs_padded, attention_mask=attention_mask)
				logits = outputs.logits

				candidate_probs = []
				for i, cand_tokens in enumerate(all_candidate_tokens[seq_idx]):
					cand_len = cand_tokens.size(0)
					insertion_logprob = 0.0
					valid_tokens = 0

					token_logits = logits[i, context_length - 1, :]
					probs = torch.nn.functional.softmax(token_logits, dim=-1)
					for token_idx in range(1, cand_len):
						pos = context_length + token_idx - 1
						token_logits = logits[i, pos, :]
						probs = torch.nn.functional.softmax(token_logits, dim=-1)
						token_prob = probs[cand_tokens[token_idx]].item()
						try:
							insertion_logprob += math.log(token_prob)
						except:
							valid_tokens -= 1
						valid_tokens += 1
					if valid_tokens > 0:
						candidate_probs.append(math.exp(insertion_logprob / valid_tokens))
					else:
						candidate_probs.append(0.0)
				candidate_probs = torch.tensor(candidate_probs, device=seq.device)
				answer_prob = candidate_probs[-1]
				candidate_probs = candidate_probs[:-1]
				if position_idx == 0:
					temp_weight = weight * candidate_probs
					temp_weight = temp_weight / temp_weight.sum()
					weight_positions.append(temp_weight)
					candidate_probs_positions.append(candidate_probs)
				else:
					r_m = candidate_probs - last_candidate_probs
					r_gt = answer_prob - last_answer_prob
					value_error = torch.absolute(r_m - r_gt)
					weight = weight * torch.exp(-1.0 * value_error)
					temp_weight = weight * candidate_probs
					temp_weight = temp_weight / temp_weight.sum()
					weight_positions.append(temp_weight)
					candidate_probs_positions.append(candidate_probs)
				last_candidate_probs = candidate_probs.clone()
				last_answer_prob = answer_prob.clone()
			for posid, temp_weight in enumerate(weight_positions):
				weighted_q = ((candidate_probs - candidate_probs_positions[posid]) * temp_weight).sum()
				all_weighted_q[seq_idx].append(weighted_q)
		for i in range(batch_size):
			chunks = all_weighted_q[i]
			for chunk_idx, value in enumerate(chunks):
				start = chunk_idx * chunk_size
				end = min(start + chunk_size, num_actions)
				posterior_q[i, start:end] = value
		posterior_q += outcome_reward
		return posterior_q

	@torch.no_grad()
	def process_experiences(
			self, experiences: List[Experience]
	) -> Tuple[List[Experience], List[torch.Tensor]]:
		"""Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

		Output:
		- experiences: List of Experience
		- rewards: List of rewards
		"""
		return (
				experiences,
				[experience.info["posterior_q"] for experience in experiences],
				[experience.info["outcome_reward"] for experience in experiences],
		)
