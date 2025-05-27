from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from .utils import masked_mean


class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    """

    def __init__(self, ring_attn_group=None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

        self.ring_attn_group = ring_attn_group
        if self.ring_attn_group:
            self.ring_attn_rank = dist.get_rank(self.ring_attn_group)
            self.ring_attn_world_size = dist.get_world_size(self.ring_attn_group)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # RingAttention
        if self.ring_attn_group is not None:
            total_seq_len = labels.size(-1)
            seq_len_per_process = total_seq_len // self.ring_attn_world_size
            start_idx = self.ring_attn_rank * seq_len_per_process
            end_idx = min(start_idx + seq_len_per_process, total_seq_len)
            labels = labels[..., start_idx:end_idx]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # if labels are all IGNORE_INDEX, then nn.CrossEntropyLoss will be nan
            if torch.all(shift_labels == self.IGNORE_INDEX):
                # Use mean of logits multiplied by 0 to maintain gradient flow
                loss = shift_logits.mean() * 0
            else:
                loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=self.ring_attn_group)
            loss = loss / self.ring_attn_world_size
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss


class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.2) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        ratio = masked_mean(ratio, action_mask, dim=-1).mean()
        return loss, ratio


class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps: float = None) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.clip_eps is not None:
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            surr1 = (values_clipped - returns) ** 2
            surr2 = (values - returns) ** 2
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return 0.5 * loss


class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()


class LogExpLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2204.05862
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        loss = torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()
        return loss


class DPOLoss(nn.Module):
    """
    DPO Loss
    """

    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L742
class VanillaKTOLoss(nn.Module):
    """
    KTO loss for even sampling
    """

    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        losses = torch.cat(
            (
                1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        ).mean()

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L770
class KTOLoss(nn.Module):
    """
    KTO loss for uneven sampling
    """

    def __init__(
        self, beta: float, desirable_weight: float, undesirable_weight: float, world_size: int, device: torch.device
    ) -> None:
        super().__init__()
        self.beta = beta
        self.world_size = world_size
        self.device = device
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        KL = (policy_KL_logps - reference_KL_logps).mean().detach()
        # all_reduce sums up the KL estimates across all devices (gradient will also be scaled by world size)
        dist.all_reduce(KL, op=dist.ReduceOp.SUM)
        # take average (will also scale gradients appropriately)
        KL = (KL / self.world_size).clamp(min=0)

        if policy_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - KL))
            chosen_rewards = self.beta * chosen_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            chosen_losses = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)
            chosen_rewards = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)

        if policy_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            rejected_losses = 1 - F.sigmoid(self.beta * (KL - rejected_logratios))
            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            rejected_losses = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)
            rejected_rewards = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)

        losses = torch.cat(
            (self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses), 0
        ).mean()
        return losses, chosen_rewards, rejected_rewards, KL


# Adapted from https://github.com/microsoft/LMOps/blob/main/minillm/finetune.py#L166
class KDLoss(nn.Module):
    """
    Language Model Knowledge Distillation Loss
    """

    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100

    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(logits)
        logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (label != self.IGNORE_INDEX).int()
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

        return distil_loss


class PRMLoss(nn.Module):
    """
    Process Reward Model Loss
    """

    def __init__(self, placeholder_token_id: int, reward_token_ids: Optional[list[int]] = None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)
        self.placeholder_token_id = placeholder_token_id
        self.reward_token_ids = reward_token_ids

    def forward(self, inputs: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, *, return_acc: bool = False):
        placeholder_mask = inputs == self.placeholder_token_id
        logits = logits[placeholder_mask]
        labels = labels[placeholder_mask]

        if labels.dtype == torch.float:
            # soft label
            assert len(self.reward_token_ids) == 2, "reward_token_ids should have 2 tokens for soft labels"
            logits = logits[..., self.reward_token_ids]
            positive_labels = labels.to(logits.dtype)
            negative_labels = 1 - positive_labels
            negative_labels[positive_labels != -100] = 1 - positive_labels[positive_labels != -100]
            labels = torch.stack([positive_labels, negative_labels], dim=-1)
        elif self.reward_token_ids is not None:
            # hard label with reward_token_ids set. (otherwise the whole vocab will be trained together.)
            logits = logits[..., self.reward_token_ids]
            # this is slow....
            for i, token in enumerate(self.reward_token_ids):
                labels = torch.where(labels == token, i, labels)

        loss = self.loss(logits, labels)
        if not return_acc:
            return loss

        if labels.dtype == logits.dtype:
            labels = labels.argmax(dim=-1)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, acc

class MyPRMLoss(nn.Module):
    def __init__(self, reward_token_ids):
        super().__init__()
        self.reward_token_ids = reward_token_ids
        self.loss = nn.CrossEntropyLoss()
    def forward(self, logits, reward, action_mask, return_acc):
        if logits.size(-1) > 5000:   # actor, lm head
            logits = logits[:, :, self.reward_token_ids]
     #   else:   # critic, new head
     #       logits = logits[:, :, :]
        logits = logits[action_mask]
        reward = reward[action_mask]
        labels = reward.type(torch.LongTensor).to("cuda")
        loss = self.loss(logits, labels)
        if not return_acc:
            return loss
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, acc

class CEValueLossMask(nn.Module):
    def __init__(self, ensemble=True, mask_prob=0.0):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.ensemble = ensemble
        self.mask_prob = mask_prob

    """
    @torch.no_grad()
    def weight_update(self, value, returns):
    #    value_error = torch.absolute(value - torch.unsqueeze(returns, -1))
        value_error = (value - torch.unsqueeze(returns, -1)) ** 2
       # posterior = torch.ones_like(value_error) * torch.exp(-value_error)
        weight = torch.exp(-value_error)
        posterior = torch.cumprod(weight, dim=1)
        posterior /= posterior.sum(axis=-1, keepdims=True)
        print("posterior", posterior.size(), posterior[0, ...])
        return posterior # (value * posterior).sum(axis=-1)
    """

    def weight_update(self, value, returns):
        chunk_size = 32
        #value_error = (value - torch.unsqueeze(returns, -1)) ** 2
        value_error = torch.absolute(value - torch.unsqueeze(returns, -1))
        batch_size, seq_len, num_ensembles = value_error.shape
        weight = torch.exp(-5.0*value_error)
        posterior = torch.zeros_like(value_error)

        chunk_indices = torch.arange(0, seq_len, chunk_size)
        num_chunks = len(chunk_indices)
        chunk_weight = weight[:, chunk_indices, :]  # Shape: [batch_size, num_chunks, num_ensembles]
        chunk_posterior = torch.cumprod(chunk_weight, dim=1)

        for i in range(num_chunks):
            start_idx = chunk_indices[i]
            end_idx = start_idx + chunk_size if start_idx + chunk_size < seq_len else seq_len
            posterior[:, start_idx:end_idx, :] = chunk_posterior[:, i:i+1, :]
        posterior /= posterior.sum(axis=-1, keepdims=True)
        print("posterior", posterior.size(), posterior[0, ...])
        return posterior
        
    def forward(self, logits, returns, action_mask, values, return_acc):
        logits = logits[action_mask]
        returns_ = returns[action_mask]
        labels = returns_.type(torch.LongTensor).to("cuda")
        if self.ensemble:
            ensemble_logits = logits.view(logits.size(0), -1, 2)
            probs = F.softmax(ensemble_logits, dim=-1)
            num_ensembles = ensemble_logits.size(1)
        #    """
            labels_expanded = labels.unsqueeze(-1).expand(-1, num_ensembles).unsqueeze(-1)
            loss = -torch.log(torch.gather(probs, -1, labels_expanded)).squeeze(-1)
            posterior = self.weight_update(values, returns).detach()
            posterior = posterior[action_mask]
          #  posterior += torch.ones_like(posterior) * 0.1
            loss = (loss * posterior).sum() / posterior.size(0)
            """
            num_masked = int(num_ensembles * self.mask_prob)
            random_mask = torch.zeros(num_ensembles, dtype=torch.bool, device=logits.device)
            random_mask[:num_masked] = True
            random_mask = random_mask[torch.randperm(num_ensembles)]
            labels_expanded = labels.unsqueeze(-1).expand(-1, num_ensembles).unsqueeze(-1)
            loss = -torch.log(torch.gather(probs, -1, labels_expanded)).squeeze(-1)
            masked_loss = loss * random_mask.view(1, -1)
            loss = masked_loss.sum() / (random_mask.sum() * logits.size(0))
            """
            acc = (ensemble_logits.argmax(dim=-1) == labels_expanded.squeeze(-1)).float().mean()
        else:
            loss = self.loss(logits, labels)
            acc = (logits.argmax(dim=-1) == labels).float().mean()
        if not return_acc:
            return loss
        return loss, acc

    
class CEValueLoss(nn.Module):
    def __init__(self, ensemble=True, mask_prob=0.0, bnn=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.ensemble = ensemble
        self.mask_prob = mask_prob
        self.bnn = bnn

    def kl_divergence(self, logvar):
      #  prior_var = 1.0 #0.02
      #  return -torch.mean(logvar - torch.exp(logvar) / prior_var - mu**2)
        logvar = logvar.float()
      #  mu_b = mu_b.float()
        var_p = torch.exp(logvar)
        var_q = 1.0
#        kl = 0.5 * (torch.log(var_q / var_p) + (var_p + (mu)**2) / var_q - 1)
        kl = -logvar + (var_p) / var_q - 1 # + (mu_a)**2
      #  kl = mu ** 2
      #  mu_a_sq = torch.mean((mu_a)**2) 
      #  mu_b_sq = torch.mean((mu_b)**2)
        return torch.mean(kl)#, mu_a_sq, mu_b_sq

    def bayesian_loss(self, pred, target, model):
        likelihood = F.cross_entropy(pred, target)
        #for name, param in model.named_parameters():
         #   if 'lora_B_logvar' in name:
         #       print("lora_B_logvar", name, param)
        kl = sum(
            self.kl_divergence(param)#, model.get_parameter(name.replace('_logvar.default', '.default.weight')))#[0]
            for name, param in model.named_parameters()
            if 'lora_A_logvar' in name or 'lora_B_logvar' in name
        )
        return likelihood + 0.0005 * kl, kl.detach(), 0.0, 0.0# mu_a_sq.detach(), mu_b_sq.detach()

    def forward(self, logits, returns, action_mask, values, return_acc):
        logits = logits[action_mask]
        returns_ = returns[action_mask]

        #chunk_size = 16
        #chunk_indices = torch.arange(0, returns_.size(-1), chunk_size)
        #logits = logits[chunk_indices]
        #returns_ = returns_[chunk_indices]

        labels = returns_.type(torch.LongTensor).to("cuda")
        if self.ensemble:
            loss, kl, mu_a_sq, mu_b_sq = self.bayesian_loss(logits, labels, self.bnn)   #self.loss(logits, labels)
        else:
            loss = self.loss(logits, labels)
            kl = 0.0
            mu_a_sq, mu_b_sq = 0.0, 0.0
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        if not return_acc:
            return loss
        return loss, acc, kl, mu_a_sq, mu_b_sq
    

class ValueLossOurs(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps=None, ensemble=True, bnn=None) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.ensemble = ensemble
        self.bnn = bnn

    def kl_divergence(self, logvar):
      #  prior_var = 1.0 #0.02
      #  return -torch.mean(logvar - torch.exp(logvar) / prior_var - mu**2)
        logvar = logvar.float()
      #  mu_b = mu_b.float()
        var_p = torch.exp(logvar)
        var_q = 1.0
#        kl = 0.5 * (torch.log(var_q / var_p) + (var_p + (mu)**2) / var_q - 1)
        kl = -logvar + (var_p) / var_q - 1 # + (mu_a)**2
      #  kl = mu ** 2
      #  mu_a_sq = torch.mean((mu_a)**2) 
      #  mu_b_sq = torch.mean((mu_b)**2)
        return torch.mean(kl)#, mu_a_sq, mu_b_sq
    
    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.clip_eps is not None:
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            surr1 = (values_clipped - returns) ** 2
            surr2 = (values - returns) ** 2
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2
        loss = masked_mean(loss, action_mask, dim=-1).mean() * 0.5
        if self.ensemble:
            kl = sum(
                self.kl_divergence(param)#, model.get_parameter(name.replace('_logvar.default', '.default.weight')))#[0]
                for name, param in self.bnn.named_parameters()
                if 'lora_A_logvar' in name or 'lora_B_logvar' in name
            )
            loss += 0.0005 * kl
            kl = kl.detach()
        else:
            kl = 0.0
        return loss,  kl
