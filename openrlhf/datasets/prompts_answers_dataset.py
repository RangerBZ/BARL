from torch.utils.data import Dataset
from tqdm import tqdm
from openrlhf.utils.verifier import extract_boxed_content

def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None, prm_step_separator='') -> str:
    if apply_chat_template:
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{{}}."
        chat = [{"content": system_prompt, "role": "system"}, {"content": data[input_key], "role": "user"}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
    return prompt

class PromptAnswerDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        output_key = getattr(self.strategy.args, "output_key", None)
        candidate_key = getattr(self.strategy.args, "candidate_key", None)
        prm_step_separator = getattr(self.strategy.args, "prm_step_separator", '')
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        self.answers = []
        self.candidates = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(data, input_template, input_key, apply_chat_template, prm_step_separator)
            answer = data[output_key]
            self.prompts.append(prompt)
            self.answers.append(answer)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return (self.prompts[idx], self.answers[idx])
    
    def collate_fn(self, item_list):
        # item_list: [(prompt_1, answer_1), (prompt_2, answer_2), ...]
        collated = []
        for prompt, answer in item_list:
            collated.append([prompt, answer])
        # collated: [[prompt_1, answer_1], [prompt_2, answer_2], ...]
        return collated

