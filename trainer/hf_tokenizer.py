import torch
from typing import Any
from torch import tensor
from functools import lru_cache
from transformers import AutoTokenizer, AutoModel


class HFTokenizer:
    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        # TODO: Check with other tokenizers

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    @lru_cache(maxsize=1024, typed=False)
    def get_model_output(self, text: str) -> Any:
        encoded_input = self.tokenizer(
            text,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        model_output = self.model(**encoded_input)
        pooled_representation = model_output.pooler_output
        result = pooled_representation[0].detach()
        return result

    def get_pooled_representation_str(self, text: str) -> tensor:
        model_output = self.get_model_output(text)
        return model_output

    def get_pooled_representation(self, token_list: list) -> tensor:
        model_output = self.get_model_output(" ".join(token_list))
        return model_output
