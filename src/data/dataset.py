import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class CoLADataset(Dataset):
    def __init__(self, path: str, tokenizer: AutoTokenizer):
        super().__init__()
        # Read in only label and sentence
        self.data = pd.read_csv(
            path,
            sep="\t",
            names=["x","label","y","sentence"],
            header=None,
        )[["label", "sentence"]]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        # Dynamic padding is when we pad the sentences to the
        # longest sentence at the time of batch creation.
        # https://www.youtube.com/watch?v=7q5NyFT8REg
        # We use fixed padding.
        tokenized = self.tokenizer.encode_plus(
            data["sentence"],
            return_tensors="pt",
            padding="max_length",
            max_length=15,
            truncation=True,
        )
        label = torch.tensor([data["label"]])
        return {
            "label": label,
            "sentence": tokenized,
        }
