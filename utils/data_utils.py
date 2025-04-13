import os
import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):

    def __init__(self, manifest_path: str):
        super().__init__()
        self.entries = []

        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                parts = line.split('\t')

                if len(parts) != 3:
                    continue
                feat_path, ids_str, transcript = parts

                token_ids = list(map(int, ids_str.split()))
                self.entries.append((feat_path, token_ids, transcript))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):

        feat_path, token_ids, transcript = self.entries[idx]

        mel_spec = torch.load(feat_path)  # shape: [n_mels, T]

        mel_spec = mel_spec.T

        token_ids = torch.tensor(token_ids, dtype=torch.long)
        return mel_spec, token_ids, transcript
