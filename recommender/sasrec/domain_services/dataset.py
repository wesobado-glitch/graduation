import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from recommender.sasrec.domain_services.sequence_ops import pad_or_truncate


class SASRecDataset(Dataset):
    """
    Builds teacher-forcing sequences for SASRec training.

    For each user:
      input_seq  : padded sequence s_1 … s_{n-1}
      target_seq : padded sequence s_2 … s_n  (shifted by 1)
      neg_seq    : one random negative item per time step
    """

    def __init__(
        self,
        user_sequences: Dict[int, List[int]],
        num_items: int,
        max_len: int,
    ):
        self.users = list(user_sequences.keys())
        self.seqs = user_sequences
        self.num_items = num_items
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        user = self.users[idx]
        seq = self.seqs[user]

        input_seq = pad_or_truncate(seq[:-1], self.max_len)
        target_seq = pad_or_truncate(seq[1:], self.max_len)

        item_set = set(seq)
        neg_seq = []
        for _ in range(self.max_len):
            neg = random.randint(1, self.num_items)
            while neg in item_set:
                neg = random.randint(1, self.num_items)
            neg_seq.append(neg)

        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long),
            torch.tensor(neg_seq, dtype=torch.long),
        )
