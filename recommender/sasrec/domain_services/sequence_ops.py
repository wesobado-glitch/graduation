from typing import Dict, List, Tuple


UserSequences = Dict[int, List[int]]


def pad_or_truncate(seq: List[int], max_len: int) -> List[int]:
    seq = seq[-max_len:]
    pad_len = max_len - len(seq)
    return [0] * pad_len + list(seq)


def split_sequences(
    user_sequences: UserSequences,
) -> Tuple[UserSequences, UserSequences, UserSequences]:
    """Leave-one-out split: last item for test, second-to-last for val."""
    train_seqs, val_seqs, test_seqs = {}, {}, {}
    for user, seq in user_sequences.items():
        if len(seq) < 3:
            continue
        train_seqs[user] = seq[:-2]
        val_seqs[user] = seq[:-1]
        test_seqs[user] = seq
    return train_seqs, val_seqs, test_seqs
