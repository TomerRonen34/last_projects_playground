import math
import re
from pathlib import Path
from typing import Sequence, Union

import numpy as np
from torchmetrics.functional import rouge_score, sacre_bleu_score


def sacrebleu(*, pred: str, label: str) -> float:
    return sacre_bleu_score(preds=[pred], target=[[label]], lowercase=True).item()


def rouge(*, pred: str, label: str, rouge_key: str = "rougeL") -> float:
    return rouge_score(preds=pred, target=label, rouge_keys=rouge_key, normalizer=_text_normalizer
                       )[f"{rouge_key}_fmeasure"].item()


def _text_normalizer(text: str) -> str:
    """ the default behavior in rouge_score keeps only [a-z0-9] characters """
    return re.sub(r'\W', ' ', text.lower())


def read_lines(path: Union[Path, str]) -> list[str]:
    text = Path(path).read_text()
    lines = text.split('\n')
    while lines[-1] == '':
        lines = lines[:-1]
    return lines


def dict_of_lists_to_list_of_dicts(d: dict[list]) -> list[dict]:
    return [dict(zip(d.keys(), vals)) for vals in zip(*d.values())]


def flatten(nested_list: list[list]) -> list:
    return [item for sublist in nested_list for item in sublist]


def batch_indices(n: int, batch_size: int) -> Sequence[Sequence[int]]:
    num_batches = math.ceil(n / batch_size)
    batches = np.array_split(np.arange(n), num_batches)
    return batches
