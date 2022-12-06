from typing import Union
from pathlib import Path


def _load_sentences(data_dir: Union[Path, str], src_lang: str, tgt_lang: str, dataset_name: str = "wmt20"
                   ) -> tuple[list[str], list[str]]:
    assert src_lang.islower() and len(src_lang) == 2
    assert tgt_lang.islower() and len(tgt_lang) == 2
    data_dir = Path(data_dir)
    base_path = data_dir / f"{dataset_name}.{src_lang}-{tgt_lang}"
    src_path = base_path.with_suffix(".src")
    src_sentences = _read_lines(src_path)
    tgt_path = base_path.with_suffix(".ref")
    tgt_sentences = _read_lines(tgt_path)
    return src_sentences, tgt_sentences


def _read_lines(path: Union[Path, str]) -> list[str]:
    text = path.read_text(path)
    lines = text.split('\n')
    while lines[-1] == '':
        lines = lines[:-1]
    return lines
