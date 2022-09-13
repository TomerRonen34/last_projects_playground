# WMT19 dev set:
# wget http://data.statmt.org/wmt19/translation-task/dev.tgz
# tar zxvf dev.tgz

from pathlib import Path
import ftfy


def main(subset_name: str = "newstest", year: int = 2018, src_lang: str = "ru", tgt_lang: str = "en"):
    curr_dir = Path(__file__).parent
    data_dir = curr_dir / "wmt19_dev"
    dump_dir = curr_dir / "wmt19_dev_clean"

    split_name = f"{subset_name}{year}-{src_lang}{tgt_lang}"
    src_path = data_dir / f"{split_name}-src.{src_lang}.sgm"
    tgt_path = data_dir / f"{split_name}-ref.{tgt_lang}.sgm"

    src_sentences = _read_sentence_file(src_path)
    tgt_sentences = _read_sentence_file(tgt_path)

    dump_dir = dump_dir / split_name
    dump_dir.mkdir(parents=True, exist_ok=True)

    _dump_sentence_file(src_sentences, src_lang, dump_dir)
    _dump_sentence_file(tgt_sentences, tgt_lang, dump_dir)


def _read_sentence_file(path: Path) -> list[str]:
    lines = path.read_text().split('\n')
    sentences = [_extract_sentence_from_line(
        line) for line in lines if _has_content(line)]
    return sentences


def _has_content(line: str) -> bool:
    return line.startswith("<seg id=")


def _extract_sentence_from_line(line: str) -> str:
    sentence = line.split('>')[1].split('<')[0]
    sentence = ftfy.fix_text(sentence)
    for bad_substring in ['«', '»']:
        sentence = sentence.replace(bad_substring, '')    
    return sentence


def _dump_sentence_file(sentences: list[str], lang: str, dump_dir: Path) -> None:
    dump_path = dump_dir / f"valid.{lang}"
    content = '\n'.join(sentences)
    dump_path.write_text(content)


if __name__ == "__main__":
    main()
