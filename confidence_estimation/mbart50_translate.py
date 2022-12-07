import math
from typing import Union
from pathlib import Path
from transformers.models.mbart50.tokenization_mbart50_fast import FAIRSEQ_LANGUAGE_CODES
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import Dataset
from typing import Optional
from tqdm import tqdm
import numpy as np
from torch import Tensor
import jsonlines
from fire import Fire

LANG_CODE_TO_FAIRSEQ_FORMAT = {
    long_language_code[:2]: long_language_code for long_language_code in FAIRSEQ_LANGUAGE_CODES}


class MBart50Translator:
    """
    python -m mbart50_translate --device="cpu" --num_examples=10 --batch_size=2 --data_dir="data/wmt20" --dump_dir="mbart50_dumps" --src_lang="en" --tgt_lang="de"
    GPUS=1 run_with_slurm gen $(which run_python_script) -m mbart50_translate --data_dir="data/wmt20" --dump_dir="mbart50_dumps" --src_lang="en" --tgt_lang="ru"
    """

    def __init__(self,
                 data_dir: str,
                 dump_dir: str,
                 src_lang: str,
                 tgt_lang: str,
                 num_examples: Optional[int] = 200,
                 batch_size: int = 4,
                 device: str = "cuda:0",
                 dataset_name: str = "wmt20",
                 num_beams: int = 5,
                 max_output_to_input_ratio: float = 2.,
                 length_penalty: float = 1.0,
                 generate_multiple_options: Union[str, bool] = "auto",
                 ) -> None:
        assert "en" in (src_lang, tgt_lang)
        assert {src_lang, tgt_lang}.issubset(
            LANG_CODE_TO_FAIRSEQ_FORMAT.keys())
        self.data_dir = Path(data_dir)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.device = device
        self.dataset_name = dataset_name
        self.max_output_to_input_ratio = max_output_to_input_ratio

        self.dump_path = Path(
            dump_dir) / f"{dataset_name}_{src_lang}-{tgt_lang}_{num_examples}examples.jsonl"
        assert not self.dump_path.exists()

        if generate_multiple_options == "auto":
            generate_multiple_options = (self.src_lang == "en")
        self.num_return_sequences = num_beams if generate_multiple_options else 1
        self.generation_kwargs = {
            "num_beams": num_beams, "num_return_sequences": self.num_return_sequences, "length_penalty": length_penalty}

        model_name = "facebook/mbart-large-50-many-to-one-mmt" if tgt_lang == "en" else "facebook/mbart-large-50-one-to-many-mmt"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tgt_lang == "en":
            self.tokenizer.src_lang = LANG_CODE_TO_FAIRSEQ_FORMAT[self.src_lang]
        else:
            self.generation_kwargs["forced_bos_token_id"] = self.tokenizer.lang_code_to_id[LANG_CODE_TO_FAIRSEQ_FORMAT[self.tgt_lang]]

        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)

    def __call__(self) -> None:
        dataset = self._prepare_dataset()
        num_batches = math.ceil(len(dataset) / self.batch_size)
        batches = np.array_split(np.arange(len(dataset)), num_batches)
        for batch_indices in tqdm(batches, desc="generating"):
            batch = dataset[batch_indices]
            gen_results = self._generate(batch)
            self._dump_batch(gen_results)
            # metrics = self._calculate_metrics(
            #     gen_results["gen_texts"], batch["tgt_sequence"])

    def _generate(self, batch: dict) -> dict[list]:
        input_ids, attention_mask = self._prepare_batch_for_generation(batch)
        gen_output = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            **self.generation_kwargs,
            max_new_tokens=int(
                self.max_output_to_input_ratio * input_ids.shape[1]),
            return_dict_in_generate=True,
            output_scores=True,
        )
        # drop the eos token that starts generation
        sequences = gen_output.sequences[:, 1:]
        sequences = [seq[seq != self.tokenizer.pad_token_id].tolist()
                     for seq in sequences]
        scores = gen_output.sequences_scores.tolist()
        texts = self.tokenizer.batch_decode(
            gen_output.sequences, skip_special_tokens=True)
        src_ids = np.repeat(batch["id"], self.num_return_sequences).tolist()
        gen_results = {"gen_sequence": sequences,
                       "gen_score": scores, "gen_text": texts, "src_id": src_ids}
        return gen_results

    def _prepare_batch_for_generation(self, batch: dict) -> tuple[Tensor, Tensor]:
        model_inputs = {k: batch[k] for k in ["input_ids", "attention_mask"]}
        model_inputs = self.data_collator(
            dict_of_lists_to_list_of_dicts(model_inputs))
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        input_ids, attention_mask = model_inputs["input_ids"], model_inputs["attention_mask"]
        return input_ids, attention_mask

    def _dump_batch(self, to_dump: dict[list]) -> None:
        self.dump_path.parent.mkdir(exist_ok=True, parents=True)
        to_dump = dict_of_lists_to_list_of_dicts(to_dump)
        with jsonlines.open(self.dump_path, 'a') as jsonl_writer:
            jsonl_writer.write_all(to_dump)

    def _prepare_dataset(self, seed: int = 10) -> Dataset:
        src_sentences, tgt_sentences = self._load_sentences()
        dataset = Dataset.from_dict(
            {"src_sentence": src_sentences, "tgt_sentence": tgt_sentences, "id": range(len(src_sentences))})
        perm = np.random.RandomState(seed).permutation(len(dataset))
        if self.num_examples is not None:
            perm = perm[:self.num_examples]
        dataset = dataset.select(perm)
        dataset = dataset.map(self.tokenizer, batched=True,
                              input_columns=["src_sentence"])
        return dataset

    def _load_sentences(self) -> tuple[list[str], list[str]]:
        base_path = self.data_dir / \
            f"{self.dataset_name}.{self.src_lang}-{self.tgt_lang}"
        src_sentences = read_lines(str(base_path) + ".src")
        tgt_sentences = read_lines(str(base_path) + ".ref")
        return src_sentences, tgt_sentences


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


if __name__ == "__main__":
    Fire(MBart50Translator)
