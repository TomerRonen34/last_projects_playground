from functools import partial
from pathlib import Path
from typing import Optional, Union

import bert_score
import jsonlines
import numpy as np
from bert_score import BERTScorer
from datasets import Dataset
from fire import Fire
from torch import Tensor
from tqdm import tqdm
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq)
from transformers.models.mbart50.tokenization_mbart50_fast import \
    FAIRSEQ_LANGUAGE_CODES
from utils import (batch_indices, dict_of_lists_to_list_of_dicts, read_lines,
                   rouge, sacrebleu)

LANG_CODE_TO_FAIRSEQ_FORMAT = {
    long_language_code[:2]: long_language_code for long_language_code in FAIRSEQ_LANGUAGE_CODES}


class MBart50Translator:
    """
    python -m mbart50_translate --device="cpu" --num_examples=10 --batch_size=2 --data_dir="data/wmt20" --dump_dir="mbart50_dumps" --src_lang="en" --tgt_lang="de"
    GPUS=1 run_with_slurm gen $(which run_python_script) -m mbart50_translate  --num_examples=10 --data_dir="data/wmt20" --dump_dir="mbart50_dumps" --src_lang="en" --tgt_lang="ru"
    GPUS=1 run_with_slurm gen $(which run_python_script) -m mbart50_translate  --num_examples=10 --data_dir="data/wmt20" --dump_dir="mbart50_dumps" --src_lang="ru" --tgt_lang="en"
    """

    def __init__(self,
                 data_dir: str,
                 src_lang: str,
                 tgt_lang: str,
                 dump_dir: str = "mbart50_dumps",
                 num_examples: Optional[int] = 200,
                 batch_size: int = 4,
                 bertscore_batch_size: Optional[int] = None,
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
        self.bertscore_batch_size = bertscore_batch_size if bertscore_batch_size is not None else batch_size
        self.device = device
        self.dataset_name = dataset_name
        self.max_output_to_input_ratio = max_output_to_input_ratio

        self.generations_dump_path = Path(
            dump_dir) / f"{dataset_name}_{src_lang}-{tgt_lang}_{num_examples}examples.jsonl"
        self.metrics_dump_path = self.generations_dump_path.with_suffix(
            ".with_metrics.jsonl")

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
        self._run_generation()
        self._run_metrics_calculation()

    def _run_generation(self) -> None:
        assert not self.generations_dump_path.exists()
        dataset = self._prepare_dataset()
        batches = batch_indices(len(dataset), self.batch_size)
        for indices in tqdm(batches, desc="generating"):
            batch = dataset[indices]
            gen_results = self._generate(batch)
            inflated_batch = {k: np.repeat(
                v, self.num_return_sequences).tolist() for k, v in batch.items()}
            to_dump = {**inflated_batch, **gen_results}
            self._append_to_jsonlines_file(to_dump, self.generations_dump_path)

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
        gen_results = {"gen_sequence": sequences,
                       "gen_score": scores, "gen_text": texts}
        return gen_results

    def _prepare_batch_for_generation(self, batch: dict) -> tuple[Tensor, Tensor]:
        model_inputs = {k: batch[k] for k in ["input_ids", "attention_mask"]}
        model_inputs = self.data_collator(
            dict_of_lists_to_list_of_dicts(model_inputs))
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        input_ids, attention_mask = model_inputs["input_ids"], model_inputs["attention_mask"]
        return input_ids, attention_mask

    def _append_to_jsonlines_file(self, to_dump: dict[list], path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        to_dump = dict_of_lists_to_list_of_dicts(to_dump)
        with jsonlines.open(path, 'a') as jsonl_writer:
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

    def _run_metrics_calculation(self) -> None:
        assert not self.metrics_dump_path.exists()
        if hasattr(self, "model"):
            del self.model
        if not hasattr(self, "bert_scorer"):
            self.bert_scorer = self._load_bert_scorer()

        gen_dataset = Dataset.from_json(str(self.generations_dump_path))

        batches = batch_indices(len(gen_dataset), self.bertscore_batch_size)
        for indices in tqdm(batches, desc="calculating metrics"):
            batch = gen_dataset[indices]
            gen_texts, tgt_sentences = batch["gen_text"], batch["tgt_sentence"]
            precisions, recalls, f1s = self.bert_scorer.score(
                cands=gen_texts, refs=tgt_sentences)
            metrics_results = {"bertscore_f1": f1s.tolist(),
                               "bertscore_precision": precisions.tolist(),
                               "bertscore_recall": recalls.tolist()}
            metrics_results["bleu_score"] = [
                sacrebleu(pred=gen, label=tgt) for gen, tgt in zip(gen_texts, tgt_sentences)]
            for rouge_key in ("rougeL", "rouge2", "rouge1"):
                rouge_scores = [rouge(pred=gen, label=tgt, rouge_key=rouge_key)
                                for gen, tgt in zip(gen_texts, tgt_sentences)]
                metrics_results[f"{rouge_key}_score"] = rouge_scores
            to_dump = {**batch, **metrics_results}
            self._append_to_jsonlines_file(to_dump, self.metrics_dump_path)

    def _load_bert_scorer(self) -> BERTScorer:
        bertscore_baseline_languages = [path.name for path in (
            Path(bert_score.__file__).parent / "rescale_baseline").iterdir()]
        rescale_with_baseline = (self.tgt_lang in bertscore_baseline_languages)
        model_type = "microsoft/deberta-xlarge-mnli" if self.tgt_lang == "en" else None
        bert_scorer = BERTScorer(model_type=model_type, lang=self.tgt_lang,
                                 rescale_with_baseline=rescale_with_baseline, device=self.device)
        bert_scorer.score(['a', ], ['a'])  # make sure no exception is thrown
        return bert_scorer


if __name__ == "__main__":
    Fire(MBart50Translator)
