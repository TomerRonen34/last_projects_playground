from pathlib import Path
from typing import Optional, Union

import bert_score
import jsonlines
import numpy as np
import torch
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
LABEL_PAD = -100


class MBart50Translator:
    """
    python -m mbart50_translate --device="cpu" --num_examples=10 --batch_size=2 --data_dir="data/wmt20" --dump_dir="mbart50_dumps" --src_lang="en" --tgt_lang="de"
    GPUS=1 run_with_slurm gen $(which run_python_script) -m mbart50_translate  --num_examples=10 --data_dir="data/wmt20" --dump_dir="mbart50_dumps" --src_lang="en" --tgt_lang="ru"
    GPUS=1 run_with_slurm gen $(which run_python_script) -m mbart50_translate  --is_backtranslation=True  --num_examples=10 --data_dir="data/wmt20" --dump_dir="mbart50_dumps" --src_lang="en" --tgt_lang="ta"
    python -m mbart50_translate  --device=cpu  --is_backtranslation=True  --num_examples=10 --data_dir="data/wmt20" --dump_dir="mbart50_dumps" --src_lang="en" --tgt_lang="ta"
    """

    def __init__(self,
                 data_dir: str,
                 src_lang: str,
                 tgt_lang: str,
                 is_backtranslation: bool = False,
                 dump_dir: str = "mbart50_dumps",
                 num_examples: Optional[int] = 200,
                 batch_size: int = 4,
                 bertscore_batch_size: Optional[int] = None,
                 device: str = "cuda:0",
                 dataset_name: str = "wmt20",
                 num_beams: int = 5,
                 num_beam_groups: int = 1,
                 diversity_penalty: float = 0.5,  # only matters if num_beam_groups != 1
                 max_output_to_input_ratio: float = 2.,
                 length_penalty: float = 1.0,
                 generate_multiple_options: Union[str, bool] = True,
                 ) -> None:
        assert "en" in (src_lang, tgt_lang)
        assert {src_lang, tgt_lang}.issubset(
            LANG_CODE_TO_FAIRSEQ_FORMAT.keys())

        self.is_backtranslation = is_backtranslation
        self.orig_dump_path = None
        self.key_prefix = ''
        if self.is_backtranslation:
            _, self.orig_dump_path = self._build_paths(
                dump_dir, dataset_name, src_lang, tgt_lang, num_examples, is_backtranslation=False)
            assert self.orig_dump_path.exists(), self.orig_dump_path
            src_lang, tgt_lang = tgt_lang, src_lang
            self.key_prefix = "bt_"

        self.data_dir = Path(data_dir)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.bertscore_batch_size = bertscore_batch_size if bertscore_batch_size is not None else batch_size
        self.device = device
        self.dataset_name = dataset_name
        self.max_output_to_input_ratio = max_output_to_input_ratio

        self.generations_dump_path, self.metrics_dump_path = self._build_paths(
            dump_dir, dataset_name, src_lang, tgt_lang, num_examples, is_backtranslation)

        self.num_return_sequences = num_beams if generate_multiple_options else 1
        self.generation_kwargs = {
            "num_beams": num_beams, "num_return_sequences": self.num_return_sequences, "length_penalty": length_penalty,
            "num_beam_groups": num_beam_groups, "diversity_penalty": diversity_penalty}

        model_name = "facebook/mbart-large-50-many-to-one-mmt" if tgt_lang == "en" else "facebook/mbart-large-50-one-to-many-mmt"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tgt_lang != "en":
            self.generation_kwargs["forced_bos_token_id"] = self.tokenizer.lang_code_to_id[LANG_CODE_TO_FAIRSEQ_FORMAT[self.tgt_lang]]

        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)

    @staticmethod
    def _build_paths(dump_dir: str, dataset_name: str, src_lang: str, tgt_lang: str, num_examples: int, is_backtranslation: bool) -> tuple[str, str]:
        dump_dir = Path(dump_dir)
        bt_suffix = '' if not is_backtranslation else "_backtranslation"
        lang_pair = f"{src_lang}-{tgt_lang}" if not is_backtranslation else f"{tgt_lang}-{src_lang}"
        base_name = f"{dataset_name}_{lang_pair}_{num_examples}examples{bt_suffix}"
        generations_dump_path = dump_dir / \
            f"{base_name}_only_generations.jsonl"
        metrics_dump_path = dump_dir / f"{base_name}.jsonl"
        return generations_dump_path, metrics_dump_path

    def __call__(self) -> None:
        self._run_generation()
        self._run_metrics_calculation()

    def _run_generation(self) -> None:
        assert not self.generations_dump_path.exists()
        dataset = self._prepare_dataset()
        batches = batch_indices(len(dataset), self.batch_size)
        for indices in tqdm(batches, desc="generating"):
            batch = dataset[indices]
            input_ids, attention_mask, labels = self._prepare_batch(batch)
            gen_results = self._generate(input_ids, attention_mask)
            labels_logprob_results = self._calculate_labels_logprob(
                input_ids, attention_mask, labels)
            to_dump = {
                **self._inflate_batch(batch), **gen_results, **self._inflate_batch(labels_logprob_results)}
            self._append_to_jsonlines_file(to_dump, self.generations_dump_path)

    def _inflate_batch(self, batch: dict) -> dict:
        batch_len = len(next(iter(batch.values())))
        repeat_inds = np.repeat(np.arange(batch_len),
                                self.num_return_sequences)
        inflated_batch = {k: [v[i] for i in repeat_inds]
                          for k, v in batch.items()}
        return inflated_batch

    def _generate(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, list]:
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
        gen_results = {f"{self.key_prefix}gen_sequence": sequences,
                       f"{self.key_prefix}gen_logprob": scores,
                       f"{self.key_prefix}gen_text": texts}
        return gen_results

    def _calculate_labels_logprob(self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor) -> dict[str, list]:
        decoder_input_ids = torch.where(
            labels != LABEL_PAD, labels, self.tokenizer.pad_token_id)
        start_of_generation_eos_column = self.tokenizer.eos_token_id * \
            labels.new_ones((labels.shape[0], 1))
        decoder_input_ids = torch.concat(
            [start_of_generation_eos_column, decoder_input_ids], dim=1)
        decoder_attention_mask = (
            decoder_input_ids != self.tokenizer.pad_token_id).int()

        label_pad_column = LABEL_PAD * labels.new_ones((labels.shape[0], 1))
        labels_for_loss = torch.concat(
            [label_pad_column, labels[:, 1:], label_pad_column], dim=1)
        forward_out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                 decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask,
                                 labels=labels_for_loss)

        logprobs = forward_out.logits.log_softmax(dim=-1)
        # without forced bos token, without extra predicted token at the end
        logprobs = logprobs[:, 1:-1]
        # without eos that starts generation, without forced bos token
        labels = decoder_input_ids[:, 2:].unsqueeze(-1)
        labels_mask = (labels != self.tokenizer.pad_token_id)
        token_logprobs = logprobs.gather(index=labels, dim=-1)
        token_logprobs = torch.where(
            labels_mask, token_logprobs, logprobs.new([0.])).squeeze(-1)
        sequence_logprobs = token_logprobs.sum(
            dim=-1) / labels_mask.squeeze(-1).sum(dim=-1)
        manual_loss = -token_logprobs.sum() / labels_mask.sum()
        assert torch.allclose(forward_out.loss, manual_loss)
        labels_logprob_results = {
            f"{self.key_prefix}labels_logprob": sequence_logprobs.tolist()}
        return labels_logprob_results

    def _prepare_batch(self, batch: dict) -> tuple[Tensor, Tensor, Tensor]:
        keys = ["input_ids", "attention_mask", "labels"]
        model_inputs = {key: batch[f"{self.key_prefix}{key}"] for key in keys}
        model_inputs = self.data_collator(
            dict_of_lists_to_list_of_dicts(model_inputs))
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        input_ids, attention_mask, labels = [model_inputs[k] for k in keys]
        return input_ids, attention_mask, labels

    def _append_to_jsonlines_file(self, to_dump: dict[list], path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        to_dump = dict_of_lists_to_list_of_dicts(to_dump)
        with jsonlines.open(path, 'a') as jsonl_writer:
            jsonl_writer.write_all(to_dump)

    def _prepare_dataset(self) -> Dataset:
        if not self.is_backtranslation:
            return self._prepare_wmt_dataset()
        else:
            return self._prepare_backtranslation_dataset()

    def _prepare_wmt_dataset(self, seed: int = 10) -> Dataset:
        src_sentences, tgt_sentences = self._load_sentences()
        dataset = Dataset.from_dict(
            {"src_sentence": src_sentences, "tgt_sentence": tgt_sentences, "id": range(len(src_sentences))})
        perm = np.random.RandomState(seed).permutation(len(dataset))
        if self.num_examples is not None:
            perm = perm[:self.num_examples]
        dataset = dataset.select(perm)
        self.tokenizer.src_lang = LANG_CODE_TO_FAIRSEQ_FORMAT[self.src_lang]
        dataset = dataset.map(self.tokenizer,
                              batched=True, input_columns=["src_sentence"])
        self.tokenizer.src_lang = LANG_CODE_TO_FAIRSEQ_FORMAT[self.tgt_lang]
        dataset = dataset.map(lambda text: {"labels": self.tokenizer(text)["input_ids"]},
                              batched=True, input_columns=["tgt_sentence"])
        return dataset

    def _load_sentences(self) -> tuple[list[str], list[str]]:
        base_path = self.data_dir / \
            f"{self.dataset_name}.{self.src_lang}-{self.tgt_lang}"
        src_sentences = read_lines(str(base_path) + ".src")
        tgt_sentences = read_lines(str(base_path) + ".ref")
        return src_sentences, tgt_sentences

    def _prepare_backtranslation_dataset(self) -> Dataset:
        dataset = Dataset.from_json(str(self.orig_dump_path))
        input_ids = dataset["gen_sequence"]
        attention_mask = [[1]*len(x) for x in input_ids]
        labels = dataset["input_ids"]
        dataset = dataset.add_column(f"{self.key_prefix}input_ids", input_ids)
        dataset = dataset.add_column(
            f"{self.key_prefix}attention_mask", attention_mask)
        dataset = dataset.add_column(f"{self.key_prefix}labels", labels)
        return dataset

    def _run_metrics_calculation(self) -> None:
        assert not self.metrics_dump_path.exists()
        if hasattr(self, "model"):
            del self.model
        if not hasattr(self, "bert_scorer"):
            self.bert_scorer = self._load_bert_scorer()

        gen_dataset = Dataset.from_json(str(self.generations_dump_path))

        batches = batch_indices(len(gen_dataset), self.bertscore_batch_size)
        target_text_column = "tgt_sentence" if not self.is_backtranslation else "src_sentence"
        for indices in tqdm(batches, desc="calculating metrics"):
            batch = gen_dataset[indices]
            gen_texts, tgt_sentences = batch[f"{self.key_prefix}gen_text"], batch[target_text_column]
            precisions, recalls, f1s = self.bert_scorer.score(
                cands=gen_texts, refs=tgt_sentences)
            metrics_results = {f"{self.key_prefix}bertscore_f1": f1s.tolist(),
                               f"{self.key_prefix}bertscore_precision": precisions.tolist(),
                               f"{self.key_prefix}bertscore_recall": recalls.tolist()}
            metrics_results[f"{self.key_prefix}bleu_score"] = [
                sacrebleu(pred=gen, label=tgt) for gen, tgt in zip(gen_texts, tgt_sentences)]
            for rouge_key in ("rougeL", "rouge2", "rouge1"):
                rouge_scores = [rouge(pred=gen, label=tgt, rouge_key=rouge_key)
                                for gen, tgt in zip(gen_texts, tgt_sentences)]
                metrics_results[f"{self.key_prefix}{rouge_key}_score"] = rouge_scores
            to_dump = {**batch, **metrics_results}
            self._append_to_jsonlines_file(to_dump, self.metrics_dump_path)

        self.generations_dump_path.unlink()

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
