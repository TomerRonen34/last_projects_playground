from typing import Optional, Union
import jsonlines
from pathlib import Path

import json
import numpy as np
import openai
from fire import Fire
from openai.openai_object import OpenAIObject
from tqdm import tqdm
from time import sleep
import re

from utils import rouge, sacrebleu

# TODO: a single forward pass (echo while generating)

CURR_DIR = Path(__file__).parent
LANGUAGE_CODE_TO_NAME = {"de": "German", "en": "English",
                         "ru": "Russian", "tr": "Turkish", "fi": "Finnish"}

# "text-curie-001"
# "text-babbage-001"
# "text-davinci-002"


def main(model_name: str = "text-davinci-002", num_examples: int = 200, src_lang: str = None, tgt_lang: str = None,
         backtranslation_dir: Optional[str] = None):
    """
    python confidence_estimation.py --src_lang=en --tgt_lang=fi
    python confidence_estimation.py --src_lang=en --tgt_lang=tr
    python confidence_estimation.py --backtranslation_dir="/home/olab/tomerronen1/git_repos/last_projects_playground/confidence_estimation/openai_dump/wmt19_en_to_fi__text-davinci-002__200_examples"
    python confidence_estimation.py --backtranslation_dir="/home/olab/tomerronen1/git_repos/last_projects_playground/confidence_estimation/openai_dump/wmt19_fi_to_en__text-davinci-002__200_examples"
    python confidence_estimation.py --backtranslation_dir="/home/olab/tomerronen1/git_repos/last_projects_playground/confidence_estimation/openai_dump/wmt19_tr_to_en__text-davinci-002__200_examples"
    sleep 1800 && python confidence_estimation.py --backtranslation_dir="/home/olab/tomerronen1/git_repos/last_projects_playground/confidence_estimation/openai_dump/wmt19_en_to_tr__text-davinci-002__200_examples"
    """
    if backtranslation_dir is None:
        assert (src_lang is not None) and (tgt_lang is not None)
        data_dir = CURR_DIR / "wmt19_dev_clean" / \
            f"newstest2018-{src_lang}{tgt_lang}"
        dump_dir = CURR_DIR / "openai_dump" / \
            f"wmt19_{src_lang}_to_{tgt_lang}__{model_name}__{num_examples}_examples"
    else:
        if backtranslation_dir is not None:
            assert (src_lang is None) and (tgt_lang is None)
        data_dir, dump_dir, src_lang, tgt_lang = _prepare_backtranslation(backtranslation_dir)
    
    if not data_dir.exists():
        raise ValueError(f"dir {data_dir} does not exist.")

    metrics_dump_path = dump_dir / "metrics.jsonl"
    responses_dump_path = dump_dir / "responses.jsonl"
    sentences_dump_path = dump_dir / "sentences.jsonl"
    num_dumped = 0
    if metrics_dump_path.exists():
        with jsonlines.open(metrics_dump_path, 'r') as reader:
            metrics = list(reader.iter())
        num_dumped = len(metrics)
    if num_dumped == num_examples:
        print("\n\nAlready queried and dumped that data")
        return
    elif dump_dir.exists():
        raise ValueError(
            f"dump dir exists but contains only {num_dumped} out of {num_examples} examples - '{dump_dir}'")
    dump_dir.mkdir(exist_ok=False, parents=True)

    source_sentences = _read_lines(data_dir / f"valid.{src_lang}")
    target_sentences = _read_lines(data_dir / f"valid.{tgt_lang}")

    task_prompt = f"Translate {LANGUAGE_CODE_TO_NAME[src_lang]} to {LANGUAGE_CODE_TO_NAME[tgt_lang]}:"
    openai_caller = OpenAICaller(model_name, task_prompt)

    if len(source_sentences) > num_examples:
        example_indices = np.random.RandomState(seed=1337).permutation(
            len(source_sentences))[:num_examples]
    else:
        example_indices =  np.arange(len(source_sentences))

    for i_example in tqdm(example_indices):
        input_text = source_sentences[i_example]
        target_text = target_sentences[i_example]
        pred_text, pred_perplexity, echo_perplexity, response_json = openai_caller.predict(
            input_text)
        bleu_score = sacrebleu(
            pred=pred_text, label=target_text)
        rougeL_score = rouge(
            pred=pred_text, label=target_text, rouge_key="rougeL")
        rouge2_score = rouge(
            pred=pred_text, label=target_text, rouge_key="rouge2")
        metrics = {
            "i_example": int(i_example),
            "pred_perplexity": pred_perplexity,
            "echo_perplexity": echo_perplexity,
            "bleu_score": bleu_score,
            "rougeL_score": rougeL_score,
            "rouge2_score": rouge2_score,
        }
        sentences = {
            "i_example": int(i_example),
            "input_text": input_text,
            "target_text": target_text,
            "pred_text": pred_text,
        }
        response_json = {
            "i_example": int(i_example),
            "response_json": response_json,
        }
        with jsonlines.open(metrics_dump_path, 'a') as f:
            f.write(metrics)
        with jsonlines.open(responses_dump_path, 'a') as f:
            f.write(response_json)
        with jsonlines.open(sentences_dump_path, 'a') as f:
            f.write(sentences)
        sleep(4)


def _read_lines(path: Path) -> list[str]:
    text = path.read_text()
    lines = text.strip().split('\n')
    return lines


def _write_lines(lines: list[str], path: Path) -> None:
    lines = [line.replace('\n', ' ') for line in lines]
    text = '\n'.join(lines) + '\n'
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(text)


class OpenAICaller:
    def __init__(self,
                 model_name: str,
                 task_prompt: str,
                 ):
        self.model_name = model_name
        self.task_prompt = task_prompt

    def predict(self, input_text: str) -> tuple[str, float, float, dict]:
        inference_response = self._make_inference_call(input_text)
        response_json = json.loads(str(inference_response))
        pred_text, pred_perplexity, echo_perplexity = _process_multipart_response(
            inference_response)
        return pred_text, pred_perplexity, echo_perplexity, response_json

    def _make_inference_call(self, input_text: str) -> OpenAIObject:
        max_tokens = len(input_text.split()) * 2
        inference_request_text = f'{self.task_prompt}\n{input_text}\n'
        inference_response = openai.Completion.create(
            model=self.model_name,
            prompt=inference_request_text,
            temperature=0, max_tokens=max_tokens, logprobs=1,
            echo=True)
        return inference_response


def _process_multipart_response(response: OpenAIObject):
    echoed_input_text, echo_perplexity = _process_response(
        response, part="echo")
    pred_text, pred_perplexity = _process_response(response, part="pred")
    return pred_text, pred_perplexity, echo_perplexity


def _process_response(response: OpenAIObject, part: str) -> tuple[list[str], list[float]]:
    logprobs_object = response.choices[0].logprobs
    logprobs = logprobs_object.token_logprobs[1:]  # first logprob is null
    tokens = logprobs_object.tokens[1:]

    if part == "echo":
        i_start, i_end = _find_echo_boundary(tokens)
    elif part == "pred":
        i_start, i_end = _find_pred_boundary(tokens)
    else:
        raise ValueError(f"Unrecognized part '{part}'")

    tokens = tokens[i_start: i_end]
    logprobs = logprobs[i_start: i_end]

    text = ''.join(tokens).strip()
    avg_logprob = np.mean(logprobs)
    return text, avg_logprob


def _find_echo_boundary(tokens: list[str]) -> tuple[int, int]:
    i_first_newline, i_second_newline = np.flatnonzero(
        (np.array(tokens) == '\n'))[:2]
    i_start = i_first_newline + 1
    i_end = i_second_newline
    return i_start, i_end


def _find_pred_boundary(tokens: list[str]) -> tuple[int, int]:
    i_second_newline = np.flatnonzero((np.array(tokens) == '\n'))[1]
    i_start = i_second_newline + 1
    i_end_of_text = np.flatnonzero((np.array(tokens) == "<|endoftext|>"))
    if len(i_end_of_text) > 0:
        i_end = i_end_of_text[0]
    else:
        i_end = len(tokens)
    return i_start, i_end


def _prepare_backtranslation(backtranslation_dir: Union[str, Path]) -> tuple[Path, Path, str, str]:
    backtranslation_dir = Path(backtranslation_dir)
    assert backtranslation_dir.exists()

    orig_src_lang, orig_tgt_lang = re.findall("(\w\w)_to_(\w\w)", backtranslation_dir.name)[0]
    src_lang, tgt_lang = orig_tgt_lang, orig_src_lang
    data_dir = backtranslation_dir / "backtranslation" / "data"
    dump_dir = backtranslation_dir / "backtranslation" / "dump"

    with jsonlines.open(backtranslation_dir / "sentences.jsonl", 'r') as reader:
        sentences = list(reader.iter())
    pred_sentences, input_sentences = zip(*[(row["pred_text"], row["input_text"]) for row in sentences])
    _write_lines(pred_sentences, data_dir / f"valid.{orig_tgt_lang}")
    _write_lines(input_sentences, data_dir / f"valid.{orig_src_lang}")

    return data_dir, dump_dir, src_lang, tgt_lang


if __name__ == "__main__":
    Fire(main)
