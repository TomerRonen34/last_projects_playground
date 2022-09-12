import json
from pathlib import Path

import numpy as np
import openai
from fire import Fire
from openai.openai_object import OpenAIObject
from tqdm import tqdm
from time import sleep

from utils import sacrebleu

DEUTCH_WRITE_A_SENTENCE = "Schreibe einen Satz: "


def main(model_name: str = "text-curie-001", num_examples: int = 200):
    data_dir = Path(
        "/home/olab/tomerronen1/data/fairseq-mcrerank/dr_nmt_paper/parallel/text_data/clean_detok")
    dump_dir = Path(__file__).parent / "metrics"
    dump_dir.mkdir(exist_ok=True, parents=True)
    dump_path = dump_dir / \
        f"AAA_wmt19_de_to_en__{model_name}__{num_examples}_examples__metrics.json"

    english_sentences = _read_lines(data_dir / "valid.en")
    deutch_sentences = _read_lines(data_dir / "valid.de")

    openai_caller = OpenAICaller(model_name=model_name)

    metrics = {
        "i_example": [],
        "pred_perplexity": [],
        "echo_perplexity": [],
        "bleu_score": [],
    }
    example_indices = np.random.RandomState(seed=1337).permutation(
        len(english_sentences))[:num_examples]
    for i_example in tqdm(example_indices):
        pred_text, pred_perplexity, echo_perplexity = openai_caller.predict(
            deutch_sentences[i_example])
        sleep(5)
        bleu_score = sacrebleu(
            pred=pred_text, label=english_sentences[i_example])

        metrics["i_example"].append(int(i_example))
        metrics["pred_perplexity"].append(pred_perplexity)
        metrics["echo_perplexity"].append(echo_perplexity)
        metrics["bleu_score"].append(bleu_score)

    with open(dump_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def _read_lines(path: Path) -> list[str]:
    text = path.read_text()
    lines = text.strip().split('\n')
    return lines


class OpenAICaller:
    def __init__(self,
                 model_name: str = "text-babbage-001",
                 task_prompt: str = "Translate German to English: ",
                 echo_prompt: str = DEUTCH_WRITE_A_SENTENCE,
                 ):
        self.model_name = model_name
        self.task_prompt = task_prompt
        self.echo_prompt = echo_prompt

    def predict(self, input_text: str) -> tuple[str, float, float]:
        inference_response = self._make_inference_call(input_text)
        echo_response = self._make_echo_call(input_text)
        pred_text, pred_perplexity = _process_response(inference_response)
        echoed_input_text, echo_perplexity = _process_response(echo_response)
        return pred_text, pred_perplexity, echo_perplexity

    def _make_inference_call(self, input_text: str) -> OpenAIObject:
        max_tokens = len(input_text.split()) * 2
        inference_request_text = f'{self.task_prompt}"{input_text}"'
        inference_response = openai.Completion.create(
            model=self.model_name,
            prompt=inference_request_text,
            temperature=0, max_tokens=max_tokens, logprobs=1, echo=False)
        return inference_response

    def _make_echo_call(self, input_text: str) -> OpenAIObject:
        echo_request_text = f'{self.echo_prompt}"{input_text}"'
        echo_response = openai.Completion.create(
            model=self.model_name,
            prompt=echo_request_text,
            temperature=0, max_tokens=0, logprobs=1, echo=True)
        return echo_response


def _process_response(response: OpenAIObject) -> tuple[str, float]:
    tokens, logprobs = _extract_quoted_tokens(response)
    text = ''.join(tokens)
    avg_logprob = np.mean(logprobs)
    return text, avg_logprob


def _extract_quoted_tokens(response: OpenAIObject, drop_tokens_with_null_logprob=True) -> tuple[list[str], list[float]]:
    logprobs_object = response.choices[0].logprobs
    tokens = logprobs_object.tokens
    logprobs = logprobs_object.token_logprobs

    quote_inds = [i for i, token in enumerate(tokens) if '"' in token]
    num_quotes = len(quote_inds)
    if num_quotes >= 2:
        tokens = tokens[quote_inds[0] + 1: quote_inds[1]]
        logprobs = logprobs[quote_inds[0] + 1: quote_inds[1]]
    elif drop_tokens_with_null_logprob:
        tokens = tokens[1:]
        logprobs = logprobs[1:]

    return tokens, logprobs


if __name__ == "__main__":
    Fire(main)
