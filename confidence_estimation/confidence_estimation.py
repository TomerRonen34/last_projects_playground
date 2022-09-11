from openai.openai_object import OpenAIObject
import numpy as np
from confidence_estimation.confidence_estimation import extract_quoted_tokens
from utils import sacrebleu
import openai

DEUTCH_WRITE_A_SENTENCE = "Schreibe einen Satz: "


def main(model_name: str = "text-babbage-001", num_examples: int = 10):
    openai_caller = OpenAICaller(model_name=model_name)

    bleu_score = sacrebleu(pred=predicted_text, label=label)
    return avg_input_logprob, avg_precited_logprob, bleu_score


class OpenAICaller:
    def __init__(self,
                 model_name: str = "text-babbage-001",
                 task_prompt: str = "Translate German to English: ",
                 echo_prompt: str = DEUTCH_WRITE_A_SENTENCE,
                 ):
        self.model_name = model_name
        self.task_prompt = task_prompt
        self.echo_prompt = echo_prompt

# input_text = "schauspieler orlando bloom und model miranda kerr wollen künftig getrennte wege gehen."
# label = "actors orlando bloom and model miranda kerr want to go their separate ways."
    def predict(self, input_text: str) -> tuple[str, float, float]:
        inference_response = self._make_inference_call(input_text)
        echo_response = self._make_echo_call(input_text)
        pred_text, pred_perplexity = _process_response(inference_response)
        echoed_input_text, echo_perplexity = _process_response(echo_response)
        assert echoed_input_text == input_text
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
    tokens, logprobs = extract_quoted_tokens(response)
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
