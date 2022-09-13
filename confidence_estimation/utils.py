from torchmetrics.functional import sacre_bleu_score, rouge_score


def sacrebleu(pred: str, label: str) -> float:
    return sacre_bleu_score(preds=[pred], target=[[label]], lowercase=True).item()


def rouge(pred: str, label: str, rouge_key: str = "rougeL") -> float:
    return rouge_score(preds=pred, target=label, rouge_keys=rouge_key)[f"{rouge_key}_fmeasure"].item()

