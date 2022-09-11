from torchmetrics.functional import sacre_bleu_score


def sacrebleu(pred: str, label: str):
    return sacre_bleu_score(preds=[pred], target=[[label]], lowercase=True).item()
