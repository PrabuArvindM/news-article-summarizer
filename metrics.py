from rouge_score import rouge_scorer
from bert_score import score


def compute_scores(reference, generated):

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    rouge = scorer.score(reference, generated)["rougeL"].fmeasure

    P, R, F1 = score([generated], [reference], lang="en")

    bert = F1.mean().item()

    return rouge, bert