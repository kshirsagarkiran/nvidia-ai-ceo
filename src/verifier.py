"""Entailment-based verification of findings (BART-MNLI).

The corroboration heuristic counts how many sources cite a claim, but it can be fooled
by shared boilerplate (e.g. the legal language every SEC 8-K repeats). This verifier
checks *semantic* support instead: for each finding it asks a Natural Language Inference
model whether the cited evidence (premise) ENTAILS the claim (hypothesis). The entailment
probability then drives a more honest confidence score.

Model: facebook/bart-large-mnli. Its label order is
    0 = contradiction, 1 = neutral, 2 = entailment
"""
from __future__ import annotations

from typing import List, Dict

import config

ENTAILMENT_IDX = 2


class Verifier:
    def __init__(self, model_name: str = None):
        import torch                                              # lazy: only needed if enabled
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.torch = torch
        name = model_name or config.VERIFIER_MODEL
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name)
        self.model.eval()

    def entailment(self, premise: str, hypothesis: str) -> float:
        """P(premise entails hypothesis)."""
        inputs = self.tok(premise, hypothesis, return_tensors="pt",
                          truncation=True, max_length=512)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0]
        probs = self.torch.softmax(logits, dim=-1)
        return float(probs[ENTAILMENT_IDX])

    def verify(self, claim: str, evidence_texts: List[str]) -> Dict:
        """Score a claim against its cited evidence.

        We check entailment two ways and keep the strongest:
          * per-chunk: does any single chunk entail the claim? (good for literal claims)
          * combined : does the cited evidence *collectively* entail it? (good for claims
                       that aggregate several sources, e.g. "partnerships with A, B, C, D")
        Each chunk is trimmed before combining so several fit inside the model's 512-token
        window.
        """
        texts = [e for e in evidence_texts if e]
        if not texts:
            return {"entailment": 0.0}
        per_chunk = max(self.entailment(t, claim) for t in texts)
        combined_premise = " ".join(t[:200] for t in texts)        # let many sources fit
        combined = self.entailment(combined_premise, claim)
        return {"entailment": round(max(per_chunk, combined), 3)}


def blend_confidence(entailment: float, distinct_docs: int) -> float:
    """Confidence = 60% semantic entailment + 40% corroboration.

    Entailment asks 'does the evidence actually support this claim?'; corroboration asks
    'how many distinct *documents* back it?' (four different articles count more than one).
    """
    corroboration = min(1.0, 0.34 * distinct_docs)   # 1 doc -> 0.34, 2 -> 0.68, 3+ -> 1.0
    return round(0.6 * entailment + 0.4 * corroboration, 2)
