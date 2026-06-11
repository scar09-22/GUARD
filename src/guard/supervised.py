"""Modern supervised detector scores for GUARD's scorer-agnostic certificate.

METHODOLOGY.md deliberately keeps the base scorers modest (GPT-2 statistics, TF-IDF) because
the conformal certificate is scorer-agnostic: validity depends only on human exchangeability,
never on detection quality. This module adds the OTHER end of the scorer spectrum so the
validity map (E1-E8) can be replicated with the detectors deployers actually use:

- ``TransformerScorer``: a fine-tuned ``AutoModelForSequenceClassification`` (default
  distilroberta-base), trained on the study's TRAIN split with a plain torch loop. Like
  ``guard.scores.TfidfScorer`` it is bound by the leakage contract — it MUST be fit on the
  entity-disjoint training half only (METHODOLOGY.md E5); fitting on any text that later
  appears in calibration or test invalidates the certificate experiments.
- ``OffShelfDetector``: a frozen, publicly released AI-text detector (default
  Hello-SimpleAI/chatgpt-detector-roberta) used zero-shot. No fitting, no leakage surface;
  it probes whether certificates wrapped around off-the-shelf detectors behave differently
  from certificates around in-domain-trained ones.

Both expose ``fit(texts, labels)`` / ``score(texts) -> np.ndarray`` matching the
``TfidfScorer`` interface, with scores oriented HIGHER = more AI-like: the difference
logit(AI class) - logit(human class). Logit differences are monotone in the softmax
probability but unbounded, giving the conformal layer a continuous score with few ties.
Empty or non-string texts score NaN; ``guard.conformal.conformal_pvalues`` maps NaN to
p = 1 (never accuse a text the scorer could not read).

Labels follow the project convention: 0 = human-written, 1 = AI-generated.

Determinism note: ``TransformerScorer.fit`` seeds torch before head initialisation and
drives batch shuffling from a fixed numpy generator, so runs are reproducible up to
backend-level non-determinism of the device (MPS/CUDA kernels are not bitwise stable).
Scores are NEVER cached on disk (unlike the GPT-2 statistics in ``guard.scores``): they
depend on the particular training fit, exactly like TF-IDF.

No experimental results are hard-coded anywhere in this module.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = ["TransformerScorer", "OffShelfDetector"]

# Fraction of total optimisation steps used for linear LR warmup (then linear decay to 0).
_WARMUP_FRAC: float = 0.1

# Inference batch size for the frozen off-the-shelf detector (its max_length of 512 makes
# batches ~2.7x heavier than the fine-tuned scorer's at 192; keep memory bounded on 8 GB).
_OFFSHELF_BATCH: int = 8


def _resolve_device(device: str) -> str:
    """Return `device` if usable, else fall back to CPU with a warning (MPS-aware)."""
    import torch

    if device.startswith("mps") and not torch.backends.mps.is_available():
        warnings.warn("MPS requested but unavailable; falling back to CPU", stacklevel=3)
        return "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA requested but unavailable; falling back to CPU", stacklevel=3)
        return "cpu"
    return device


def _scoreable_mask(texts: Sequence[Any]) -> np.ndarray:
    """Boolean mask of texts that are non-empty strings (everything else scores NaN)."""
    return np.array(
        [isinstance(t, str) and bool(t.strip()) for t in texts], dtype=bool
    )


class TransformerScorer:
    """Fine-tuned transformer detector score: logit(AI) - logit(human), higher = AI-like.

    LEAKAGE CONTRACT: fit on the entity-disjoint TRAIN split only (METHODOLOGY.md E5),
    exactly as for ``guard.scores.TfidfScorer``. Labels must be 0 = human, 1 = AI; with that
    convention the AI class is index 1 of the freshly initialised two-logit head, so the
    score orientation is fixed by construction (no label-order ambiguity, unlike the
    off-the-shelf detector below).

    All heavy imports and the model itself load lazily; constructing the object is free.
    """

    def __init__(
        self,
        model_name: str = "distilroberta-base",
        device: str = "mps",
        max_length: int = 192,
        epochs: int = 2,
        batch_size: int = 16,
        lr: float = 2e-5,
        seed: int = 0,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = int(max_length)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.seed = int(seed)
        self._torch = None
        self._tokenizer = None
        self._model = None

    # ------------------------------- internals ------------------------------- #

    def _build(self) -> None:
        """Load tokenizer + a FRESH two-logit model (head init seeded for reproducibility)."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = _resolve_device(self.device)
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        torch.manual_seed(self.seed)  # deterministic classification-head init
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )
        self._model.to(self.device)

    def _encode(self, texts: list[str]) -> dict:
        """Tokenize one batch (padding to the batch max keeps memory bounded)."""
        enc = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    # --------------------------------- API ----------------------------------- #

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "TransformerScorer":
        """Fine-tune on the training split: AdamW, linear warmup/decay, shuffled batches.

        Empty texts are dropped (they carry no signal and the tokenizer would emit
        special-tokens-only inputs). Prints the mean loss per epoch. The model is left in
        eval mode; subsequent ``score`` calls run under ``no_grad``.
        """
        y_all = np.asarray(labels, dtype=int)
        if set(np.unique(y_all)) - {0, 1}:
            raise ValueError("TransformerScorer labels must be 0 (human) / 1 (AI).")
        if len(texts) != y_all.size:
            raise ValueError(f"got {len(texts)} texts but {y_all.size} labels")
        keep = _scoreable_mask(texts)
        train_texts = [str(t) for t, k in zip(texts, keep) if k]
        y = y_all[keep]
        n_dropped = int((~keep).sum())
        if n_dropped:
            print(f"[guard.supervised] fit: dropped {n_dropped} empty texts", flush=True)
        if len(train_texts) == 0:
            raise ValueError("no non-empty training texts")

        self._build()
        torch = self._torch
        model = self._model
        model.train()

        n = len(train_texts)
        n_batches = (n + self.batch_size - 1) // self.batch_size
        total_steps = max(1, self.epochs * n_batches)
        warmup_steps = max(1, int(round(_WARMUP_FRAC * total_steps)))

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        rng = np.random.default_rng(self.seed)
        labels_t = torch.as_tensor(y, dtype=torch.long)

        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            losses: list[float] = []
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                enc = self._encode([train_texts[i] for i in idx])
                batch_labels = labels_t[idx].to(self.device)
                out = model(**enc, labels=batch_labels)
                optimizer.zero_grad(set_to_none=True)
                out.loss.backward()
                optimizer.step()
                scheduler.step()
                losses.append(float(out.loss.detach()))
            print(
                f"[guard.supervised] epoch {epoch + 1}/{self.epochs}: "
                f"mean loss {float(np.mean(losses)):.4f} ({len(losses)} batches)",
                flush=True,
            )

        model.eval()
        return self

    def score(self, texts: Sequence[str]) -> np.ndarray:
        """logit(AI) - logit(human) per text; batched, no_grad; NaN for empty texts."""
        if self._model is None:
            raise RuntimeError("TransformerScorer.score called before fit() or load().")
        torch = self._torch
        out = np.full(len(texts), np.nan, dtype=float)
        valid = np.flatnonzero(_scoreable_mask(texts))
        self._model.eval()
        with torch.no_grad():
            for start in range(0, valid.size, self.batch_size):
                idx = valid[start : start + self.batch_size]
                enc = self._encode([str(texts[i]) for i in idx])
                logits = self._model(**enc).logits.float().cpu().numpy()  # (B, 2)
                out[idx] = logits[:, 1] - logits[:, 0]
        return out

    def save(self, path: str | Path) -> None:
        """Persist the fine-tuned weights (state_dict only; architecture = `model_name`)."""
        if self._model is None:
            raise RuntimeError("TransformerScorer.save called before fit() or load().")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._torch.save(self._model.state_dict(), path)

    def load(self, path: str | Path) -> "TransformerScorer":
        """Rebuild the architecture from `model_name` and load a saved state_dict."""
        self._build()
        state = self._torch.load(Path(path), map_location="cpu")
        self._model.load_state_dict(state)
        self._model.to(self.device)
        self._model.eval()
        return self


class OffShelfDetector:
    """Frozen public AI-text detector, used zero-shot; score = logit(AI) - logit(human).

    Label-order check: released detectors do NOT agree on which class index means "AI"
    (some ship id2label = {0: 'Human', 1: 'ChatGPT'}, others the reverse, others generic
    'LABEL_0'/'LABEL_1'). At load time the model's ``config.id2label`` is inspected: an
    index whose label contains an AI-ish keyword (chatgpt / gpt / ai / machine / fake /
    generated / bot) is taken as the AI class and its complement as human; a label
    containing human / real fixes the human side likewise. If neither side can be
    identified (generic labels), index 1 is assumed AI — the HuggingFace convention —
    and a warning is emitted so the assumption is auditable.

    ``fit`` is a deliberate no-op: the detector is frozen by design (interface parity with
    ``TransformerScorer`` / ``TfidfScorer`` so experiment code can treat scorers uniformly,
    and zero leakage surface — nothing is ever trained on study text).
    """

    # Keyword sets for the id2label inspection (matched on lowercased label words/substrings).
    _AI_KEYS: tuple[str, ...] = ("chatgpt", "gpt", "machine", "fake", "generated", "bot")
    _AI_WORDS: frozenset[str] = frozenset({"ai"})  # exact-word only ('ai' is a common substring)
    _HUMAN_KEYS: tuple[str, ...] = ("human", "real")

    def __init__(
        self,
        model_name: str = "Hello-SimpleAI/chatgpt-detector-roberta",
        device: str = "mps",
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = int(max_length)
        self.ai_index: int | None = None  # set at load time from config.id2label
        self.human_index: int | None = None
        self._torch = None
        self._tokenizer = None
        self._model = None

    @classmethod
    def _classify_label(cls, label: str) -> str:
        """Map one id2label entry to 'ai', 'human', or 'unknown'."""
        low = str(label).lower()
        words = set("".join(c if c.isalnum() else " " for c in low).split())
        if any(k in low for k in cls._AI_KEYS) or (words & cls._AI_WORDS):
            return "ai"
        if any(k in low for k in cls._HUMAN_KEYS):
            return "human"
        return "unknown"

    def _resolve_label_order(self, id2label: dict) -> tuple[int, int]:
        """(ai_index, human_index) from config.id2label; fall back to (1, 0) with a warning."""
        kinds = {int(i): self._classify_label(lab) for i, lab in id2label.items()}
        ai = [i for i, k in kinds.items() if k == "ai"]
        hu = [i for i, k in kinds.items() if k == "human"]
        if len(ai) == 1 and (len(hu) == 1 or len(kinds) == 2):
            ai_idx = ai[0]
            hu_idx = hu[0] if len(hu) == 1 else next(i for i in kinds if i != ai_idx)
            return ai_idx, hu_idx
        if len(hu) == 1 and len(kinds) == 2:
            hu_idx = hu[0]
            return next(i for i in kinds if i != hu_idx), hu_idx
        warnings.warn(
            f"{self.model_name}: ambiguous id2label {dict(id2label)!r}; "
            "assuming index 1 = AI, index 0 = human (HuggingFace convention)",
            stacklevel=3,
        )
        return 1, 0

    def _ensure_loaded(self) -> None:
        """Lazy load: tokenizer, model (eval, frozen) and the AI/human index resolution."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = _resolve_device(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        self.ai_index, self.human_index = self._resolve_label_order(
            self._model.config.id2label
        )
        print(
            f"[guard.supervised] {self.model_name}: id2label="
            f"{dict(self._model.config.id2label)!r} -> AI index {self.ai_index}, "
            f"human index {self.human_index}",
            flush=True,
        )

    def fit(self, texts: Sequence[str] | None = None, labels: Sequence[int] | None = None
            ) -> "OffShelfDetector":
        """No-op (frozen detector); exists for interface parity with the trainable scorers."""
        return self

    def score(self, texts: Sequence[str]) -> np.ndarray:
        """logit(AI) - logit(human) per text; batched, no_grad; NaN for empty texts."""
        self._ensure_loaded()
        torch = self._torch
        out = np.full(len(texts), np.nan, dtype=float)
        valid = np.flatnonzero(_scoreable_mask(texts))
        with torch.no_grad():
            for start in range(0, valid.size, _OFFSHELF_BATCH):
                idx = valid[start : start + _OFFSHELF_BATCH]
                enc = self._tokenizer(
                    [str(texts[i]) for i in idx],
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self._model(**enc).logits.float().cpu().numpy()
                out[idx] = logits[:, self.ai_index] - logits[:, self.human_index]
        return out


if __name__ == "__main__":
    # Smoke test: 20 MAiDE English texts, 1 epoch, scores finite and better than chance on
    # the training data itself. OffShelfDetector is NOT exercised here (first use downloads
    # ~500 MB); its label-order resolution is unit-testable via _classify_label without I/O.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from sklearn.metrics import roc_auc_score

    from guard.data import load_maide_english

    root = Path(__file__).resolve().parents[2]
    df = load_maide_english(root / "data" / "all_data.csv")
    texts = (
        df[df["label"] == 0]["text"].head(10).tolist()
        + df[df["label"] == 1]["text"].head(10).tolist()
    )
    labels = [0] * 10 + [1] * 10

    scorer = TransformerScorer(epochs=1, batch_size=4, max_length=128)
    scorer.fit(texts, labels)
    s = scorer.score(texts)
    assert np.isfinite(s).all(), "non-finite scores on non-empty texts"
    auc = float(roc_auc_score(labels, s))
    print(f"[guard.supervised] smoke: train-set AUC = {auc:.3f}")
    assert auc > 0.5, f"training-set AUC {auc:.3f} not above chance"
    assert np.isnan(scorer.score([""]))[0], "empty text must score NaN"
    print("[guard.supervised] smoke test passed")
