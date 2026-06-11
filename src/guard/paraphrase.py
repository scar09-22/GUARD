"""LLM-paraphrase editor: the realistic strong 'innocent rewriting' / evasion proxy.

The lexical attacks in `guard.attacks` (function-word noise, WordNet synonyms, spell
cleanup) are cheap proxies for tool-assisted editing. This module supplies the STRONG
end of that spectrum: a sequence-to-sequence paraphraser
(humarin/chatgpt_paraphraser_on_T5_base, a T5-base fine-tuned on ChatGPT paraphrase
pairs) that rewrites a whole text the way a writing assistant would.

Role in the validity map (METHODOLOGY.md, Section 2):

- Applied to HUMAN test text it is a certificate attack — paraphrased humans are no
  longer exchangeable with the clean calibration humans, so the FPR <= alpha bound is
  at risk (E4's deployment-critical case, at full strength rather than lexical proxy).
- Applied to AI text it is an evasion attack — validity is untouched (calibration
  humans unchanged) but detection power may degrade.

Reproducibility contract: `paraphrase(text, seed)` seeds torch globally per call, so a
given (text, seed) pair maps to one output on a fixed (torch version, device) stack.
Because generation is by far the dominant cost, `paraphrase_many` persists every output
in a parquet cache keyed by sha256(f"{text}|{seed}") — each (text, seed) pair is
generated once ever, and all downstream experiments replay from the cache.

Failure policy: any generation exception or empty decode falls back to returning the
ORIGINAL text (a no-op edit, conservative for both attack arms) and increments
`n_failures`. Fallbacks are cached like ordinary outputs so experiment results stay
deterministic across runs; delete the cache row to force a retry.

No experimental results are hard-coded anywhere in this module.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Sequence

import pandas as pd

__all__ = ["T5Paraphraser"]

# Single column stored in the paraphrase cache.
_CACHE_COL = "paraphrase"


def _cache_key(text: str, seed: int) -> str:
    """Cache key for one (text, seed) generation: sha256 of 'text|seed'."""
    return hashlib.sha256(f"{text}|{seed}".encode("utf-8")).hexdigest()


def _load_cache(cache_path: Path) -> pd.DataFrame:
    """Load the paraphrase cache (index = sha256 of 'text|seed'); empty frame if absent."""
    if cache_path.exists():
        cache = pd.read_parquet(cache_path)
        if cache.index.name != "sha256" and "sha256" in cache.columns:
            cache = cache.set_index("sha256")
        return cache
    return pd.DataFrame(
        {_CACHE_COL: pd.Series([], dtype=str)}, index=pd.Index([], name="sha256")
    )


def _save_cache(cache: pd.DataFrame, cache_path: Path) -> None:
    """Atomically replace the cache file (write temp, then rename)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    cache.to_parquet(tmp_path)
    os.replace(tmp_path, cache_path)


class T5Paraphraser:
    """Sampling T5 paraphraser with per-call seeding and a persistent generation cache.

    The HuggingFace tokenizer and model are loaded lazily on first use, so constructing
    this object is free (important for scripts that may only hit the paraphrase cache).
    Inputs are truncated to `max_length` tokens and generated one at a time (batch of 1)
    to keep unified-memory pressure low on MPS.

    Attributes:
        n_failures: count of generations that raised or decoded to empty (the original
            text was returned for each).
    """

    def __init__(
        self,
        model_name: str = "humarin/chatgpt_paraphraser_on_T5_base",
        device: str = "mps",
        max_length: int = 256,
        num_beams: int = 1,
        do_sample: bool = True,
        top_p: float = 0.92,
        temperature: float = 1.1,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = int(max_length)
        self.num_beams = int(num_beams)
        self.do_sample = bool(do_sample)
        self.top_p = float(top_p)
        self.temperature = float(temperature)
        self.n_failures = 0
        self._torch = None
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        """Load tokenizer + model on first use (lazy; heavy imports stay out of module load)."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    def paraphrase(self, text: str, seed: int) -> str:
        """Paraphrase one text; deterministic per (text, seed) on a fixed torch/device stack.

        Seeds torch globally (torch.manual_seed covers CPU, CUDA and MPS generators),
        formats the input as ``"paraphrase: " + text`` per the model card, truncates the
        INPUT to `max_length` tokens, and samples one decoder output. On any exception
        or an empty decode, returns the ORIGINAL text and increments `n_failures`.
        """
        if not isinstance(text, str) or not text.strip():
            self.n_failures += 1
            return text
        try:
            self._ensure_loaded()
            torch = self._torch
            torch.manual_seed(int(seed))
            enc = self._tokenizer(
                "paraphrase: " + text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            gen_kwargs: dict[str, object] = {
                "max_length": self.max_length,
                "num_beams": self.num_beams,
                "do_sample": self.do_sample,
            }
            if self.do_sample:
                gen_kwargs["top_p"] = self.top_p
                gen_kwargs["temperature"] = self.temperature
            with torch.no_grad():
                out_ids = self._model.generate(**enc, **gen_kwargs)
            decoded = self._tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
            if not decoded:
                self.n_failures += 1
                return text
            return decoded
        except Exception:
            self.n_failures += 1
            return text

    def paraphrase_many(
        self,
        texts: Sequence[str],
        seeds: Sequence[int],
        cache_path: str | Path,
    ) -> list[str]:
        """Paraphrase many (text, seed) pairs through a persistent parquet cache.

        The cache at `cache_path` is keyed by sha256(f"{text}|{seed}") so each pair is
        generated ONCE ever, across processes and runs. Pairs already cached are read
        back without touching the model; missing pairs are generated one at a time
        (batch of 1), with a progress print every 25 generations and a periodic cache
        save every 100 (plus a final save). Returns outputs in input order; duplicate
        (text, seed) pairs within the call are generated once.
        """
        texts = [str(t) for t in texts]
        seeds = [int(s) for s in seeds]
        if len(texts) != len(seeds):
            raise ValueError(
                f"texts and seeds must have equal length, got {len(texts)} vs {len(seeds)}"
            )

        cache_path = Path(cache_path)
        cache = _load_cache(cache_path)
        keys = [_cache_key(t, s) for t, s in zip(texts, seeds)]
        cached_keys = set(cache.index) if _CACHE_COL in cache.columns else set()

        # Unique missing pairs, preserving first-occurrence order.
        missing: dict[str, tuple[str, int]] = {}
        for k, t, s in zip(keys, texts, seeds):
            if k not in cached_keys and k not in missing:
                missing[k] = (t, s)

        if missing:
            new_rows: dict[str, str] = {}
            for i, (k, (t, s)) in enumerate(missing.items()):
                if i % 25 == 0:
                    print(
                        f"[guard.paraphrase] {i}/{len(missing)} texts paraphrased "
                        f"(cache: {cache_path.name}, failures: {self.n_failures})",
                        flush=True,
                    )
                new_rows[k] = self.paraphrase(t, s)
                if (i + 1) % 100 == 0:
                    cache = self._merge_and_save(cache, new_rows, cache_path)
                    new_rows = {}
            cache = self._merge_and_save(cache, new_rows, cache_path)
            print(
                f"[guard.paraphrase] {len(missing)}/{len(missing)} texts paraphrased; "
                f"cache saved ({self.n_failures} failures).",
                flush=True,
            )

        lookup = cache[_CACHE_COL].to_dict()
        return [str(lookup[k]) for k in keys]

    @staticmethod
    def _merge_and_save(
        cache: pd.DataFrame, new_rows: dict[str, str], cache_path: Path
    ) -> pd.DataFrame:
        """Append `new_rows` to `cache`, dedupe (fresh rows win), and persist atomically."""
        if not new_rows:
            return cache
        new_df = pd.DataFrame(
            {_CACHE_COL: pd.Series(new_rows, dtype=str)}
        )
        new_df.index.name = "sha256"
        cache = pd.concat([cache, new_df])
        cache = cache[~cache.index.duplicated(keep="last")]
        _save_cache(cache, cache_path)
        return cache


if __name__ == "__main__":
    # Self-test (downloads ~850 MB model on first run); enable with GUARD_TEST_PARA=1.
    if os.environ.get("GUARD_TEST_PARA") == "1":
        paraphraser = T5Paraphraser()
        samples = [
            "The hotel room was clean and the staff were friendly.",
            "Breakfast was cold and the elevator was broken the whole stay.",
        ]
        for sample in samples:
            out = paraphraser.paraphrase(sample, seed=0)
            print(f"[guard.paraphrase] IN : {sample}")
            print(f"[guard.paraphrase] OUT: {out}")
        print(f"[guard.paraphrase] failures: {paraphraser.n_failures}")
    else:
        print("[guard.paraphrase] set GUARD_TEST_PARA=1 to run the model self-test.")
