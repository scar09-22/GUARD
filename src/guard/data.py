"""Dataset loading and leakage-aware splitting for the GUARD validity study.

GUARD's certificate (see conformal.py) is valid only when calibration humans and test humans
are exchangeable. Whether that holds is largely decided HERE, at the data layer:

- `load_maide_english` builds the MAiDE-up English hotel-review frame (1000 human / 1000 AI
  reviews over 100 hotels), the in-domain corpus for E1/E4/E5/E6.
- `grouped_split` partitions BY HOTEL so no entity spans calibration and test — the honest,
  deployment-faithful protocol (E5's grouped arm).
- `random_split` is the leaky comparator: plain stratified row split, allowing the same
  hotel's reviews on both sides (E5's leaky arm).
- `load_raid` reads the cached RAID pools (reviews-domain generators for E2/E7, abstracts for
  the E3 human-domain-shift probe) and degrades gracefully when a cache file is absent.

No experimental results are computed here; this module only produces frames with columns
the rest of the pipeline relies on: text, label (0 = human, 1 = AI), and grouping metadata.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "load_maide_english",
    "grouped_split",
    "random_split",
    "load_raid",
    "humans",
    "ais",
]


def _normalise_whitespace(s: str) -> str:
    """Collapse all runs of whitespace to single spaces and strip the ends."""
    return " ".join(str(s).split())


def load_maide_english(csv_path: str | Path = "data/all_data.csv") -> pd.DataFrame:
    """Load the English slice of MAiDE-up as a tidy frame [text, label, hotel, city].

    - Keeps rows with Review_Language == "English" only.
    - text = Upside_Review + " " + Downside_Review (missing halves treated as empty,
      whitespace-normalised); rows whose combined text is empty are dropped.
    - label = int(source): 0 = human-written, 1 = AI-generated.
    - hotel / city retained for entity-disjoint (grouped) calibration splits.
    """
    df = pd.read_csv(csv_path)
    df = df[df["Review_Language"] == "English"].copy()

    up = df["Upside_Review"].fillna("").astype(str)
    down = df["Downside_Review"].fillna("").astype(str)
    text = (up + " " + down).map(_normalise_whitespace)

    out = pd.DataFrame(
        {
            "text": text,
            "label": df["source"].astype(int),
            "hotel": df["Hotel Name"].astype(str),
            "city": df["City Name"].astype(str),
        }
    )
    out = out[out["text"].str.len() > 0].reset_index(drop=True)
    return out


def grouped_split(
    df: pd.DataFrame, test_frac: float = 0.5, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entity-disjoint split: partition the HOTELS, not the rows.

    Returns (df_a, df_b) where df_b's hotels make up ~test_frac of all hotels and no hotel
    appears on both sides. This mirrors deployment on a review platform: the humans you
    calibrate on are never the same entities you later judge. Used as the honest arm of the
    E5 leakage experiment and as the default protocol elsewhere.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")
    hotels = np.array(sorted(df["hotel"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(hotels)
    n_test = max(1, int(round(test_frac * hotels.size)))
    test_hotels = set(hotels[:n_test])
    mask_b = df["hotel"].isin(test_hotels)
    df_a = df[~mask_b].copy().reset_index(drop=True)
    df_b = df[mask_b].copy().reset_index(drop=True)
    return df_a, df_b


def random_split(
    df: pd.DataFrame, test_frac: float = 0.5, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leaky comparator: stratified ROW split ignoring entities.

    Rows are shuffled within each label and ~test_frac of each label goes to df_b, so the
    same hotel's reviews can (and do) land on both sides. Quantifying the optimism this
    induces in the certificate is the point of E5.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")
    rng = np.random.default_rng(seed)
    test_idx: list[np.ndarray] = []
    for _, grp in df.groupby("label"):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        n_test = max(1, int(round(test_frac * idx.size)))
        test_idx.append(idx[:n_test])
    mask_b = df.index.isin(np.concatenate(test_idx))
    df_a = df[~mask_b].copy().reset_index(drop=True)
    df_b = df[mask_b].copy().reset_index(drop=True)
    return df_a, df_b


def load_raid(parquet_path: str | Path) -> pd.DataFrame | None:
    """Load a cached RAID pool as [text, label, model], or None if the cache is missing.

    The reviews-domain pool (results/cache/raid_reviews_pool.parquet) may not have been
    built yet; callers must treat None as "skip the RAID arms of this experiment".
    label: 0 = human, 1 = AI; model names the generator ("human" for human rows).
    """
    path = Path(parquet_path)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    out = df[["text", "label", "model"]].copy()
    out["text"] = out["text"].astype(str).map(_normalise_whitespace)
    out["label"] = out["label"].astype(int)
    out = out[out["text"].str.len() > 0].reset_index(drop=True)
    return out


def humans(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of the human-written rows (label == 0)."""
    return df[df["label"] == 0].copy().reset_index(drop=True)


def ais(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of the AI-generated rows (label == 1)."""
    return df[df["label"] == 1].copy().reset_index(drop=True)
