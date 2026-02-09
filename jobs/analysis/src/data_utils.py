from __future__ import annotations

import pandas as pd


def clean_founders_number(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize non-numeric founders count values before numeric coercion."""
    df["0_founders_number"] = df["0_founders_number"].replace({"4+": "4", "Personne morale": None})
    return df
