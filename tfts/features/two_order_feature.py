"""Two stage features, like ratio, diff"""

import logging
from typing import Any

import pandas as pd

from .registry import FeatureRegistry, registry

logger = logging.getLogger(__name__)


@registry
def add_2order_feature(
    data: pd.DataFrame,
    columns: str | list[str],
    config: dict[str, Any] | None = None,
    time_col: str | None = None,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Add second-order features such as diff and pct_change."""
    if isinstance(columns, str):
        columns = [columns]

    result = data.copy()
    config = config or {}
    operations = config.get("operations", ["diff"])
    periods = config.get("periods", [1])

    if time_col is not None and time_col in result.columns:
        result = result.sort_values(by=time_col)

    for column in columns:
        if column not in result.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")

        if group_cols:
            grouped = result.groupby(group_cols, observed=True)[column]
            for period in periods:
                if "diff" in operations:
                    result[f"{column}_diff_{period}"] = grouped.diff(period)
                if "pct_change" in operations:
                    result[f"{column}_pct_change_{period}"] = grouped.pct_change(period)
        else:
            for period in periods:
                if "diff" in operations:
                    result[f"{column}_diff_{period}"] = result[column].diff(period)
                if "pct_change" in operations:
                    result[f"{column}_pct_change_{period}"] = result[column].pct_change(period)

    logger.info("Added second-order features for %s columns", len(columns))
    return result
