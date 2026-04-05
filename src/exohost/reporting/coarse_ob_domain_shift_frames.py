# Файл `coarse_ob_domain_shift_frames.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score

from exohost.evaluation.hierarchical_tasks import GAIA_ID_COARSE_CLASSIFICATION_TASK
from exohost.reporting.coarse_ob_domain_shift_contracts import (
    CoarseOBDomainShiftReviewBundle,
)


def build_domain_membership_summary_frame(
    bundle: CoarseOBDomainShiftReviewBundle,
) -> pd.DataFrame:
    # Короткая сводка по объему train/downstream boundary domains.
    return pd.DataFrame(
        [
            {
                "quality_state": bundle.config.quality_state,
                "hot_teff_min_k": float(bundle.config.hot_teff_min_k),
                "n_rows_train_boundary": int(bundle.train_boundary_df.shape[0]),
                "n_rows_downstream_boundary": int(bundle.downstream_boundary_df.shape[0]),
                "n_rows_train_scored": int(bundle.train_scored_df.shape[0]),
                "n_rows_downstream_scored": int(bundle.downstream_scored_df.shape[0]),
            }
        ]
    )


def build_domain_class_balance_frame(bundle: CoarseOBDomainShiftReviewBundle) -> pd.DataFrame:
    # Сравниваем баланс true классов между train-time и downstream domains.
    return _build_group_distribution_frame(
        _concat_domains(bundle.train_boundary_df, bundle.downstream_boundary_df),
        group_column="domain_name",
        label_column="spectral_class",
        group_label_name="domain_name",
        label_name="true_spectral_class",
    )


def build_domain_predicted_class_summary_frame(
    bundle: CoarseOBDomainShiftReviewBundle,
) -> pd.DataFrame:
    # Сравниваем predicted coarse labels между train-time и downstream domains.
    return _build_group_distribution_frame(
        _concat_domains(bundle.train_scored_df, bundle.downstream_scored_df),
        group_column="domain_name",
        label_column="coarse_predicted_label",
        group_label_name="domain_name",
        label_name="coarse_predicted_label",
    )


def build_domain_confusion_frame(bundle: CoarseOBDomainShiftReviewBundle) -> pd.DataFrame:
    # Строим confusion-like table по каждому domain отдельно.
    scored_df = _concat_domains(bundle.train_scored_df, bundle.downstream_scored_df)
    if scored_df.empty:
        return pd.DataFrame(
            columns=[
                "domain_name",
                "true_spectral_class",
                "coarse_predicted_label",
                "n_rows",
                "share_within_true_class",
            ]
        )

    grouped_df = (
        scored_df.groupby(
            ["domain_name", "spectral_class", "coarse_predicted_label"],
            dropna=False,
            sort=True,
        )
        .agg(n_rows=("source_id", "size"))
        .reset_index()
        .rename(columns={"spectral_class": "true_spectral_class"})
    )
    totals = (
        grouped_df.groupby(["domain_name", "true_spectral_class"], dropna=False, sort=True)["n_rows"]
        .transform("sum")
        .astype("int64")
    )
    grouped_df["share_within_true_class"] = grouped_df["n_rows"] / totals
    return grouped_df.sort_values(
        ["domain_name", "true_spectral_class", "n_rows", "coarse_predicted_label"],
        ascending=[True, True, False, True],
        kind="mergesort",
        ignore_index=True,
    )


def build_domain_probability_summary_frame(
    bundle: CoarseOBDomainShiftReviewBundle,
) -> pd.DataFrame:
    # Сравниваем `P(O)` и `P(B)` по domain и true class.
    scored_df = _concat_domains(bundle.train_scored_df, bundle.downstream_scored_df)
    required_columns = {"coarse_probability__O", "coarse_probability__B"}
    if not required_columns.issubset(scored_df.columns):
        return pd.DataFrame(
            columns=[
                "domain_name",
                "true_spectral_class",
                "n_rows",
                "median_probability__O",
                "median_probability__B",
                "mean_probability__O",
                "mean_probability__B",
            ]
        )
    grouped_df = (
        scored_df.groupby(["domain_name", "spectral_class"], dropna=False, sort=True)
        .agg(
            n_rows=("source_id", "size"),
            median_probability__O=("coarse_probability__O", "median"),
            median_probability__B=("coarse_probability__B", "median"),
            mean_probability__O=("coarse_probability__O", "mean"),
            mean_probability__B=("coarse_probability__B", "mean"),
        )
        .reset_index()
        .rename(columns={"spectral_class": "true_spectral_class"})
    )
    return grouped_df.sort_values(
        ["domain_name", "true_spectral_class"],
        ascending=[True, True],
        kind="mergesort",
        ignore_index=True,
    )


def build_domain_physics_summary_frame(bundle: CoarseOBDomainShiftReviewBundle) -> pd.DataFrame:
    # Сравниваем физику train/downstream domains внутри true `O` и true `B`.
    domain_df = _concat_domains(bundle.train_boundary_df, bundle.downstream_boundary_df)
    if domain_df.empty:
        return pd.DataFrame(
            columns=[
                "domain_name",
                "true_spectral_class",
                "n_rows",
                "median_teff_gspphot",
                "median_logg_gspphot",
                "median_mh_gspphot",
                "median_bp_rp",
                "median_parallax",
                "median_parallax_over_error",
                "median_ruwe",
                "median_radius_feature",
                "median_radius_flame",
            ]
        )
    grouped_df = (
        domain_df.groupby(["domain_name", "spectral_class"], dropna=False, sort=True)
        .agg(
            n_rows=("source_id", "size"),
            median_teff_gspphot=("teff_gspphot", "median"),
            median_logg_gspphot=("logg_gspphot", "median"),
            median_mh_gspphot=("mh_gspphot", "median"),
            median_bp_rp=("bp_rp", "median"),
            median_parallax=("parallax", "median"),
            median_parallax_over_error=("parallax_over_error", "median"),
            median_ruwe=("ruwe", "median"),
            median_radius_feature=("radius_feature", "median"),
            median_radius_flame=("radius_flame", "median"),
        )
        .reset_index()
        .rename(columns={"spectral_class": "true_spectral_class"})
    )
    return grouped_df.sort_values(
        ["domain_name", "true_spectral_class"],
        ascending=[True, True],
        kind="mergesort",
        ignore_index=True,
    )


def build_domain_missingness_summary_frame(
    bundle: CoarseOBDomainShiftReviewBundle,
    *,
    feature_columns: tuple[str, ...] = GAIA_ID_COARSE_CLASSIFICATION_TASK.feature_columns,
) -> pd.DataFrame:
    # Считаем missingness по ключевым feature columns внутри каждого domain/class.
    domain_df = _concat_domains(bundle.train_boundary_df, bundle.downstream_boundary_df)
    rows: list[dict[str, object]] = []
    for domain_name, domain_group in domain_df.groupby("domain_name", dropna=False, sort=True):
        for true_class, class_group in domain_group.groupby("spectral_class", dropna=False, sort=True):
            n_rows = int(class_group.shape[0])
            for feature_name in feature_columns:
                if feature_name not in class_group.columns:
                    continue
                feature_series = class_group.loc[:, feature_name]
                if not isinstance(feature_series, pd.Series):
                    raise TypeError(f"Expected pandas Series for feature {feature_name}.")
                missing_share = float(feature_series.isna().mean())
                rows.append(
                    {
                        "domain_name": str(domain_name),
                        "true_spectral_class": str(true_class),
                        "feature_name": feature_name,
                        "n_rows": n_rows,
                        "missing_share": missing_share,
                    }
                )
    return pd.DataFrame.from_records(rows)


def build_domain_shift_auc_frame(
    bundle: CoarseOBDomainShiftReviewBundle,
    *,
    feature_columns: tuple[str, ...] = GAIA_ID_COARSE_CLASSIFICATION_TASK.feature_columns,
) -> pd.DataFrame:
    # Оцениваем, насколько feature различает train-time и downstream внутри true class.
    domain_df = _concat_domains(bundle.train_boundary_df, bundle.downstream_boundary_df)
    rows: list[dict[str, object]] = []
    for true_class, class_group in domain_df.groupby("spectral_class", dropna=False, sort=True):
        domain_series = class_group["domain_name"].astype("string")
        for feature_name in feature_columns:
            if feature_name not in class_group.columns:
                continue
            numeric_series = class_group[feature_name].map(_to_optional_float)
            valid_mask = numeric_series.notna() & domain_series.isin(
                ["train_time", "downstream_pass"]
            )
            if not bool(valid_mask.any()):
                continue
            valid_df = pd.DataFrame(
                {
                    "domain_name": domain_series.loc[valid_mask].astype("string"),
                    "feature_value": numeric_series.loc[valid_mask].astype("float64"),
                }
            )
            if valid_df["domain_name"].nunique(dropna=True) < 2:
                continue
            target_binary = valid_df["domain_name"].map(
                lambda value: 1 if value == "downstream_pass" else 0
            )
            auc_downstream = float(roc_auc_score(target_binary, valid_df["feature_value"]))
            separability_auc = max(auc_downstream, 1.0 - auc_downstream)
            rows.append(
                {
                    "true_spectral_class": str(true_class),
                    "feature_name": feature_name,
                    "n_rows_used": int(valid_df.shape[0]),
                    "auc_ovr_downstream": auc_downstream,
                    "separability_auc": separability_auc,
                    "higher_value_domain": (
                        "downstream_pass" if auc_downstream >= 0.5 else "train_time"
                    ),
                    "median_train_time": _series_median(
                        valid_df.loc[valid_df["domain_name"] == "train_time", "feature_value"]
                    ),
                    "median_downstream_pass": _series_median(
                        valid_df.loc[
                            valid_df["domain_name"] == "downstream_pass",
                            "feature_value",
                        ]
                    ),
                }
            )
    result = pd.DataFrame.from_records(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["true_spectral_class", "separability_auc", "feature_name"],
        ascending=[True, False, True],
        kind="mergesort",
        ignore_index=True,
    )


def _concat_domains(train_df: pd.DataFrame, downstream_df: pd.DataFrame) -> pd.DataFrame:
    train_part = train_df.copy()
    downstream_part = downstream_df.copy()
    if "domain_name" not in train_part.columns:
        train_part["domain_name"] = "train_time"
    if "domain_name" not in downstream_part.columns:
        downstream_part["domain_name"] = "downstream_pass"
    if "spectral_class" not in train_part.columns and "spec_class" in train_part.columns:
        train_part["spectral_class"] = train_part["spec_class"].astype("string").str.upper()
    return pd.concat([train_part, downstream_part], ignore_index=True, sort=False)


def _build_group_distribution_frame(
    df: pd.DataFrame,
    *,
    group_column: str,
    label_column: str,
    group_label_name: str,
    label_name: str,
) -> pd.DataFrame:
    if group_column not in df.columns or label_column not in df.columns:
        return pd.DataFrame(columns=[group_label_name, label_name, "n_rows", "share_within_group"])
    grouped_df = (
        df.groupby([group_column, label_column], dropna=False, sort=True)
        .agg(n_rows=("source_id", "size"))
        .reset_index()
        .rename(columns={group_column: group_label_name, label_column: label_name})
    )
    totals = (
        grouped_df.groupby(group_label_name, dropna=False, sort=True)["n_rows"]
        .transform("sum")
        .astype("int64")
    )
    grouped_df["share_within_group"] = grouped_df["n_rows"] / totals
    return grouped_df.sort_values(
        [group_label_name, "n_rows", label_name],
        ascending=[True, False, True],
        kind="mergesort",
        ignore_index=True,
    )


def _series_median(series: pd.Series) -> float:
    return float(series.median())


def _to_optional_float(value: object) -> float | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return None if pd.isna(value) else float(value)
    if isinstance(value, str):
        stripped_value = value.strip()
        if not stripped_value:
            return None
        try:
            return float(stripped_value)
        except ValueError:
            return None
    return None
