# Файл `build_host_field_training_dataset.py` слоя `datasets`.
#
# Этот файл отвечает только за:
# - loader-слой и сборку рабочих dataframe из relation-слоя;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `datasets` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from typing import cast

import pandas as pd

from exohost.contracts.feature_contract import ROUTER_FEATURES, unique_columns
from exohost.contracts.label_contract import (
    HOST_FIELD_TARGET_COLUMN,
    TARGET_SPECTRAL_CLASSES,
)

HOST_FIELD_GROUP_COLUMNS: tuple[str, ...] = ("spec_class", "evolution_stage")
HOST_FIELD_FEATURE_COLUMNS: tuple[str, ...] = ROUTER_FEATURES
HOST_FIELD_COLUMNS: tuple[str, ...] = unique_columns(
    ("source_id",),
    HOST_FIELD_GROUP_COLUMNS,
    ("spec_subclass",),
    HOST_FIELD_FEATURE_COLUMNS,
    (HOST_FIELD_TARGET_COLUMN,),
)


def build_host_field_training_dataset(
    host_df: pd.DataFrame,
    router_df: pd.DataFrame,
    *,
    field_to_host_ratio: int = 1,
    random_state: int = 42,
) -> pd.DataFrame:
    # Собираем matched датасет:
    # известные host-звезды против field-объектов той же coarse-группы.
    if field_to_host_ratio < 1:
        raise ValueError("field_to_host_ratio must be at least 1.")

    prepared_host = select_target_host_rows(host_df)
    prepared_field = select_target_field_rows(
        router_df,
        host_source_ids=cast(pd.Series, prepared_host.loc[:, "source_id"]),
    )
    sampled_field = sample_matched_field_rows(
        prepared_field,
        host_frame=prepared_host,
        field_to_host_ratio=field_to_host_ratio,
        random_state=random_state,
    )

    host_labeled = prepared_host.assign(host_label="host")
    field_labeled = sampled_field.assign(host_label="field")
    combined = pd.concat([host_labeled, field_labeled], ignore_index=True)
    result = combined.loc[:, [name for name in HOST_FIELD_COLUMNS if name in combined.columns]].copy()
    result["host_label"] = result["host_label"].astype(str)
    return sort_host_field_frame(result)


def select_target_host_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Для первой волны берём только целевые host-классы F/G/K/M.
    target_frame = df.loc[df["spec_class"].isin(TARGET_SPECTRAL_CLASSES)].copy()
    if target_frame.empty:
        raise ValueError("Host training frame does not contain target spectral classes.")
    return target_frame.reset_index(drop=True)


def select_target_field_rows(
    df: pd.DataFrame,
    *,
    host_source_ids: pd.Series,
) -> pd.DataFrame:
    # Убираем known hosts из field-пула и оставляем целевые coarse-классы.
    target_frame = df.loc[df["spec_class"].isin(TARGET_SPECTRAL_CLASSES)].copy()
    source_id_strings = host_source_ids.astype(str)
    filtered = target_frame.loc[
        ~target_frame["source_id"].astype(str).isin(source_id_strings)
    ].reset_index(drop=True)
    if filtered.empty:
        raise ValueError("Router field pool is empty after excluding known hosts.")
    return filtered


def sample_matched_field_rows(
    field_df: pd.DataFrame,
    *,
    host_frame: pd.DataFrame,
    field_to_host_ratio: int,
    random_state: int,
) -> pd.DataFrame:
    # Сэмплируем field-объекты в том же распределении по spec_class и stage,
    # что и host-выборка.
    sampled_frames: list[pd.DataFrame] = []

    group_counts = host_frame.groupby(
        list(HOST_FIELD_GROUP_COLUMNS),
        as_index=False,
    ).size()
    group_counts = pd.DataFrame(group_counts)
    group_counts.columns = ["spec_class", "evolution_stage", "host_count"]

    for row in group_counts.to_dict(orient="records"):
        spec_class = str(row["spec_class"])
        evolution_stage = str(row["evolution_stage"])
        host_count = int(row["host_count"])
        field_count = host_count * field_to_host_ratio

        group_frame = field_df.loc[
            (field_df["spec_class"] == spec_class)
            & (field_df["evolution_stage"] == evolution_stage)
        ]
        if int(group_frame.shape[0]) < field_count:
            raise ValueError(
                "Field pool does not contain enough matched rows for "
                f"group ({spec_class}, {evolution_stage})."
            )

        sampled_frames.append(
            group_frame.sample(
                n=field_count,
                random_state=random_state,
                replace=False,
            )
        )

    return pd.concat(sampled_frames, ignore_index=True)


def sort_host_field_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Делаем детерминированный порядок строк в host-vs-field датасете.
    return (
        df.assign(_source_sort_key=df["source_id"].astype(str))
        .sort_values(
            ["host_label", "spec_class", "evolution_stage", "_source_sort_key"],
            kind="mergesort",
            ignore_index=True,
        )
        .drop(columns="_source_sort_key")
    )
