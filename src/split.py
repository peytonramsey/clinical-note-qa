"""Phase 2: split the data by note so no note appears in more than one split."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATA = Path("data")
SEED = 42
EVAL_PER_TASK = 50


def split(df: pd.DataFrame):
    # group on a hash of the note text, not patient_id, so identical notes can't leak
    df = df.copy()
    df["note_id"] = pd.util.hash_pandas_object(df["note"], index=False)

    # 70% train, 30% held out
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, hold_idx = next(gss.split(df, groups=df["note_id"]))
    train, hold = df.iloc[train_idx], df.iloc[hold_idx]

    # split the held-out 30% into validation and test (15% / 15%)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_idx, test_idx = next(gss2.split(hold, groups=hold["note_id"]))
    val, test = hold.iloc[val_idx], hold.iloc[test_idx]
    return train, val, test


def check_no_leakage(train, val, test):
    assert set(train.note_id).isdisjoint(test.note_id)
    assert set(train.note_id).isdisjoint(val.note_id)
    assert set(val.note_id).isdisjoint(test.note_id)


def task_mix(train, val, test):
    return pd.concat({"train": train.task.value_counts(normalize=True),
                      "val": val.task.value_counts(normalize=True),
                      "test": test.task.value_counts(normalize=True)}, axis=1).round(3)


def make_eval_set(test: pd.DataFrame) -> pd.DataFrame:
    # groupby().sample keeps the "task" column; groupby().apply drops it in pandas 3
    assert test.task.value_counts().min() >= EVAL_PER_TASK
    return test.groupby("task").sample(n=EVAL_PER_TASK, random_state=SEED)


def main():
    df = pd.read_parquet(DATA / "raw" / "asclepius.parquet")
    train, val, test = split(df)

    check_no_leakage(train, val, test)
    print(len(train), len(val), len(test))
    print(task_mix(train, val, test))

    eval_set = make_eval_set(test)
    assert len(eval_set) == EVAL_PER_TASK * df["task"].nunique()

    train.to_parquet(DATA / "train.parquet")
    val.to_parquet(DATA / "val.parquet")
    test.to_parquet(DATA / "test.parquet")
    eval_set.to_parquet(DATA / "eval_400.parquet")
    print("eval set:", len(eval_set), "rows")


if __name__ == "__main__":
    main()
