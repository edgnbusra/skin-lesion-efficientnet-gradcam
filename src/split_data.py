"""
HAM10000'i train/val/test olarak boler.
Onemli: ayni lezyonun (lesion_id) birden fazla fotografi olabiliyor.
Bunlari farkli split'lere dagitirsak veri sizintisi (data leakage) olur -
model ayni lezyonu train'de görüp test'te "hatirlayarak" basarili gorunebilir.
Bu yuzden GroupShuffleSplit ile lesion_id bazinda bolunuyor.
"""
import argparse
import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split(metadata_csv, output_dir, train_size=0.7, val_size=0.15, seed=42):
    df = pd.read_csv(metadata_csv)

    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idx, rest_idx = next(gss1.split(df, groups=df["lesion_id"]))
    train_df = df.iloc[train_idx]
    rest_df = df.iloc[rest_idx]

    relative_val_size = val_size / (1 - train_size)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=relative_val_size, random_state=seed)
    val_idx, test_idx = next(gss2.split(rest_df, groups=rest_df["lesion_id"]))
    val_df = rest_df.iloc[val_idx]
    test_df = rest_df.iloc[test_idx]

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")
    print("Sinif dagilimi (train):")
    print(train_df["dx"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/HAM10000_metadata.csv")
    parser.add_argument("--output_dir", default="data")
    args = parser.parse_args()
    split(args.metadata, args.output_dir)
