"""
数据预处理脚本
================
从 Excel 原始评分数据生成长格式 CSV。

执行方式：
    python data_prep.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
SRC = DATA_DIR / "jester_ratings.xlsx"
DST = DATA_DIR / "ratings_long.csv"


def main():
    print(f"读取 {SRC} ...")
    df = pd.read_excel(SRC)
    print(f"原始数据形状: {df.shape}")

    # 去掉第一列（评分数量）
    df = df.drop(df.columns[0], axis=1)
    # 列名重命名为 1..158
    df.columns = range(1, len(df.columns) + 1)
    # 99 表示未评分 -> NaN
    df = df.replace(99, np.nan)

    # 转长格式
    long_df = df.reset_index().rename(columns={"index": "user_id"})
    long_df = long_df.melt(
        id_vars=["user_id"],
        var_name="joke_id",
        value_name="rating",
    )
    long_df = long_df.dropna(subset=["rating"])
    long_df = long_df.astype({"user_id": int, "joke_id": int})

    DST.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(DST, index=False)

    print(f"已写入 {DST}")
    print(f"  总行数: {len(long_df):,}")
    print(f"  用户数: {long_df.user_id.nunique():,}")
    print(f"  笑话数: {long_df.joke_id.nunique()}")


if __name__ == "__main__":
    main()
