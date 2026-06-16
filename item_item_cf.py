"""
笑话推荐 - Item-Item 协同过滤
================================
数据流：CSV(long) -> pivot(笑话×用户, 缺值填0) -> cosine_similarity
推荐  ：按用户对3个笑话的评分，对相似度向量加权求和，取未评分的 Top-5
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 让脚本在未安装 streamlit 时也能直接 python xxx.py 运行
try:
    import streamlit as st
except ImportError:
    class _StShim:
        @staticmethod
        def cache_resource(func):
            return func
    st = _StShim()


CSV_PATH = 'data/ratings_long.csv'    # 长格式：user_id, joke_id, rating


# ============================================================
# 1. 加载数据 + 计算 Item-Item 相似度（用 @st.cache_resource 缓存）
# ============================================================
@st.cache_resource
def compute_similarity_matrix(csv_path: str = CSV_PATH):
    """
    1) 从 CSV 读长格式评分
    2) pivot 为 笑话×用户 矩阵，缺失值填 0
    3) 计算笑话两两之间的余弦相似度

    返回:
        sim_matrix  : np.ndarray, shape = (n_jokes, n_jokes)
        joke_ids    : list[int],   与矩阵行一一对应
        id_to_idx   : dict,        joke_id -> 行下标
    """
    df = pd.read_csv(csv_path)
    pivot = df.pivot_table(
        index='joke_id', columns='user_id', values='rating', fill_value=0
    )
    joke_ids = sorted(pivot.index.tolist())          # 1..N 升序
    matrix = pivot.loc[joke_ids].values.astype(float)

    sim = cosine_similarity(matrix)
    id_to_idx = {jid: i for i, jid in enumerate(joke_ids)}
    return sim, joke_ids, id_to_idx


# ============================================================
# 2. 统一推荐逻辑（与 SVD 共用）
# ============================================================
def recommend(user_ratings: dict,
              sim_matrix: np.ndarray,
              joke_ids: list,
              id_to_idx: dict,
              top_k: int = 5):
    """
    根据用户对若干笑话的评分，加权求和相似度向量，
    排除用户已评分笑话，输出 Top-K 推荐。

    参数:
        user_ratings : {joke_id: rating} 用户对每个笑话的评分
        sim_matrix   : 相似度矩阵 (n_jokes, n_jokes)
        joke_ids     : 与矩阵行对应的笑话 ID 列表
        id_to_idx    : joke_id -> 行下标
        top_k        : 推荐数量

    返回:
        [(joke_id, score), ...]   按 score 降序，长度 = top_k
    """
    n = len(joke_ids)
    scores = np.zeros(n)

    for jid, rating in user_ratings.items():
        idx = id_to_idx[jid]
        scores += rating * sim_matrix[idx]            # 按评分加权

    # 排除已评分的笑话
    rated_idx = [id_to_idx[jid] for jid in user_ratings if jid in id_to_idx]
    scores[rated_idx] = -np.inf

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(joke_ids[i], float(scores[i])) for i in top_indices]


# ============================================================
# 3. 独立运行示例
# ============================================================
if __name__ == '__main__':
    sim, joke_ids, id_to_idx = compute_similarity_matrix()

    print(f"笑话总数: {len(joke_ids)}")
    print(f"相似度矩阵形状: {sim.shape}")

    # 假设用户对 3 个热门笑话的评分
    user_ratings = {19: 8.5, 72: 6.0, 105: 9.2}
    print(f"\n用户对 3 个笑话的评分: {user_ratings}")

    recs = recommend(user_ratings, sim, joke_ids, id_to_idx, top_k=5)
    print("\n【Item-Item CF 推荐 Top-5】")
    print("-" * 50)
    for rank, (jid, score) in enumerate(recs, 1):
        print(f"  {rank}. 笑话 {jid:<4}  推荐分数: {score:8.4f}")
