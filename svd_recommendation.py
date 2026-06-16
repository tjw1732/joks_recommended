"""
笑话推荐 - SVD 矩阵分解
================================
数据流：CSV(long) -> pivot(笑话×用户, 缺值填0) -> TruncatedSVD(20)
       -> 在潜在空间 cosine_similarity
推荐  ：与 Item-Item CF 共用同一 recommend() 逻辑
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
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


CSV_PATH = 'data/ratings_long.csv'
N_COMPONENTS = 20


# ============================================================
# 1. 加载数据 + SVD 降维 + 潜在空间相似度（用 @st.cache_resource 缓存）
# ============================================================
@st.cache_resource
def compute_similarity_matrix(csv_path: str = CSV_PATH,
                              n_components: int = N_COMPONENTS,
                              random_state: int = 42):
    """
    1) 从 CSV 读长格式评分
    2) pivot 为 笑话×用户 矩阵，缺失值填 0
    3) TruncatedSVD 压缩到 n_components 维潜在空间
    4) 在潜在空间计算余弦相似度

    返回:
        sim_matrix  : np.ndarray, shape = (n_jokes, n_jokes)
        joke_ids    : list[int],   与矩阵行一一对应
        id_to_idx   : dict,        joke_id -> 行下标
    """
    df = pd.read_csv(csv_path)
    pivot = df.pivot_table(
        index='joke_id', columns='user_id', values='rating', fill_value=0
    )
    joke_ids = sorted(pivot.index.tolist())
    matrix = pivot.loc[joke_ids].values.astype(float)

    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    latent = svd.fit_transform(matrix)                  # (n_jokes, 20)

    sim = cosine_similarity(latent)
    id_to_idx = {jid: i for i, jid in enumerate(joke_ids)}
    return sim, joke_ids, id_to_idx


# ============================================================
# 2. 统一推荐逻辑（与 Item-Item CF 共用）
# ============================================================
def recommend(user_ratings: dict,
              sim_matrix: np.ndarray,
              joke_ids: list,
              id_to_idx: dict,
              top_k: int = 5):
    """
    根据用户对若干笑话的评分，加权求和相似度向量，
    排除用户已评分笑话，输出 Top-K 推荐。
    """
    n = len(joke_ids)
    scores = np.zeros(n)

    for jid, rating in user_ratings.items():
        idx = id_to_idx[jid]
        scores += rating * sim_matrix[idx]

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

    # 与 Item-Item CF 同样的输入，方便对比
    user_ratings = {19: 8.5, 72: 6.0, 105: 9.2}
    print(f"\n用户对 3 个笑话的评分: {user_ratings}")

    recs = recommend(user_ratings, sim, joke_ids, id_to_idx, top_k=5)
    print("\n【SVD 潜在空间 推荐 Top-5】")
    print("-" * 50)
    for rank, (jid, score) in enumerate(recs, 1):
        print(f"  {rank}. 笑话 {jid:<4}  推荐分数: {score:8.4f}")
