"""
笑话推荐系统 - Streamlit Web 应用
================================
ModelScope 部署入口文件 (app.py)

用户交互流程：
  浏览笑话 → 评分 → 获取推荐 → 评价推荐 → 查看满意度
"""
import numpy as np
import pandas as pd
import streamlit as st

from item_item_cf import (
    compute_similarity_matrix as cf_compute,
    recommend as cf_recommend,
)
from svd_recommendation import (
    compute_similarity_matrix as svd_compute,
    recommend as svd_recommend,
)


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="笑话推荐系统",
    page_icon="😂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 加载笑话文本（@st.cache_data 缓存）
# ============================================================
@st.cache_data
def load_jokes(path: str = "data/Dataset4JokeSet.xlsx"):
    df = pd.read_excel(path, header=None)
    df.columns = ["joke"]
    df.index = range(1, len(df) + 1)
    df.index.name = "joke_id"
    return df


df_jokes = load_jokes()
RATING_MIN, RATING_MAX = -10.0, 10.0     # 评分滑杆范围


# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.title("😂 笑话推荐系统")
    st.markdown("---")

    method_label = st.radio(
        "推荐方法",
        ("Item-Item 协同过滤", "SVD 矩阵分解"),
        help="两种方法都基于评分加权求和相似度向量，差异在于相似度的构造方式。",
    )
    method_key = "cf" if "Item-Item" in method_label else "svd"

    st.markdown("---")
    st.markdown("### 📊 数据概览")
    st.markdown("- 评分数据：106,488 条")
    st.markdown("- 用户数量：7,698")
    st.markdown("- 笑话数量：136（实际有评分）")

    st.markdown("---")
    st.markdown("### 💡 使用说明")
    st.markdown(
        "1. 给 **3 个随机笑话** 评分\n"
        "2. 点击 **生成推荐** 按钮\n"
        "3. 给 **5 个推荐笑话** 评分\n"
        "4. 点击 **计算满意度** 查看结果"
    )

    if st.button("🔄 重新开始"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ============================================================
# 计算相似度（@st.cache_resource 缓存，二选一加载）
# ============================================================
@st.cache_resource
def load_cf():
    return cf_compute()

@st.cache_resource
def load_svd():
    return svd_compute()

if method_key == "cf":
    sim, joke_ids, id_to_idx = load_cf()
else:
    sim, joke_ids, id_to_idx = load_svd()


# ============================================================
# 初始化 session_state
# ============================================================
if "step" not in st.session_state:
    st.session_state.step = 1
if "rated_jokes" not in st.session_state or not st.session_state.rated_jokes:
    st.session_state.rated_jokes = np.random.choice(joke_ids, 3, replace=False).tolist()
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "rec_ratings" not in st.session_state:
    st.session_state.rec_ratings = {}


# ============================================================
# 标题
# ============================================================
st.title("🎭 个性化笑话推荐")
st.caption(
    f"当前方法：**{method_label}**　|　"
    f"相似度矩阵：{sim.shape[0]} × {sim.shape[1]}"
)
st.markdown("---")


# ============================================================
# Step 1. 浏览笑话 + 评分
# ============================================================
st.header("📝 第 1 步：浏览笑话 & 评分")
st.caption("下方随机展示 3 个笑话，请拖动滑杆给出您的评分（-10 ~ +10，分越高代表越好笑）。")

cols = st.columns(3, gap="large")
for i, jid in enumerate(st.session_state.rated_jokes):
    with cols[i]:
        st.subheader(f"笑话 #{jid}")
        with st.container(border=True, height=350):
            st.write(df_jokes.loc[jid, "joke"])
        rating = st.slider(
            f"您的评分（{RATING_MIN} ~ {RATING_MAX}）",
            min_value=RATING_MIN,
            max_value=RATING_MAX,
            value=0.0,
            step=0.5,
            key=f"rate_{method_key}_{jid}",
        )
        st.session_state.user_ratings[jid] = rating

st.markdown("")

col_btn1, col_btn2, _ = st.columns([1, 1, 4])
with col_btn1:
    reroll = st.button("🎲 换一组笑话", use_container_width=True)
with col_btn2:
    go_rec = st.button("✨ 生成推荐", type="primary", use_container_width=True)

if reroll:
    st.session_state.rated_jokes = np.random.choice(joke_ids, 3, replace=False).tolist()
    st.session_state.user_ratings = {}
    st.session_state.recommendations = []
    st.session_state.rec_ratings = {}
    st.session_state.step = 1
    st.rerun()

if go_rec:
    if not st.session_state.user_ratings:
        st.warning("⚠️ 请先为笑话评分！")
    else:
        if method_key == "cf":
            recs = cf_recommend(st.session_state.user_ratings, sim, joke_ids, id_to_idx, top_k=5)
        else:
            recs = svd_recommend(st.session_state.user_ratings, sim, joke_ids, id_to_idx, top_k=5)

        st.session_state.recommendations = recs
        st.session_state.rec_ratings = {jid: 0.0 for jid, _ in recs}
        st.session_state.step = 2
        st.success("🎉 已为您生成 5 个推荐！请在下方继续评价。")
        st.rerun()


# ============================================================
# Step 2. 推荐结果展示 + 评价推荐
# ============================================================
if st.session_state.step >= 2 and st.session_state.recommendations:
    st.markdown("---")
    st.header("🎁 第 2 步：评价推荐结果")
    st.info(
        "👇 请您为以下 5 个推荐的笑话也打个分（默认 0 表示中性），"
        "最后点击底部 **计算满意度** 按钮。"
    )

    rec_cols = st.columns(2, gap="medium")
    for i, (jid, score) in enumerate(st.session_state.recommendations):
        with rec_cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"#### #{i+1}　笑话 {jid}")
                st.caption(f"推荐分数：{score:.4f}")
                with st.expander("查看笑话内容", expanded=False):
                    st.write(df_jokes.loc[jid, "joke"])
                r = st.slider(
                    "您的评分",
                    min_value=RATING_MIN,
                    max_value=RATING_MAX,
                    value=st.session_state.rec_ratings.get(jid, 0.0),
                    step=0.5,
                    key=f"rec_{method_key}_{jid}",
                )
                st.session_state.rec_ratings[jid] = r

    st.markdown("")
    if st.button("📊 计算满意度", type="primary", use_container_width=False):
        st.session_state.step = 3
        st.rerun()


# ============================================================
# Step 3. 查看满意度
# ============================================================
if st.session_state.step >= 3 and st.session_state.recommendations:
    st.markdown("---")
    st.header("🏆 第 3 步：推荐满意度")

    ratings = list(st.session_state.rec_ratings.values())
    # 归一化 [-10, 10] -> [0, 1] -> 0-100%
    normalized = [(r - RATING_MIN) / (RATING_MAX - RATING_MIN) for r in ratings]
    satisfaction = float(np.mean(normalized)) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("推荐方法", method_label)
    c2.metric("平均原始评分", f"{np.mean(ratings):.2f}")
    c3.metric("满意度", f"{satisfaction:.1f}%")

    st.markdown("#### 各推荐笑话的得分明细")
    detail = pd.DataFrame(
        {
            "笑话ID": [jid for jid, _ in st.session_state.recommendations],
            "推荐分数": [f"{s:.4f}" for _, s in st.session_state.recommendations],
            "您的评分": [f"{r:.2f}" for r in ratings],
            "归一化得分": [f"{n*100:.1f}%" for n in normalized],
        }
    )
    st.dataframe(detail, use_container_width=True, hide_index=True)

    if satisfaction >= 75:
        st.success(f"🎉 满意度很高（{satisfaction:.1f}%），推荐符合您的口味！")
    elif satisfaction >= 50:
        st.info(f"🙂 满意度一般（{satisfaction:.1f}%），推荐尚可接受。")
    else:
        st.warning(f"🤔 满意度较低（{satisfaction:.1f}%），可以换种方法试试。")
