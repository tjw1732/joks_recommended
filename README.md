# 😂 个性化笑话推荐系统

基于协同过滤与 SVD 矩阵分解的笑话推荐 Web 应用。  
本项目改编自「实验七：笑话推荐系统」，用 Streamlit 包装为可在 ModelScope 创空间一键部署的 Web 应用。

---

## ✨ 功能特性

| 功能 | 说明 |
|---|---|
| 1. 浏览笑话 & 评分 | 随机展示 3 个笑话，拖动滑杆给出 -10 ~ +10 评分 |
| 2. 个性化推荐 | 根据用户对 3 个笑话的评分，调用推荐函数生成 Top-5 |
| 3. 评价推荐 | 用户对 5 个推荐笑话逐个评分 |
| 4. 满意度 | 把 5 个评分归一化到 0-100% 并求平均 |
| 5. 双方法对比 | 侧边栏可切换 **Item-Item CF** / **SVD 矩阵分解** |
| 6. 缓存优化 | `@st.cache_resource` 避免每次交互重复计算相似度矩阵 |

---

## 🗂 项目目录结构

```
.
├── app.py                   # ★ Streamlit 主入口（ModelScope 默认）
├── item_item_cf.py          # Item-Item 协同过滤模块
├── svd_recommendation.py    # SVD 矩阵分解模块
├── data_prep.py             # 从 Excel 生成长格式 CSV 的预处理脚本
├── requirements.txt         # 依赖列表
├── README.md                # 本文档
├── .gitignore
│
├── data/                    # 数据目录
│   ├── Dataset4JokeSet.xlsx         # 158 条笑话文本
│   ├── jester_ratings.xlsx          # 原始评分数据（7698 × 158）
│   └── ratings_long.csv             # 长格式评分（106,488 条，1.45 MB）
│
└── 实验七代码+...ipynb      # 原始实验 notebook（仅供参考）
```

> ModelScope 部署时，**只需 `app.py / item_item_cf.py / svd_recommendation.py / requirements.txt / README.md / data/`** 这几样。

---

## 🚀 本地运行

```powershell
# 1. 安装依赖
pip install -r requirements.txt

# 2. （可选）重新生成 CSV —— data/ratings_long.csv 已包含
python data_prep.py

# 3. 启动应用
streamlit run app.py
```

启动后访问 <http://localhost:8501>。

---

## ☁️ 部署到 ModelScope 创空间

1. 在 [ModelScope](https://www.modelscope.cn/) 注册并进入「创空间」
2. 选择 **Streamlit** 框架
3. 上传本仓库代码（确保根目录有 `app.py` 与 `requirements.txt`）
4. 在「运行环境」选择 Python 3.10+，等待依赖安装
5. 启动后即可访问公网 URL

ModelScope 创空间会自动识别：
- `app.py` 作为 Streamlit 入口
- `requirements.txt` 自动 `pip install`
- `data/` 下的文件可被应用直接读取

---

## 🧠 推荐算法说明

### 共同推荐逻辑（两个方法共用）

```python
def recommend(user_ratings, sim_matrix, joke_ids, id_to_idx, top_k=5):
    scores = np.zeros(len(joke_ids))
    for jid, rating in user_ratings.items():
        scores += rating * sim_matrix[id_to_idx[jid]]   # 评分加权
    scores[[id_to_idx[j] for j in user_ratings]] = -np.inf  # 排除已评分
    return top_k_sorted(scores)
```

**思路**：用户对 3 个笑话分别打了分，每个评分过的笑话都有一个与其他笑话的相似度向量。把 3 个相似度向量按用户评分**加权求和**，得到一个综合推荐分数向量，从中选取用户**未评分且得分最高**的 5 个笑话。

### 方法一：Item-Item 协同过滤
```
CSV → pivot(笑话×用户, fill=0) → cosine_similarity → sim_matrix
```

### 方法二：SVD 矩阵分解
```
CSV → pivot(笑话×用户, fill=0)
    → TruncatedSVD(n_components=20, random_state=42)  // 158×7698 → 158×20
    → cosine_similarity(潜在空间) → sim_matrix
```

| 维度 | Item-Item CF | SVD |
|---|---|---|
| 特征空间 | 7,698 维用户评分向量 | 20 维潜在因子 |
| 冷启动 | 差 | 一般 |
| 准确性 | 高 | 高 |
| 可解释性 | 中 | 差 |

---

## 👣 用户交互流程

```
┌─────────────────────────────────────────────────────┐
│  侧边栏：选择方法 (Item-Item CF / SVD)              │
└────────────────────┬────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│  ① 浏览笑话 → 评分                                   │
│     随机展示 3 个笑话，拖动滑杆给分（-10 ~ +10）      │
└────────────────────┬────────────────────────────────┘
                     ▼ 点击「生成推荐」
┌─────────────────────────────────────────────────────┐
│  ② 获取推荐                                          │
│     推荐函数根据用户评分输出 Top-5 候选笑话            │
└────────────────────┬────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│  ③ 评价推荐                                          │
│     对 5 个推荐笑话逐个评分（默认 0 = 中性）           │
└────────────────────┬────────────────────────────────┘
                     ▼ 点击「计算满意度」
┌─────────────────────────────────────────────────────┐
│  ④ 查看满意度                                        │
│     归一化公式 (r - (-10)) / 20 × 100%               │
│     5 个分数取平均即为推荐满意度                       │
└─────────────────────────────────────────────────────┘
```

---

## 📊 数据来源

- 笑话文本：[Jester Joke Dataset](https://goldberg.berkeley.edu/jester-data/) —— 158 条英文笑话
- 用户评分：[final] April 2015 to Nov 30 2019 - Transformed Jester Data
  - 7,698 名用户对 158 个笑话的评分
  - 评分范围 -10.00 ~ +10.00（实际 -42.28 ~ 28.58，99 表示未评分）
  - 已预处理为 `data/ratings_long.csv`（106,488 条非空评分）

---

## 📝 实验报告要点

1. **数据流改造**：原 notebook 走「Excel 宽表 → melt 长表」；现改为「Excel 宽表 → data_prep.py → CSV 长表 → pivot」。
2. **代码模块化**：原 notebook 27 个 cell 拆为 4 个文件 —— `app.py / item_item_cf.py / svd_recommendation.py / data_prep.py`。
3. **统一推荐函数**：`recommend()` 在两个方法间复用，仅相似度矩阵不同。
4. **缓存策略**：`@st.cache_data` 加载笑话文本；`@st.cache_resource` 缓存相似度矩阵（CF/SVD 各自缓存一份），用户切换方法或重新评分时不会重复计算。
5. **Streamlit 兼容 shim**：`item_item_cf.py` / `svd_recommendation.py` 中对 `import streamlit` 做了 `try/except`，未装 streamlit 时也能 `python xxx.py` 单独跑。
