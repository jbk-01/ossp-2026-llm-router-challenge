import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

tr = pd.read_csv("build/train.csv"); tr["prompt"]=tr["prompt"].fillna("")
dv = pd.read_csv("build/dev.csv");   dv["prompt"]=dv["prompt"].fillna("")

def nf(d):
    p=d["prompt"]; L=p.str.len().clip(lower=1)
    return np.c_[np.log1p(L), p.str.count(r"[가-힣]")/L, p.str.count(r"[0-9]")/L,
                 p.str.count(r"[+\-*/=^]")/L, p.str.count(r"[(){}\[\]]")/L,
                 p.str.count(r"\n")/L, p.str.contains(r"def |return |import ").astype(float),
                 (L>2000).astype(float)]

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), max_features=30000,
                      min_df=3, sublinear_tf=True)
X  = sp.hstack([vec.fit_transform(tr["prompt"]), nf(tr)]).tocsr()
Xd = sp.hstack([vec.transform(dv["prompt"]),      nf(dv)]).tocsr()

pl = Ridge(alpha=1).fit(X, tr["score_ax31-light"]).predict(Xd).clip(0,1)
gain = (dv["score_ax31"] - dv["score_ax31-light"]).values
cost = (dv["cost_ax31"] - dv["cost_ax31-light"]).values
BUDGET = dv["cost_ax31-light"].sum() * 0.166   # 현재 fast 실제 사용량과 동일

print(f"{'전략':<34}{'승급수':>7}{'실제이득':>10}")
for thr in [1.01, 0.8, 0.6, 0.5, 0.4, 0.3]:
    ok = np.where((pl < thr) & (cost > 0))[0]
    order = ok[np.argsort(cost[ok])]
    c=0; n=0; g=0.0
    for i in order:
        if c+cost[i] <= BUDGET: c+=cost[i]; n+=1; g+=gain[i]
    label = "제한없음(현재)" if thr>1 else f"예측 light점수 < {thr}"
    print(f"{label:<34}{n:>7}{g:>10.2f}")
print(f"\n{'같은 예산 최적':<34}{'':>7}{105.00:>10.2f}")
