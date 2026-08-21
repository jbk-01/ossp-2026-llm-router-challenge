# SPDX-FileCopyrightText: Copyright 2026 뭘했음청년들
# SPDX-License-Identifier: Apache-2.0
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

tr = pd.read_csv("build/train.csv"); tr["prompt"] = tr["prompt"].fillna("")
dv = pd.read_csv("build/dev.csv");   dv["prompt"] = dv["prompt"].fillna("")

def nf(d):
    p = d["prompt"]; L = p.str.len().clip(lower=1)
    return np.c_[np.log1p(L), p.str.count(r"[가-힣]")/L, p.str.count(r"[0-9]")/L,
                 p.str.count(r"[+\-*/=^]")/L, p.str.count(r"[(){}\[\]]")/L,
                 p.str.count(r"\n")/L, p.str.count(r"\?")/L,
                 p.str.contains(r"def |return |import ").astype(float),
                 (L > 2000).astype(float)]

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4),
                      max_features=30000, min_df=3, sublinear_tf=True)
Ttr = vec.fit_transform(tr["prompt"]); Tdv = vec.transform(dv["prompt"])
Xtr = sp.hstack([Ttr, nf(tr)]).tocsr(); Xdv = sp.hstack([Tdv, nf(dv)]).tocsr()

svd = TruncatedSVD(n_components=120, random_state=0)
Dtr = np.c_[svd.fit_transform(Ttr), nf(tr)]
Ddv = np.c_[svd.transform(Tdv), nf(dv)]

act = (dv["score_ax31"] - dv["score_ax31-light"]).values
K = 152

def report(name, pred):
    idx = np.argsort(-pred)[:K]
    print(f"{name:<26} 상관 {np.corrcoef(pred, act)[0,1]:6.3f}   상위{K} 이득 {act[idx].sum():7.2f}")

# 1) 현재 방식
report("Ridge (현재)", Ridge(alpha=1).fit(Xtr, tr["score_ax31"]-tr["score_ax31-light"]).predict(Xdv))

# 2) 로지스틱 확률 차이
yl = (tr["score_ax31-light"] > 0.5).astype(int)
ya = (tr["score_ax31"] > 0.5).astype(int)
pl = LogisticRegression(max_iter=2000, C=1).fit(Xtr, yl).predict_proba(Xdv)[:,1]
pa = LogisticRegression(max_iter=2000, C=1).fit(Xtr, ya).predict_proba(Xdv)[:,1]
report("Logistic 확률차", pa - pl)
report("Logistic (1-pl)*pa", (1-pl)*pa)

# 3) 부스팅으로 이득 직접 예측
report("HGB 이득직접",
       HGB(max_iter=300, random_state=0).fit(Dtr, tr["score_ax31"]-tr["score_ax31-light"]).predict(Ddv))

# 4) 부스팅 점수 예측 후 차이
ha = HGB(max_iter=300, random_state=0).fit(Dtr, tr["score_ax31"]).predict(Ddv)
hl = HGB(max_iter=300, random_state=0).fit(Dtr, tr["score_ax31-light"]).predict(Ddv)
report("HGB 점수차", ha - hl)

print(f"{'최적(정답)':<26}                 상위{K} 이득  105.00")
print(f"{'무작위 기대':<26}                 상위{K} 이득 {act.mean()*K:7.2f}")
