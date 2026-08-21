# SPDX-FileCopyrightText: Copyright 2026 뭘했음청년들
# SPDX-License-Identifier: Apache-2.0
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
X = sp.hstack([vec.fit_transform(tr["prompt"]), nf(tr)]).tocsr()
Xd = sp.hstack([vec.transform(dv["prompt"]), nf(dv)]).tocsr()

actual = dv["score_ax31"] - dv["score_ax31-light"]

# 방법 A: 두 예측의 차이 (현재 방식)
a = Ridge(alpha=1).fit(X, tr["score_ax31"]).predict(Xd)
b = Ridge(alpha=1).fit(X, tr["score_ax31-light"]).predict(Xd)
diff = a - b

# 방법 B: 이득 직접 예측
g = Ridge(alpha=1).fit(X, tr["score_ax31"]-tr["score_ax31-light"]).predict(Xd)

print(f"A) 차이 방식   상관 {np.corrcoef(diff, actual)[0,1]:.3f}")
print(f"B) 직접 예측   상관 {np.corrcoef(g, actual)[0,1]:.3f}")

# 상위 152개를 골랐을 때 실제 이득
for name, pred in [("A 차이", diff), ("B 직접", g)]:
    idx = np.argsort(-pred)[:152]
    print(f"{name}: 상위152 실제이득 {actual.iloc[idx].sum():.2f}")
print(f"최적(정답)   : 105.00")
