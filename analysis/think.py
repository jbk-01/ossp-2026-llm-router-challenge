# SPDX-FileCopyrightText: Copyright 2026 뭘했음청년들
# SPDX-License-Identifier: Apache-2.0
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

tr = pd.read_csv("build/train.csv"); tr["prompt"]=tr["prompt"].fillna("")
dv = pd.read_csv("build/dev.csv");   dv["prompt"]=dv["prompt"].fillna("")

def grp(d):
    p=d["prompt"]; L=p.str.len().clip(lower=1)
    kor=p.str.count(r"[가-힣]")/L; dig=p.str.count(r"[0-9]")/L
    code=p.str.contains(r"def |return |import |\bclass\b", regex=True)
    g=np.where(d["chars"]>2000,"긴문맥",
      np.where(code,"코드",
      np.where(kor>0.3,"한국어",
      np.where(dig>0.05,"숫자많음","일반영어"))))
    return pd.Series(g, index=d.index)

for d in (tr, dv):
    d["g"]=grp(d)
    d["gt"]=d["score_axk1-think"]-d["score_ax31-light"]
    d["ct"]=d["cost_axk1-think"]-d["cost_ax31-light"]

print("=== think 승급: 그룹별 (Train / Dev) ===")
print(f"{'그룹':<8}{'Tr이득':>8}{'Dv이득':>8}{'Tr효율':>10}{'Dv효율':>10}")
for g in tr["g"].unique():
    a=tr[tr["g"]==g]; b=dv[dv["g"]==g]
    ea=a["gt"].sum()/a["ct"].sum(); eb=b["gt"].sum()/b["ct"].sum()
    print(f"{g:<8}{a['gt'].mean():>8.3f}{b['gt'].mean():>8.3f}{ea:>10.2f}{eb:>10.2f}")

# think 이득 예측 가능성
def nf(d):
    p=d["prompt"]; L=p.str.len().clip(lower=1)
    return np.c_[np.log1p(L), p.str.count(r"[가-힣]")/L, p.str.count(r"[0-9]")/L,
                 p.str.count(r"[+\-*/=^]")/L, p.str.count(r"[(){}\[\]]")/L,
                 p.str.count(r"\n")/L, p.str.contains(r"def |return |import ").astype(float),
                 (L>2000).astype(float)]
vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,4),max_features=30000,min_df=3,sublinear_tf=True)
X=sp.hstack([vec.fit_transform(tr["prompt"]),nf(tr)]).tocsr()
Xd=sp.hstack([vec.transform(dv["prompt"]),nf(dv)]).tocsr()
p=Ridge(alpha=1).fit(X,tr["gt"]).predict(Xd)
print(f"\nthink 이득 예측 상관: {np.corrcoef(p, dv['gt'])[0,1]:.3f}  (ax31은 0.101이었음)")

# 그룹 평균만으로 예측했을 때
gmean=tr.groupby("g")["gt"].mean()
pg=dv["g"].map(gmean).values
print(f"그룹 평균만으로  상관: {np.corrcoef(pg, dv['gt'])[0,1]:.3f}")
