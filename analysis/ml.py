# SPDX-FileCopyrightText: Copyright 2026 뭘했음청년들
# SPDX-License-Identifier: Apache-2.0
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

M = ["ax31-light", "ax31", "axk1-think"]
RATE = {"ax31-light": (1, 4), "ax31": (2.127, 8.509), "axk1-think": (6.565, 26.26)}

tr = pd.read_csv("build/train.csv"); tr["prompt"] = tr["prompt"].fillna("")
dv = pd.read_csv("build/dev.csv");   dv["prompt"] = dv["prompt"].fillna("")

def numfeat(d):
    p = d["prompt"]; L = p.str.len().clip(lower=1)
    return np.c_[np.log1p(L),
                 p.str.count(r"[가-힣]")/L, p.str.count(r"[0-9]")/L,
                 p.str.count(r"[+\-*/=^]")/L, p.str.count(r"[(){}\[\]]")/L,
                 p.str.count(r"\n")/L, p.str.count(r"\?")/L,
                 p.str.contains(r"def |return |import ").astype(float),
                 (L > 2000).astype(float)]

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4),
                      max_features=30000, min_df=3, sublinear_tf=True)
Xtr = sp.hstack([vec.fit_transform(tr["prompt"]), numfeat(tr)]).tocsr()
Xdv = sp.hstack([vec.transform(dv["prompt"]),      numfeat(dv)]).tocsr()

pred = {}
for m in M:
    r = Ridge(alpha=1.0).fit(Xtr, tr[f"score_{m}"])
    pred[f"s_{m}"] = np.clip(r.predict(Xdv), 0, 1)
    r = Ridge(alpha=1.0).fit(Xtr, np.log1p(tr[f"out_{m}"]))
    pred[f"o_{m}"] = np.expm1(r.predict(Xdv)).clip(0)
r = Ridge(alpha=1.0).fit(Xtr, np.log1p(tr["in_ax31-light"]))
pin = np.expm1(r.predict(Xdv)).clip(1)

print("=== 예측 정확도 (Dev) ===")
for m in M:
    print(f"{m:<12} score 상관 {np.corrcoef(pred[f's_{m}'], dv[f'score_{m}'])[0,1]:.3f}  "
          f"out 상관 {np.corrcoef(pred[f'o_{m}'], dv[f'out_{m}'])[0,1]:.3f}")

def run(mult, safety, infl):
    est = {m: (pin*RATE[m][0] + pred[f"o_{m}"]*infl*RATE[m][1])/1e6 for m in M}
    base = est["ax31-light"].sum(); budget = base*mult*safety
    n = len(dv); pick = {}; spent = base
    s1 = [((pred["s_ax31"][i]-pred["s_ax31-light"][i])/(est["ax31"][i]-est["ax31-light"][i]),
           i, est["ax31"][i]-est["ax31-light"][i]) for i in range(n)
          if pred["s_ax31"][i] > pred["s_ax31-light"][i] and est["ax31"][i] > est["ax31-light"][i]]
    for _, i, e in sorted(s1, key=lambda x: -x[0]):
        if spent+e <= budget: spent += e; pick[i] = "ax31"
    s2 = [((pred["s_axk1-think"][i]-pred["s_ax31"][i])/(est["axk1-think"][i]-est["ax31"][i]),
           i, est["axk1-think"][i]-est["ax31"][i]) for i in pick
          if pred["s_axk1-think"][i] > pred["s_ax31"][i] and est["axk1-think"][i] > est["ax31"][i]]
    for _, i, e in sorted(s2, key=lambda x: -x[0]):
        if spent+e <= budget: spent += e; pick[i] = "axk1-think"
    sc = sum(dv.loc[i, f"score_{pick.get(i,'ax31-light')}"] for i in range(n))/n
    real = sum(dv.loc[i, f"cost_{pick.get(i,'ax31-light')}"] for i in range(n))
    return sc, real/dv["cost_ax31-light"].sum()

print("\n=== Dev 성적 (infl=출력토큰 안전배수) ===")
W = {"fast":.4, "balanced":.3, "premium":.3}
for safety in [0.95, 0.90]:
  for infl in [1.0, 1.3, 1.6]:
    tot = 0; line = []
    for t, mult in [("fast",1.25), ("balanced",2.0), ("premium",4.0)]:
        sc, ratio = run(mult, safety, infl)
        ok = ratio <= mult
        line.append(f"{t} {sc:.4f}({ratio:.2f}{'' if ok else ' 초과!'})")
        tot += W[t]*(sc if ok else 0)
    print(f"s={safety} infl={infl}: " + " ".join(line) + f"  최종 {tot:.4f}")

print(f"\nDev 전부 light = {dv['score_ax31-light'].mean():.4f}")
