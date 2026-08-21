# SPDX-FileCopyrightText: Copyright 2026 뭘했음청년들
# SPDX-License-Identifier: Apache-2.0
import pandas as pd, numpy as np
df = pd.read_csv("build/train-analysis.csv")
p = df["prompt"].fillna("")
def frac(pat): return p.str.count(pat)/p.str.len().clip(lower=1)
df["kor"]=frac(r"[가-힣]"); df["digit"]=frac(r"[0-9]")
df["code"]=p.str.contains(r"def |return |import |\bclass\b",regex=True)
def g(r):
    if r["chars"]>2000: return "긴문맥"
    if r["code"]: return "코드"
    if r["kor"]>0.3: return "한국어"
    if r["digit"]>0.05: return "숫자많음"
    return "일반영어"
df["g"]=df.apply(g,axis=1)

# 실제 light 비용에서 역산한 토큰 (in*1 + out*4)/1e6
print("=== 그룹별 실제 출력토큰 (light) ===")
print(df.groupby("g")["out_ax31-light"].describe()[["50%","mean","std"]].round(0))
print("\n=== 글자수 대비 입력토큰 비율 추정 ===")
# cost = (in + 4*out)/1e6 이므로 in = cost*1e6 - 4*out
df["in_est"]=df["cost_ax31-light"]*1e6-4*df["out_ax31-light"]
print((df["in_est"]/df["chars"]).groupby(df["g"]).median().round(3))

# 그룹 중앙값으로 비용 추정했을 때 오차
med=df.groupby("g")["out_ax31-light"].transform("median")
ratio=(df["in_est"]/df["chars"]).groupby(df["g"]).transform("median")
est=(df["chars"]*ratio+4*med)/1e6
print(f"\n총 비용 실제 {df['cost_ax31-light'].sum():.3f} vs 추정 {est.sum():.3f} "
      f"({est.sum()/df['cost_ax31-light'].sum():.1%})")
