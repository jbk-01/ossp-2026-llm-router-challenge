# SPDX-FileCopyrightText: Copyright 2026 뭘했음청년들
# SPDX-License-Identifier: Apache-2.0
import pandas as pd, re
df = pd.read_csv("build/train-analysis.csv")
p = df["prompt"].fillna("")
def frac(pat): return p.str.count(pat) / p.str.len().clip(lower=1)
df["kor"]=frac(r"[가-힣]"); df["digit"]=frac(r"[0-9]")
df["math"]=frac(r"[+\-*/=^()<>]")
df["code"]=p.str.contains(r"def |return |import |\bclass\b", regex=True)
def group(r):
    if r["chars"]>2000: return "긴문맥"
    if r["code"]: return "코드"
    if r["kor"]>0.3: return "한국어"
    if r["digit"]>0.05: return "숫자많음"
    if r["math"]>0.05: return "수식많음"
    return "일반영어"
df["g"]=df.apply(group,axis=1)
df["eff_a"]=df["g_ax31"]/df["e_ax31"].clip(lower=1e-12)
df["eff_t"]=df["g_axk1-think"]/df["e_axk1-think"].clip(lower=1e-12)
out=df.groupby("g").agg(
  수=("episode_id","size"),
  light비용=("cost_ax31-light","mean"),
  ax31추가=("e_ax31","mean"),
  think추가=("e_axk1-think","mean"),
  ax31가성비=("eff_a","mean"),
  think가성비=("eff_t","mean"),
).sort_values("think가성비",ascending=False)
pd.set_option("display.width",250)
print(out.round(6))
print("\n전체 대비 각 그룹의 light 비용 점유율(%)")
print((df.groupby("g")["cost_ax31-light"].sum()/df["cost_ax31-light"].sum()*100).round(1))
