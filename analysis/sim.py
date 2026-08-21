# SPDX-FileCopyrightText: Copyright 2026 뭘했음청년들
# SPDX-License-Identifier: Apache-2.0
import pandas as pd
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
base=df["cost_ax31-light"].sum(); n=len(df)

# 그룹 우선순위 (효율 높은 순)
ORDER=[("한국어","ax31"),("코드","ax31"),("숫자많음","ax31"),
       ("일반영어","ax31"),("코드","axk1-think"),("숫자많음","axk1-think"),
       ("한국어","axk1-think"),("일반영어","axk1-think")]

for tier,mult in [("fast",1.25),("balanced",2.0),("premium",4.0)]:
    budget=base*mult; cost=base; gain=0.0; done=set()
    for grp,m in ORDER:
        sub=df[(df["g"]==grp)&(~df["episode_id"].isin(done))]
        # 그룹 안에서도 가성비 순으로
        sub=sub.assign(eff=sub[f"g_{m}"]/sub[f"e_{m}"]).sort_values("eff",ascending=False)
        for _,r in sub.iterrows():
            if r[f"g_{m}"]<=0 or r[f"e_{m}"]<=0: continue
            if cost+r[f"e_{m}"]<=budget:
                cost+=r[f"e_{m}"]; gain+=r[f"g_{m}"]; done.add(r["episode_id"])
    print(f"[{tier}] {(df['score_ax31-light'].sum()+gain)/n:.4f}  "
          f"(예산 {cost/base:.3f}x, 승급 {len(done)}개)")
