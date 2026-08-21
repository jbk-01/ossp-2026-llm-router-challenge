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
RATE={"ax31-light":(1,4),"ax31":(2.127,8.509),"axk1-think":(6.565,26.26)}
M=["ax31-light","ax31","axk1-think"]
df["in_true"]=df["cost_ax31-light"]*1e6-4*df["out_ax31-light"]
IN=(df["in_true"]/df["chars"]).groupby(df["g"]).mean()
OUT={m:df.groupby("g")[f"out_{m}"].mean() for m in M}
SC={m:df.groupby("g")[f"score_{m}"].mean() for m in M}

def est(grp,ch,m):
    return (ch*IN[grp]*RATE[m][0]+OUT[m][grp]*RATE[m][1])/1e6

for tier,mult in [("fast",1.25),("balanced",2.0),("premium",4.0)]:
  for safety in [0.98,0.95,0.90]:
    base=sum(est(r["g"],r["chars"],"ax31-light") for _,r in df.iterrows())
    budget=base*mult*safety
    # 2단계 후보: (light->ax31), (ax31->think)
    step1,step2=[],{}
    for i,r in df.iterrows():
        grp,ch=r["g"],r["chars"]
        c0,c1,c2=(est(grp,ch,m) for m in M)
        g1=SC["ax31"][grp]-SC["ax31-light"][grp]
        g2=SC["axk1-think"][grp]-SC["ax31"][grp]
        if g1>0 and c1>c0: step1.append((g1/(c1-c0),i,c1-c0))
        if g2>0 and c2>c1: step2[i]=(g2/(c2-c1),g2,c2-c1)
    step1.sort(key=lambda x:-x[0])
    pick={}; spent=base
    for _,i,e in step1:
        if spent+e<=budget: spent+=e; pick[i]="ax31"
    # 남은 예산으로 think 승급
    s2=sorted(((v[0],i,v[2]) for i,v in step2.items() if pick.get(i)=="ax31"),
              key=lambda x:-x[0])
    for _,i,e in s2:
        if spent+e<=budget: spent+=e; pick[i]="axk1-think"
    sc=sum(df.loc[i,f"score_{pick.get(i,'ax31-light')}"] for i in df.index)/len(df)
    real=sum(df.loc[i,f"cost_{pick.get(i,'ax31-light')}"] for i in df.index)
    ratio=real/df["cost_ax31-light"].sum()
    nt=sum(1 for v in pick.values() if v=="axk1-think")
    print(f"[{tier}] s={safety}  점수 {sc:.4f}  비율 {ratio:.3f}/{mult}  "
          f"{'OK' if ratio<=mult else '초과!'}  ax31 {len(pick)-nt} think {nt}")
  print()
