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

RATE={"ax31-light":(1,4),"ax31":(2.127,8.509),"axk1-think":(6.565,26.26)}
M=["ax31-light","ax31","axk1-think"]

# --- 공개 Train에서 학습한 그룹별 통계 (라우터에 내장할 표) ---
df["in_true"]=df["cost_ax31-light"]*1e6-4*df["out_ax31-light"]
IN_RATIO=(df["in_true"]/df["chars"]).groupby(df["g"]).mean()
OUT={m:df.groupby("g")[f"out_{m}"].mean() for m in M}   # 평균 사용
GAIN={m:df.groupby("g")[f"score_{m}"].mean()-df.groupby("g")["score_ax31-light"].mean()
      for m in ["ax31","axk1-think"]}

# --- 여기부터가 라우터가 실제로 하는 일 (프롬프트 정보만 사용) ---
def est_cost(grp, chars, m):
    i=chars*IN_RATIO[grp]; o=OUT[m][grp]
    return (i*RATE[m][0]+o*RATE[m][1])/1e6

for tier,mult in [("fast",1.25),("balanced",2.0),("premium",4.0)]:
  for safety in [1.00, 0.95, 0.90]:
    est_base=sum(est_cost(r["g"],r["chars"],"ax31-light") for _,r in df.iterrows())
    budget=est_base*mult*safety
    cands=[]
    for i,r in df.iterrows():
        c0=est_cost(r["g"],r["chars"],"ax31-light")
        for m in ["ax31","axk1-think"]:
            gain=GAIN[m][r["g"]]; extra=est_cost(r["g"],r["chars"],m)-c0
            if gain>0 and extra>0: cands.append((gain/extra,i,m,extra))
    cands.sort(key=lambda x:-x[0])
    pick={}; spent=est_base
    for _,i,m,extra in cands:
        if i in pick: continue
        if spent+extra<=budget: spent+=extra; pick[i]=m
    # 실제 점수/비용으로 채점
    sc=sum(df.loc[i,f"score_{pick.get(i,'ax31-light')}"] for i in df.index)/len(df)
    real=sum(df.loc[i,f"cost_{pick.get(i,'ax31-light')}"] for i in df.index)
    ratio=real/df["cost_ax31-light"].sum()
    ok="OK" if ratio<=mult else "예산초과!"
    print(f"[{tier}] safety={safety:.2f}  점수 {sc:.4f}  실제비율 {ratio:.3f}/{mult}  {ok}  승급 {len(pick)}")
  print()
