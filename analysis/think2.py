import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

M=["ax31-light","ax31","axk1-think"]
RATE={"ax31-light":(1,4),"ax31":(2.127,8.509),"axk1-think":(6.565,26.26)}
tr=pd.read_csv("build/train.csv"); tr["prompt"]=tr["prompt"].fillna("")
dv=pd.read_csv("build/dev.csv");   dv["prompt"]=dv["prompt"].fillna("")

def grp(d):
    p=d["prompt"]; L=p.str.len().clip(lower=1)
    kor=p.str.count(r"[가-힣]")/L; dig=p.str.count(r"[0-9]")/L
    code=p.str.contains(r"def |return |import |\bclass\b",regex=True)
    return pd.Series(np.where(d["chars"]>2000,"긴문맥",
        np.where(code,"코드",np.where(kor>0.3,"한국어",
        np.where(dig>0.05,"숫자많음","일반영어")))),index=d.index)
for d in (tr,dv): d["g"]=grp(d)

def nf(d):
    p=d["prompt"]; L=p.str.len().clip(lower=1)
    return np.c_[np.log1p(L),p.str.count(r"[가-힣]")/L,p.str.count(r"[0-9]")/L,
        p.str.count(r"[+\-*/=^]")/L,p.str.count(r"[(){}\[\]]")/L,
        p.str.count(r"\n")/L,p.str.contains(r"def |return |import ").astype(float),
        (L>2000).astype(float)]
vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,4),max_features=30000,min_df=3,sublinear_tf=True)
X=sp.hstack([vec.fit_transform(tr["prompt"]),nf(tr)]).tocsr()
Xd=sp.hstack([vec.transform(dv["prompt"]),nf(dv)]).tocsr()

P={m:Ridge(alpha=1).fit(X,tr[f"score_{m}"]).predict(Xd).clip(0,1) for m in M}
# 비용은 그룹별 평균 토큰으로 추정 (Train에서만 학습)
OUT={m:tr.groupby("g")[f"out_{m}"].mean() for m in M}
IN=(tr["in_ax31-light"]/tr["chars"]).groupby(tr["g"]).mean()
E={m:((dv["chars"]*dv["g"].map(IN))*RATE[m][0]+dv["g"].map(OUT[m])*RATE[m][1]).values/1e6 for m in M}

n=len(dv); base=E["ax31-light"].sum()
Sc={m:dv[f"score_{m}"].values for m in M}; Cs={m:dv[f"cost_{m}"].values for m in M}
W={"fast":.4,"balanced":.3,"premium":.3}

def run(mult,safety):
    budget=base*mult*safety; spent=base; pick={}
    s1=sorted(((P["ax31"][i]-P["ax31-light"][i])/max(E["ax31"][i]-E["ax31-light"][i],1e-12),i)
              for i in range(n) if E["ax31"][i]>E["ax31-light"][i])
    for _,i in sorted(s1,key=lambda x:-x[0]):
        e=E["ax31"][i]-E["ax31-light"][i]
        if spent+e<=budget: spent+=e; pick[i]="ax31"
    s2=sorted(((P["axk1-think"][i]-P["ax31"][i])/max(E["axk1-think"][i]-E["ax31"][i],1e-12),i)
              for i in pick if E["axk1-think"][i]>E["ax31"][i])
    for _,i in sorted(s2,key=lambda x:-x[0]):
        e=E["axk1-think"][i]-E["ax31"][i]
        if spent+e<=budget: spent+=e; pick[i]="axk1-think"
    sc=sum(Sc[pick.get(i,"ax31-light")][i] for i in range(n))/n
    real=sum(Cs[pick.get(i,"ax31-light")][i] for i in range(n))
    nt=sum(1 for v in pick.values() if v=="axk1-think")
    return sc, real/Cs["ax31-light"].sum(), len(pick)-nt, nt

for safety in [0.95,0.90,0.85,0.80]:
    tot=0; out=[]
    for t,mult in [("fast",1.25),("balanced",2.0),("premium",4.0)]:
        sc,r,na,nt=run(mult,safety)
        ok = r<=mult
        out.append(f"{t} {sc:.4f}({r:.2f}{'' if ok else '초과!'} a{na}/t{nt})")
        tot+=W[t]*(sc if ok else 0)
    print(f"s={safety}: "+" ".join(out)+f"  최종 {tot:.4f}")
print("\n현재 제출본 0.6879 / 오라클 0.7837")
