import pandas as pd, re
df = pd.read_csv("build/train-analysis.csv")
p = df["prompt"].fillna("")

def frac(pat): return p.str.count(pat) / p.str.len().clip(lower=1)

df["kor"]   = frac(r"[가-힣]")
df["digit"] = frac(r"[0-9]")
df["math"]  = frac(r"[+\-*/=^()<>]")
df["code"]  = p.str.contains(r"def |return |import |\bclass\b", regex=True)
df["long"]  = df["chars"] > 2000

def group(r):
    if r["long"]:            return "긴문맥(2000자+)"
    if r["code"]:            return "코드"
    if r["kor"] > 0.3:       return "한국어"
    if r["digit"] > 0.05:    return "숫자많음"
    if r["math"] > 0.05:     return "수식많음"
    return "일반영어"

df["g"] = df.apply(group, axis=1)
out = df.groupby("g").agg(
    수=("episode_id","size"),
    light=("score_ax31-light","mean"),
    ax31=("score_ax31","mean"),
    think=("score_axk1-think","mean"),
    승급이득=("g_ax31","mean"),
    출력토큰=("out_ax31-light","median"),
).sort_values("light")
pd.set_option("display.width", 200)
print(out.round(3))
