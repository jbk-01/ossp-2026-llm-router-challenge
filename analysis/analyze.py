# SPDX-FileCopyrightText: Copyright 2026 뭘했음청년들
# SPDX-License-Identifier: Apache-2.0
import json
from decimal import Decimal
import pandas as pd

TOKEN_UNIT = 1_000_000
RATES = {
    "ax31-light": {"in": 1.0, "out": 4.0},
    "ax31": {"in": 2.127, "out": 8.509},
    "axk1-think": {"in": 6.565, "out": 26.260},
}
MODELS = list(RATES)
TIERS = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
WEIGHTS = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}

def cost(m, i, o):
    r = RATES[m]
    return (i * r["in"] + o * r["out"]) / TOKEN_UNIT

with open("data/train/outcomes.json", encoding="utf-8") as f:
    outcomes = json.load(f)
with open("data/materialized/train/inputs.json", encoding="utf-8") as f:
    inputs = json.load(f)

text = {}
for ep in inputs["episodes"]:
    text[ep["episode_id"]] = ep.get("prompt") or "\n".join(
        m["content"] for m in ep["messages"])

rows = []
for ep in outcomes["episodes"]:
    r = {"episode_id": ep["episode_id"]}
    for m in MODELS:
        d = ep["models"][m]
        r[f"score_{m}"] = float(Decimal(d["score"]))
        r[f"cost_{m}"] = cost(m, d["input_tokens"], d["output_tokens"])
        r[f"out_{m}"] = d["output_tokens"]
    rows.append(r)

df = pd.DataFrame(rows)
df["prompt"] = df["episode_id"].map(text)
df["chars"] = df["prompt"].str.len()
n = len(df)
base = df["cost_ax31-light"].sum()
print(f"문항 수: {n}\n")

print("=== 모델별 전체 사용 ===")
for m in MODELS:
    print(f"{m:<12} 평균 {df[f'score_{m}'].mean():.4f}   비용 {df[f'cost_{m}'].sum()/base:.2f}x")

print("\n=== 점수 분포 ===")
for m in MODELS:
    s = df[f"score_{m}"]
    print(f"{m:<12} 만점 {(s>=1).mean():6.1%}   0점 {(s<=0).mean():6.1%}")

best = df[["score_ax31-light","score_ax31","score_axk1-think"]].max(axis=1)
print(f"\nlight로 충분한 문항: {(df['score_ax31-light']>=best).mean():.1%}")

for m in ["ax31", "axk1-think"]:
    df[f"g_{m}"] = df[f"score_{m}"] - df["score_ax31-light"]
    df[f"e_{m}"] = df[f"cost_{m}"] - df["cost_ax31-light"]

cands = []
for _, r in df.iterrows():
    for m in ["ax31", "axk1-think"]:
        if r[f"g_{m}"] > 0 and r[f"e_{m}"] > 0:
            cands.append((r[f"g_{m}"]/r[f"e_{m}"], r["episode_id"], m,
                          r[f"g_{m}"], r[f"e_{m}"]))
cands.sort(reverse=True)

light = df["score_ax31-light"].mean()
lo = up_total = 0.0
print("\n=== 등급별 상한선 ===")
for t, mult in TIERS.items():
    budget, c, gain, used = base*mult, base, 0.0, {}
    for _, eid, m, g, e in cands:
        if eid in used: continue
        if c + e <= budget:
            c += e; gain += g; used[eid] = m
    sc = (df["score_ax31-light"].sum() + gain)/n
    a = sum(1 for v in used.values() if v=="ax31")
    k = sum(1 for v in used.values() if v=="axk1-think")
    print(f"[{t}] {mult}x  하한 {light:.4f} -> 상한 {sc:.4f}  "
          f"(예산 {c/base:.3f}x, ax31 {a}개 think {k}개 = {(a+k)/n:.1%})")
    lo += WEIGHTS[t]*light; up_total += WEIGHTS[t]*sc

print(f"\n최종 하한 {lo:.4f} / 상한 {up_total:.4f} / 폭 {up_total-lo:+.4f}")

top = df.nlargest(200, "g_ax31")
print("\n=== ax31 승급효과 상위200 vs 전체 ===")
print(f"길이(중앙값)  {top['chars'].median():8.0f} vs {df['chars'].median():.0f}")
print(f"light 점수    {top['score_ax31-light'].mean():8.3f} vs {df['score_ax31-light'].mean():.3f}")
print(f"출력토큰      {top['out_ax31-light'].median():8.0f} vs {df['out_ax31-light'].median():.0f}")

df.to_csv("build/train-analysis.csv", index=False)
print("\n저장: build/train-analysis.csv")
