import json, pandas as pd
dv = pd.read_csv("build/dev.csv")
dv["g"] = dv["score_ax31"] - dv["score_ax31-light"]
dv["e"] = dv["cost_ax31"] - dv["cost_ax31-light"]
sub = json.load(open("build/rm82/fast.json"))
pick = {d["episode_id"]: d["model_id"] for d in sub["decisions"]}
dv["sel"] = dv["episode_id"].map(pick)
up = dv[dv["sel"] != "ax31-light"]
print(f"승급 {len(up)}개 / 전체 {len(dv)}")
print(f"  실제 이득 합계   {up['g'].sum():8.2f}")
print(f"  헛된 승급(이득0) {(up['g']<=0).sum():5d}개 ({(up['g']<=0).mean():.1%})")
print(f"  손해 승급(이득<0){(up['g']<0).sum():5d}개")
# 같은 예산으로 최적 선택했다면?
budget = up["e"].sum()
cand = dv[(dv["g"]>0)&(dv["e"]>0)].copy()
cand["eff"] = cand["g"]/cand["e"]
cand = cand.sort_values("eff", ascending=False)
c=0; gain=0; n=0
for _, r in cand.iterrows():
    if c+r["e"] <= budget: c+=r["e"]; gain+=r["g"]; n+=1
print(f"\n같은 예산 최적선택: {n}개 승급, 이득 {gain:.2f}")
print(f"놓친 이득: {gain - up['g'].sum():.2f}  → 점수로 {(gain-up['g'].sum())/len(dv):.4f}")
