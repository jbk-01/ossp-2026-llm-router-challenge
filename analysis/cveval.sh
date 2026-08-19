#!/bin/bash
for b in 1.0; do
echo "=== think 증폭 b=$b ==="
for k in 0 1 2 3 4; do
  python3 - << PYEOF
import json, math, sys
from pathlib import Path
sys.path.insert(0, 'src')
from ossp_router.hash_regex import parse_artifact, predict_episode
from ossp_router.protocol import load_input
RATE={'ax31-light':(1,4),'ax31':(2.127,8.509),'axk1-think':(6.565,26.26)}
raw=json.load(open('build/cv$k-art.json'))
art=parse_artifact(raw)
# 보정계수는 학습셋에서만 측정
inp=load_input(Path('build/cv$k-tr-in.json'))
out=json.load(open('build/cv$k-tr-out.json'))
act={m:0.0 for m in RATE}
for ep in out['episodes']:
    for m,(ri,ro) in RATE.items():
        d=ep['models'][m]; act[m]+=(d['input_tokens']*ri+d['output_tokens']*ro)/1e6
pre={m:0.0 for m in RATE}
for ep in inp.episodes:
    _,c=predict_episode(ep,art)
    for m in RATE: pre[m]+=c[m]
for m in RATE:
    raw['log_cost_heads'][m]['intercept'] += math.log(act[m]/pre[m])
h=raw['score_heads']; L=h['ax31-light']; m='axk1-think'; f=$b
h[m]['intercept']=L['intercept']+f*(h[m]['intercept']-L['intercept'])
h[m]['coefficients']=[lc+f*(c-lc) for c,lc in zip(h[m]['coefficients'],L['coefficients'])]
raw['tier_safety_ratios']={'fast':0.9483,'balanced':0.9167,'premium':0.9250}
json.dump(raw, open('src/ossp_router/resources/artifact.v1.json','w'))
PYEOF
  for t in fast balanced premium; do
    PYTHONPATH=src python3 container/entrypoint.py --input build/cv$k-te-in.json --tier $t --output build/cve/$t.json > /dev/null
  done
  PYTHONPATH=src python3 -m ossp_router.cli self-check --input build/cv$k-te-in.json --outcomes build/cv$k-te-out.json --submissions build/cve --report build/cve-rep.json > /dev/null
  python3 -c "
import json;r=json.load(open('build/cve-rep.json'));t=r['tiers']
ok=all(t[x]['budget_passed'] for x in t)
print(f\"  폴드$k  {r['final_score'][:8]}  f {t['fast']['budget_ratio'][:5]} b {t['balanced']['budget_ratio'][:5]} p {t['premium']['budget_ratio'][:5]}  {'OK' if ok else '초과!'}\")
"
done
done
