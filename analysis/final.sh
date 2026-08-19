#!/bin/bash
for p in 0.94 0.85 0.78 0.72; do
  python3 -c "
import json, math
art=json.load(open('build/artifact-all.json'))
# 비용 편향 보정 (Train+Dev 실측 배율)
for m,k in [('ax31-light',1.2158),('ax31',1.1571),('axk1-think',1.4599)]:
    art['log_cost_heads'][m]['intercept'] += math.log(k)
# think 점수 증폭
h=art['score_heads']; L=h['ax31-light']; m='axk1-think'; f=2.0
h[m]['intercept']=L['intercept']+f*(h[m]['intercept']-L['intercept'])
h[m]['coefficients']=[lc+f*(c-lc) for c,lc in zip(h[m]['coefficients'],L['coefficients'])]
art['tier_safety_ratios']={'fast':0.88,'balanced':0.90,'premium':$p}
json.dump(art, open('src/ossp_router/resources/artifact.v1.json','w'))
"
  for split in train dev; do
    for t in fast balanced premium; do
      PYTHONPATH=src python3 container/entrypoint.py --input data/materialized/$split/inputs.json --tier $t --output build/fn/$t.json > /dev/null
    done
    PYTHONPATH=src python3 -m ossp_router.cli self-check --input data/materialized/$split/inputs.json --outcomes data/$split/outcomes.json --submissions build/fn --report build/fn-report.json > /dev/null
    python3 -c "
import json;r=json.load(open('build/fn-report.json'));t=r['tiers']
print(f\"  prem_s=$p [$split] 최종 {r['final_score'][:8]}  f {t['fast']['budget_ratio'][:5]} b {t['balanced']['budget_ratio'][:5]} p {t['premium']['budget_ratio'][:5]}\")
"
  done
done
