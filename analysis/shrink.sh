#!/bin/bash
for a in 1.0; do
for b in 1.6 2.0 2.5 3.0; do
  python3 -c "
import json
art=json.load(open('build/artifact-cal.json'))
h=art['score_heads']; L=h['ax31-light']
for m,f in [('ax31',$a),('axk1-think',$b)]:
    h[m]['intercept']=L['intercept']+f*(h[m]['intercept']-L['intercept'])
    h[m]['coefficients']=[lc+f*(c-lc) for c,lc in zip(h[m]['coefficients'],L['coefficients'])]
art['tier_safety_ratios']={'fast':0.88,'balanced':0.90,'premium':0.94}
json.dump(art, open('src/ossp_router/resources/artifact.v1.json','w'))
"
  for t in fast balanced premium; do
    PYTHONPATH=src python3 container/entrypoint.py --input data/materialized/dev/inputs.json --tier $t --output build/sh/$t.json > /dev/null
  done
  PYTHONPATH=src python3 -m ossp_router.cli self-check --input data/materialized/dev/inputs.json --outcomes data/dev/outcomes.json --submissions build/sh --report build/sh-report.json > /dev/null
  python3 -c "
import json;r=json.load(open('build/sh-report.json'));t=r['tiers']
print(f\"a=$a b=$b  최종 {r['final_score'][:8]}  f {t['fast']['budget_ratio'][:5]} b {t['balanced']['budget_ratio'][:5]} p {t['premium']['tier_score'][:6]}/{t['premium']['budget_ratio'][:5]}\")
"
done
done
