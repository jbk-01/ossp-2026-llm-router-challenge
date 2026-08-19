#!/bin/bash
for fs in 0.88 0.94 1.00; do
for ps in 0.94 1.00 1.06; do
  python3 -c "
import json
art=json.load(open('build/artifact-all-cal.json'))
h=art['score_heads']; L=h['ax31-light']; m='axk1-think'; f=2.0
h[m]['intercept']=L['intercept']+f*(h[m]['intercept']-L['intercept'])
h[m]['coefficients']=[lc+f*(c-lc) for c,lc in zip(h[m]['coefficients'],L['coefficients'])]
art['tier_safety_ratios']={'fast':$fs,'balanced':$fs,'premium':$ps}
json.dump(art, open('src/ossp_router/resources/artifact.v1.json','w'))
"
  echo "--- fast/bal=$fs premium=$ps ---"
  for split in train dev; do
    for t in fast balanced premium; do
      PYTHONPATH=src python3 container/entrypoint.py --input data/materialized/$split/inputs.json --tier $t --output build/tn/$t.json > /dev/null
    done
    PYTHONPATH=src python3 -m ossp_router.cli self-check --input data/materialized/$split/inputs.json --outcomes data/$split/outcomes.json --submissions build/tn --report build/tn-report.json > /dev/null
    python3 -c "
import json;r=json.load(open('build/tn-report.json'));t=r['tiers']
ok=all(t[x]['budget_passed'] for x in t)
print(f\"   [$split] {r['final_score'][:8]}  f {t['fast']['budget_ratio'][:5]} b {t['balanced']['budget_ratio'][:5]} p {t['premium']['budget_ratio'][:5]}  {'OK' if ok else '초과!'}\")
"
  done
done
done
