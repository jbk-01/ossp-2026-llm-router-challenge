#!/bin/bash
# SPDX-FileCopyrightText: Copyright 2026 뭘했음청년들
# SPDX-License-Identifier: Apache-2.0
for infl in 1.0 1.3 1.6 2.0; do
for s in 0.90 0.94; do
  python3 -c "
import json, math
a=json.load(open('build/artifact-cal.json'))
a['log_cost_heads']['axk1-think']['intercept'] += math.log($infl)
a['tier_safety_ratios']={'fast':$s,'balanced':$s,'premium':$s}
json.dump(a, open('src/ossp_router/resources/artifact.v1.json','w'))
"
  for t in fast balanced premium; do
    PYTHONPATH=src python3 container/entrypoint.py --input data/materialized/dev/inputs.json --tier $t --output build/sw/$t.json > /dev/null
  done
  PYTHONPATH=src python3 -m ossp_router.cli self-check --input data/materialized/dev/inputs.json --outcomes data/dev/outcomes.json --submissions build/sw --report build/sw-report.json > /dev/null
  python3 -c "
import json;r=json.load(open('build/sw-report.json'))
t=r['tiers']
print(f\"infl=$infl s=$s  최종 {r['final_score'][:8]}  \"
      f\"f {t['fast']['budget_ratio'][:5]} b {t['balanced']['budget_ratio'][:5]} p {t['premium']['budget_ratio'][:5]}\")
"
done
done
