#!/bin/bash
# SPDX-FileCopyrightText: Copyright 2026 뭘했음청년들
# SPDX-License-Identifier: Apache-2.0
# 2,640문항을 5폴드로 나눠 학습/평가
python3 - << 'PYEOF'
import json
inp = json.load(open('build/inputs-all.json'))
out = json.load(open('build/outcomes-all.json'))
eps_i = inp['episodes']; eps_o = {e['episode_id']: e for e in out['episodes']}
n = len(eps_i)
for k in range(5):
    te = [e for i, e in enumerate(eps_i) if i % 5 == k]
    tr = [e for i, e in enumerate(eps_i) if i % 5 != k]
    for name, eps in [('tr', tr), ('te', te)]:
        json.dump({**inp, 'split': f'cv{k}{name}', 'episodes': eps},
                  open(f'build/cv{k}-{name}-in.json', 'w'))
        json.dump({**out, 'split': f'cv{k}{name}',
                   'episodes': [eps_o[e['episode_id']] for e in eps]},
                  open(f'build/cv{k}-{name}-out.json', 'w'))
    print(k, len(tr), len(te))
PYEOF
