import json, math, sys
from pathlib import Path
sys.path.insert(0, 'src')
from ossp_router.hash_regex import parse_artifact, predict_episode
from ossp_router.protocol import load_input

RATE = {'ax31-light': (1, 4), 'ax31': (2.127, 8.509), 'axk1-think': (6.565, 26.26)}
SRC = 'src/ossp_router/resources/artifact.v1.json'

raw = json.load(open(SRC))
art = parse_artifact(raw)
inputs = load_input(Path('data/materialized/train/inputs.json'))
outcomes = json.load(open('data/train/outcomes.json'))

actual = {m: 0.0 for m in RATE}
for ep in outcomes['episodes']:
    for m, (ri, ro) in RATE.items():
        d = ep['models'][m]
        actual[m] += (d['input_tokens'] * ri + d['output_tokens'] * ro) / 1e6

pred = {m: 0.0 for m in RATE}
for ep in inputs.episodes:
    _, c = predict_episode(ep, art)
    for m in RATE:
        pred[m] += c[m]

print('=== Train 총비용 비교 ===')
for m in RATE:
    k = actual[m] / pred[m]
    print(f'{m:<12} 실제 {actual[m]:9.4f}  예측 {pred[m]:9.4f}  보정계수 {k:.4f}')

for m in RATE:
    k = actual[m] / pred[m]
    raw['log_cost_heads'][m]['intercept'] += math.log(k)

json.dump(raw, open('build/artifact-cal.json', 'w'))
print('\n저장: build/artifact-cal.json')
