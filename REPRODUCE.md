# 재현 절차

공개 Dev 880문항에서 제출 라우터의 점수와 예산 비율을 확인하는 방법입니다.
필요한 것은 Python 3.10 이상과 git뿐입니다. Docker는 없어도 됩니다.

## 1. 저장소 받기

```bash
git clone https://github.com/jbk-01/ossp-2026-llm-router-challenge.git
cd ossp-2026-llm-router-challenge
```

## 2. 공개 Train/Dev 자료 생성

AIME 원문은 라이선스상 저장소에 포함할 수 없어 각자 내려받아 결합합니다.
인터넷 연결이 필요하며 몇 분 걸립니다.

```bash
python3 -m venv .venv-data
.venv-data/bin/pip install -r data/sources/requirements-materialize-public-data.txt
.venv-data/bin/python tools/materialize_public_data.py
```

정상 종료 시 train 1760, dev 880 문항이 생성됩니다.

## 3. 라우터 실행 (세 등급)

컨테이너 진입점과 동일한 경로로 실행합니다.

```bash
for tier in fast balanced premium; do
  PYTHONPATH=src python3 container/entrypoint.py \
    --input data/materialized/dev/inputs.json \
    --tier "$tier" \
    --output "build/verify/$tier.json"
done
```

## 4. 공식 채점 도구로 검증

```bash
PYTHONPATH=src python3 -m ossp_router.cli self-check \
  --input data/materialized/dev/inputs.json \
  --outcomes data/dev/outcomes.json \
  --submissions build/verify \
  --report build/verify-report.json
```

## 5. 결과 확인

```bash
python3 -c "
import json
r = json.load(open('build/verify-report.json'))
print('final', r['final_score'])
for t in ['fast', 'balanced', 'premium']:
    d = r['tiers'][t]
    print(t, d['tier_score'][:6], d['budget_ratio'][:5], d['budget_passed'])
"
```

### 기대 출력

```
final 0.689630681818
fast 0.6559 1.182 True
balanced 0.6903 1.843 True
premium 0.7338 3.711 True
```

## 비교 기준

| 구현 | Dev 최종 | premium 비율 |
| --- | ---: | ---: |
| all-light (하한) | 0.619318 | 1.000 |
| prompt-heuristic | 0.655341 | 2.102 |
| feature-budget | 0.643011 | 2.102 |
| hash-regex (저장소 제공) | 0.695369 | 3.985 |
| 본 제출 | 0.689630 | 3.711 |
| 오라클 (도달 불가 상한) | 약 0.784 | - |

저장소 제공 hash-regex보다 최종 점수가 0.0057 낮은 대신 Premium 예산 여유를
2.8%에서 7.2%로 늘렸습니다. baselines/README.md에 따르면 hash-regex는
채점용 평가셋에서 비용 비율이 약 4.2로 나타나 Premium이 0점 처리된 전례가
있습니다. 등급 하나가 0점이 되면 가중치 0.3에 해당하는 약 0.22를 잃으므로,
점수 0.0075를 지불해 그 위험을 줄이는 선택을 했습니다.

## 컨테이너로 확인하려면

x86 머신에서는 먼저 에뮬레이터를 등록해야 하며, 재부팅하면 사라집니다.

```bash
docker run --privileged --rm tonistiigi/binfmt --install arm64

docker run --rm \
  -v "$PWD/data/materialized/dev:/challenge/input:ro" \
  -v "$PWD/build/container-out:/challenge/output" \
  docker.io/jbk010302/ossp-router@sha256:f4f398b5db829dd3a04ff00260bd466d19594e1f070445e1b30d4c6bcb42b4cc \
  --input /challenge/input/inputs.json --tier fast \
  --output /challenge/output/submission.json
```

tools/check_runtime.py의 실행 시간 측정은 QEMU 에뮬레이션 환경에서
12~18배 느리게 나오므로 합격 여부 판정에 사용할 수 없습니다. 공식
Apple Silicon 환경 참고값은 docs/runtime-benchmark.md에 있으며,
저장소 제공 hash-regex가 90초 한도에서 7.3초입니다.

## 제출 정보

- 저장소: https://github.com/jbk-01/ossp-2026-llm-router-challenge
- 코드 커밋: fa91335aee208b78c6080f97962d7302d04ea0b4
- 이미지: docker.io/jbk010302/ossp-router@sha256:f4f398b5db829dd3a04ff00260bd466d19594e1f070445e1b30d4c6bcb42b4cc
