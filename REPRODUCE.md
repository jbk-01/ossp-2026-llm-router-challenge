<!--
SPDX-FileCopyrightText: Copyright 2026 뭘했음청년들
SPDX-License-Identifier: Apache-2.0
-->

# 재현 절차

공개 Dev 880문항에서 제출 라우터의 점수와 예산 비율을 확인하는 방법입니다.
필요한 것은 Python 3.10 이상과 git뿐이며 Docker는 없어도 됩니다.

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

정상 종료 시 train 1,760문항과 dev 880문항이 생성됩니다.

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

기대 출력

```
final 0.705568181818
fast 0.6610 1.062 True
balanced 0.7173 1.696 True
premium 0.7531 2.910 True
```

## 설계 요약

| 항목 | 값 |
| --- | --- |
| 학습 자료 | 공개 Train + Dev 2,640문항 |
| 특징 | 정규식 특징 + 단어 unigram/bigram signed feature hashing |
| 예측기 | ridge 회귀 6개 head (모델 3종 x {score, log-cost}) |
| 비용 편향 보정 | light 1.222 / ax31 1.154 / think 1.461 |
| think 점수 증폭 | 2.0 |
| 등급 안전계수 | fast 0.85 / balanced 0.85 / premium 0.82 |
| 특징 추출 문자 상한 | 8,000자 |

## 비교 기준

### 공개 Dev 단독 (참고값)

본 제출은 Train+Dev 2,640문항을 모두 학습에 사용했으므로 아래 Dev 점수는
학습 자료에 대한 값이며 일반화 성능이 아닙니다.

| 구현 | Dev 점수 | premium 비율 |
| --- | ---: | ---: |
| all-light (하한) | 0.619318 | 1.000 |
| prompt-heuristic | 0.655341 | 2.102 |
| feature-budget | 0.643011 | 2.102 |
| hash-regex (저장소 제공) | 0.695369 | 3.985 |
| 본 제출 (참고값) | 0.705568 | 2.910 |

### 5-폴드 교차검증 (일반화 성능)

Train+Dev 2,640문항을 5등분해 4/5로 학습하고 나머지 1/5(528문항)에서
평가했습니다. 각 폴드의 평가 문항은 해당 폴드 학습에 사용되지 않았으며,
비용 편향 보정 계수도 각 폴드의 학습셋에서만 산출했습니다.

| 구성 | 폴드 평균 | 예산 초과 폴드 |
| --- | ---: | ---: |
| 참조 baseline 안전계수 (0.9483 / 0.9167 / 0.9250) | 0.5316 | 2 / 5 |
| 본 제출 (0.85 / 0.85 / 0.82) | 0.6605 | 0 / 5 |

참조 baseline 설정은 폴드 2에서 fast(1.312)와 premium(4.254)이 동시에
한도를 넘어 0.2047까지 하락했고, 폴드 3에서도 premium이 4.110으로
초과했습니다. 본 제출은 다섯 폴드 모두 0.636~0.673 범위를 유지했으며
예산 초과가 없었습니다.

재현 절차는 `analysis/cv.sh`(폴드 분할), `analysis/cvrun.sh`(폴드별 학습),
`analysis/cveval.sh`(홀드아웃 평가)에 있습니다.

### 설계 근거

저장소 `baselines/README.md`에 따르면 hash-regex baseline은 채점용
평가셋에서 비용 비율이 약 4.2로 나타나 Premium 등급이 0점 처리된 전례가
있습니다. 본 제출은 이를 비용 추정의 구조적 과소평가 문제로 진단하고,
모델별 실측 배율로 교정한 뒤 교차검증으로 안전계수를 결정했습니다.
등급 하나가 0점 처리되면 약 0.22를 잃으므로, 공개 Dev 점수를 일부
포기하더라도 예산 준수를 우선했습니다.

## 컨테이너로 확인하려면

x86 머신에서는 먼저 에뮬레이터를 등록해야 하며 재부팅하면 사라집니다.

```bash
docker run --privileged --rm tonistiigi/binfmt --install arm64

docker run --rm \
  -v "$PWD/data/materialized/dev:/challenge/input:ro" \
  -v "$PWD/build/container-out:/challenge/output" \
  docker.io/jbk010302/ossp-router@sha256:865ba83fbdd7f4513253bba445b29970a4bb4d792d6a54d84f44bdb6c8b493f9 \
  --input /challenge/input/inputs.json --tier fast \
  --output /challenge/output/submission.json
```

`tools/check_runtime.py`의 실행 시간 측정은 QEMU 에뮬레이션 환경에서
12~18배 느리게 나오므로 합격 여부 판정에 사용할 수 없습니다. 공식
Apple Silicon 환경 참고값은 `docs/runtime-benchmark.md`에 있으며,
저장소 제공 hash-regex가 90초 한도에서 7.3초입니다. 본 제출은 로컬
x86 기준 Train 1,760문항 처리에 4.4초가 걸립니다.

## 제출 정보

- 저장소: https://github.com/jbk-01/ossp-2026-llm-router-challenge
- 코드 커밋: dc701a861606753dd1b5a98161e901ef1078fb07
- 이미지: docker.io/jbk010302/ossp-router@sha256:865ba83fbdd7f4513253bba445b29970a4bb4d792d6a54d84f44bdb6c8b493f9
- 기술 제출 정보 검증:

```bash
python3 tools/validate_technical_submission.py
```
