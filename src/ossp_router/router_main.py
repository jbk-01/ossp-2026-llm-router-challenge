# SPDX-FileCopyrightText: Copyright 2026 뭘했음청년들
# SPDX-License-Identifier: Apache-2.0
"""컨테이너 진입점 — 학습 계수를 내장한 hash-regex 라우터.

평가 환경은 네트워크를 제공하지 않으므로 학습 계수(artifact.v1.json)를
이미지에 내장하고 importlib.resources로 읽는다. 따라서 baselines의
원본과 달리 --artifact 경로 인자를 받지 않는다.

라우터는 모델을 호출하지 않고 문항마다 model_id 하나만 선택하며,
선택에는 프롬프트 내용만 사용한다(문항 ID·입력 순서 미사용).
"""
from __future__ import annotations

import argparse
import json
import sys
from importlib import resources
from pathlib import Path
from typing import Optional, Sequence

from .heuristic import write_submission_atomic
from .hash_regex import make_hash_regex_submission, parse_artifact
from .protocol import (
    TIERS,
    ProtocolError,
    load_bundled_policy,
    load_input,
    loads_json,
)

ARTIFACT_NAME = "artifact.v1.json"


def load_bundled_artifact():
    try:
        text = resources.read_text(
            "ossp_router.resources", ARTIFACT_NAME, encoding="utf-8"
        )
    except (OSError, UnicodeError) as exc:
        raise ProtocolError(f"내장 학습 파일을 읽을 수 없습니다: {exc}") from exc
    return parse_artifact(loads_json(text))


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="router-run")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--tier", choices=TIERS, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--policy", type=Path)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inputs = load_input(args.input)
        policy = load_bundled_policy()
        artifact = load_bundled_artifact()
        plan = make_hash_regex_submission(inputs, policy, artifact, args.tier)
        write_submission_atomic(args.output, plan.submission)
    except (OSError, ProtocolError, ValueError, json.JSONDecodeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    print(f"OK: {args.tier} (예측 비율 {plan.predicted_budget_ratio:.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
