#!/bin/bash
# 폴드별 학습 (시간 걸림)
for k in 0 1 2 3 4; do
  echo "=== 폴드 $k 학습 ==="
  PYTHONPATH=src:baselines .venv-data/bin/python baselines/train_hash_regex.py \
    --input build/cv$k-tr-in.json --outcomes build/cv$k-tr-out.json \
    --validation-input build/cv$k-te-in.json --validation-outcomes build/cv$k-te-out.json \
    --artifact build/cv$k-art.json --report build/cv$k-rep.json
done
