# SPDX-FileCopyrightText: Copyright 2026 뭘했음청년들
# SPDX-License-Identifier: Apache-2.0
"""스케줄링 정책 반복 비교. 정책을 교대로 실행하고 결과를 즉시 파일에 남긴다.

실행:  python3 -u gateway/repeat_sched.py INPUTS DECISIONS N GAP W REPS
요약:  python3 gateway/repeat_sched.py summary
"""
import json, sys
from pathlib import Path
from compare_sched import pick
from sched_sim import run

OUT = Path("build/gw/repeat.jsonl")
POLICIES = ("fifo", "group", "aging")
KEYS = [("total", "총 시간"), ("load", "교체 시간"), ("mean", "평균 지연"),
        ("p90", "90% 지연"), ("max", "최대 지연"), ("switches", "교체 횟수")]

def summarize():
    rows = [json.loads(l) for l in OUT.open()]
    print("완료: " + ", ".join(f"{p} {sum(r['policy'] == p for r in rows)}회" for p in POLICIES))
    print(f"{'':9s}" + "".join(f"{p:>24s}" for p in POLICIES))
    for k, lab in KEYS:
        cells = []
        for p in POLICIES:
            v = [r[k] for r in rows if r["policy"] == p]
            cells.append(f"{sum(v)/len(v):7.1f} [{min(v):6.1f}~{max(v):6.1f}]" if v else f"{'-':>22s}")
        print(f"{lab:8s} " + "  ".join(cells))
    print("(평균 [최소~최대], 시간 단위 초)")

def main(inp, dec, n, gap, W, reps):
    jobs = pick(inp, dec, n)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for r in range(reps):
        for p in POLICIES:
            print(f"=== 반복 {r + 1}/{reps}  {p} ===", flush=True)
            res = run(p, jobs, gap, W)
            res.update(policy=p, rep=r, gap=gap, W=W)
            with OUT.open("a") as f:
                f.write(json.dumps(res) + "\n")
    summarize()

if __name__ == "__main__":
    a = sys.argv
    if a[1] == "summary":
        summarize()
    else:
        main(a[1], a[2], int(a[3]), float(a[4]), float(a[5]), int(a[6]))
