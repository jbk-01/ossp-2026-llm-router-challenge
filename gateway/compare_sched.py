# SPDX-FileCopyrightText: Copyright 2026 뭘했음청년들
# SPDX-License-Identifier: Apache-2.0
"""FIFO와 모델별 묶음(Grouped) 스케줄링의 처리 시간·대기 시간 비교.

같은 요청 집합을 두 순서로 실행해, 모델 교체 비용과 요청별 완료 시간을
측정한다. 각 실행 전 모든 모델을 메모리에서 내려 조건을 맞춘다.
"""
import json, sys, urllib.request
from run_gateway import MODEL_MAP, MAX_PROMPT_CHARS, messages_of, call

def unload_all():
    for m in MODEL_MAP.values():
        body = json.dumps({"model": m, "keep_alive": 0}).encode()
        req = urllib.request.Request("http://127.0.0.1:11434/api/generate", body,
                                     {"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=60).read()

def pick(inputs_path, decisions_path, n):
    eps = {e["episode_id"]: e for e in json.load(open(inputs_path))["episodes"]}
    out = []
    for d in json.load(open(decisions_path))["decisions"]:
        ep = eps[d["episode_id"]]
        if len("".join(m["content"] for m in messages_of(ep))) <= MAX_PROMPT_CHARS:
            out.append((len(out), d["model_id"], ep))
        if len(out) >= n:
            break
    return out

def run(name, jobs):
    unload_all()
    clock, done, load_sum, switches, prev = 0.0, {}, 0.0, 0, None
    for idx, mid, ep in jobs:
        r = call(mid, messages_of(ep))
        clock += r["wall_s"]
        done[idx] = clock
        load_sum += r["load_s"]
        if prev is not None and mid != prev:
            switches += 1
        prev = mid
        print(f"  [{name}] #{idx:2d} {mid:11s} wall {r['wall_s']:6.2f}s load {r['load_s']:5.2f}s")
    t = sorted(done.values())
    return {"total": clock, "load": load_sum, "switches": switches,
            "mean_done": sum(t) / len(t), "max_done": t[-1], "done": done}

def main(inputs_path, decisions_path, n):
    jobs = pick(inputs_path, decisions_path, n)
    order = {"axk1-think": 0, "ax31": 1, "ax31-light": 2}
    grouped = sorted(jobs, key=lambda j: (order[j[1]], j[0]))
    a = run("FIFO", jobs)
    b = run("GROUP", grouped)
    print("\n            FIFO      GROUP")
    for k, lab in [("total", "총 처리시간"), ("load", "교체 시간"), ("mean_done", "평균 완료"),
                   ("max_done", "최대 완료")]:
        print(f"{lab:8s} {a[k]:8.1f}s {b[k]:8.1f}s")
    print(f"교체 횟수   {a['switches']:5d}     {b['switches']:5d}")
    worst = max(range(len(jobs)), key=lambda i: b["done"][i] - a["done"][i])
    print(f"가장 손해 본 요청 #{worst}: FIFO {a['done'][worst]:.1f}s → GROUP {b['done'][worst]:.1f}s")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
