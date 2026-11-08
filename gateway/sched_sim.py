# SPDX-FileCopyrightText: Copyright 2026 뭘했음청년들
# SPDX-License-Identifier: Apache-2.0
"""요청이 시간차로 도착하는 상황에서 세 가지 스케줄링 정책 비교.

fifo  : 도착 순서대로 처리
group : 현재 적재된 모델의 요청을 우선 처리 (교체 최소화)
aging : group과 같되, 대기 시간이 상한 W를 넘은 요청을 최우선 처리
"""
import sys
from compare_sched import pick, unload_all
from run_gateway import messages_of, call

def choose(policy, queue, clock, prev, W):
    oldest = min(queue, key=lambda j: j[3])
    if policy == "fifo":
        return oldest
    if policy == "aging" and clock - oldest[3] > W:
        return oldest
    same = [j for j in queue if j[1] == prev]
    return min(same, key=lambda j: j[3]) if same else oldest

def run(policy, jobs, gap, W):
    unload_all()
    pending = [(i, m, e, i * gap) for i, m, e in jobs]
    clock, prev, lat, load, sw, toks = 0.0, None, {}, 0.0, 0, 0
    while pending:
        arrived = [j for j in pending if j[3] <= clock]
        if not arrived:
            clock = min(j[3] for j in pending)
            continue
        j = choose(policy, arrived, clock, prev, W)
        pending.remove(j)
        r = call(j[1], messages_of(j[2]))
        clock += r["wall_s"]
        lat[j[0]] = clock - j[3]
        load += r["load_s"]
        toks += r["output_tokens"]
        if prev is not None and j[1] != prev:
            sw += 1
        prev = j[1]
        print(f"  [{policy:5s}] #{j[0]:2d} {j[1]:11s} wait {clock - r['wall_s'] - j[3]:6.1f}s "
              f"wall {r['wall_s']:6.2f}s load {r['load_s']:5.2f}s out {r['output_tokens']:4d}tok")
    l = sorted(lat.values())
    return {"total": clock, "load": load, "switches": sw, "mean": sum(l) / len(l),
            "p90": l[int(0.9 * (len(l) - 1))], "max": l[-1], "toks": toks}

def main(inputs_path, decisions_path, n, gap, W):
    jobs = pick(inputs_path, decisions_path, n)
    res = {p: run(p, jobs, gap, W) for p in ("fifo", "group", "aging")}
    print(f"\n도착 간격 {gap}s, 대기 상한 {W}s")
    print(f"{'':10s}{'fifo':>10s}{'group':>10s}{'aging':>10s}")
    for k, lab in [("total", "총 시간"), ("load", "교체 시간"), ("mean", "평균 지연"),
                   ("p90", "90% 지연"), ("max", "최대 지연")]:
        print(f"{lab:8s}" + "".join(f"{res[p][k]:9.1f}s" for p in res))
    print(f"{'교체 횟수':7s}" + "".join(f"{res[p]['switches']:10d}" for p in res))
    print(f"{'출력 토큰':7s}" + "".join(f"{res[p]['toks']:10d}" for p in res))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
