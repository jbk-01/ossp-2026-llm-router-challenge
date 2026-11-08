# SPDX-FileCopyrightText: Copyright 2026 뭘했음청년들
# SPDX-License-Identifier: Apache-2.0
"""과부하 시 자동 등급 하향(관리 평면) 효과 측정.

aging : 대기 상한 W 기반 우선순위만 적용
adapt : aging + 큐 길이 기반 등급 하향. 큐 길이 >= QHI면 하향 모드 진입,
        <= QLO면 해제(히스테리시스). 품질은 대회 데이터의 모델별 점수로 평가.
"""
import json, sys
from compare_sched import pick, unload_all
from run_gateway import messages_of, call
from sched_sim import choose

DOWN = {"axk1-think": "ax31", "ax31": "ax31-light", "ax31-light": "ax31-light"}

def score_of(outcomes, eid, mid):
    m = outcomes[eid]["models"][mid]
    for k in ("score", "quality", "correct"):
        if k in m:
            return float(m[k])
    raise KeyError(f"점수 필드를 찾지 못함: {list(m)}")

def run(policy, jobs, gap, W, qhi, qlo, outcomes):
    unload_all()
    pending = [(i, m, e, i * gap) for i, m, e in jobs]
    clock, prev, lat, load, sw = 0.0, None, {}, 0.0, 0
    degraded, ndown, qsum, base = False, 0, 0.0, 0.0
    while pending:
        arrived = [j for j in pending if j[3] <= clock]
        if not arrived:
            clock = min(j[3] for j in pending)
            continue
        j = choose("aging", arrived, clock, prev, W)
        pending.remove(j)
        q = len(arrived)
        if policy == "adapt":
            if q >= qhi:
                degraded = True
            elif q <= qlo:
                degraded = False
        mid = DOWN[j[1]] if degraded else j[1]
        ndown += mid != j[1]
        r = call(mid, messages_of(j[2]))
        clock += r["wall_s"]
        lat[j[0]] = clock - j[3]
        load += r["load_s"]
        if prev is not None and mid != prev:
            sw += 1
        prev = mid
        eid = j[2]["episode_id"]
        qsum += score_of(outcomes, eid, mid)
        base += score_of(outcomes, eid, j[1])
        tag = f"{j[1]}->{mid}" if mid != j[1] else j[1]
        print(f"  [{policy:5s}] #{j[0]:2d} queue {q} {tag:24s} "
              f"wait {clock - r['wall_s'] - j[3]:6.1f}s wall {r['wall_s']:6.2f}s")
    l = sorted(lat.values())
    n = len(jobs)
    return {"total": clock, "load": load, "switches": sw, "mean": sum(l) / n,
            "p90": l[int(0.9 * (n - 1))], "max": l[-1],
            "down": ndown, "quality": qsum / n, "planned": base / n}

def main(inp, dec, outp, n, gap, W, qhi, qlo):
    outcomes = {e["episode_id"]: e for e in json.load(open(outp))["episodes"]}
    jobs = pick(inp, dec, n)
    res = {p: run(p, jobs, gap, W, qhi, qlo, outcomes) for p in ("aging", "adapt")}
    print(f"\n도착 간격 {gap}s, 대기 상한 {W}s, 하향 진입 큐>={qhi}, 해제 큐<={qlo}")
    print(f"{'':10s}{'aging':>10s}{'adapt':>10s}")
    for k, lab in [("total", "총 시간"), ("load", "교체 시간"), ("mean", "평균 지연"),
                   ("p90", "90% 지연"), ("max", "최대 지연")]:
        print(f"{lab:8s}" + "".join(f"{res[p][k]:9.1f}s" for p in res))
    for k, lab in [("switches", "교체 횟수"), ("down", "하향 건수")]:
        print(f"{lab:7s}" + "".join(f"{res[p][k]:10d}" for p in res))
    print(f"{'평균 품질':7s}" + "".join(f"{res[p]['quality']:10.3f}" for p in res))
    print(f"(하향 없이 라우터 원안대로 처리했을 때 품질 {res['aging']['planned']:.3f})")

if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3], int(a[4]), float(a[5]), float(a[6]), int(a[7]), int(a[8]))
