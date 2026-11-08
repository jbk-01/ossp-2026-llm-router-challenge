# SPDX-FileCopyrightText: Copyright 2026 뭘했음청년들
# SPDX-License-Identifier: Apache-2.0
"""실측 기반 폐루프 등급 하향 제어.

예상 처리 시간 = a[모델] x 라우터의 예측 비용 + (모델 교체 시) 교체 비용
a[모델]은 처리할 때마다 실측값으로 지수이동평균 갱신한다.
과부하(큐 >= QHI) 모드에서도 하향 시 예상 시간이 MIN_GAIN초 이상
줄어들 때만 하향한다.
"""
import json, sys
from pathlib import Path
from compare_sched import pick, unload_all
from run_gateway import messages_of, call
from sched_sim import choose
from adapt_sim import DOWN, score_of
from ossp_router.hash_regex import parse_artifact, predict_episode
from ossp_router.protocol import load_input

ART = "src/ossp_router/resources/artifact.v1.json"
SWITCH_S, ALPHA, MIN_GAIN = 3.5, 0.3, 5.0

def predicted_costs(inputs_path):
    art = parse_artifact(json.load(open(ART)))
    return {ep.episode_id: predict_episode(ep, art)[1]
            for ep in load_input(Path(inputs_path)).episodes}

class Profile:
    def __init__(self):
        self.a = {}
    def est(self, mid, cost, loaded):
        if mid in self.a:
            a = self.a[mid]
        elif self.a:
            a = sum(self.a.values()) / len(self.a)
        else:
            return None
        return a * cost + (0.0 if mid == loaded else SWITCH_S)
    def update(self, mid, cost, busy_s):
        r = busy_s / max(cost, 1e-12)
        self.a[mid] = r if mid not in self.a else (1 - ALPHA) * self.a[mid] + ALPHA * r

def corr(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return sxy / (sx * sy) if sx and sy else float("nan")

def run(policy, jobs, gap, W, qhi, qlo, outcomes, C):
    unload_all()
    prof = Profile()
    pending = [(i, m, e, i * gap) for i, m, e in jobs]
    clock, prev, lat, load, sw = 0.0, None, {}, 0.0, 0
    degraded, ndown, qsum, ests, acts = False, 0, 0.0, [], []
    while pending:
        arrived = [j for j in pending if j[3] <= clock]
        if not arrived:
            clock = min(j[3] for j in pending)
            continue
        j = choose("aging", arrived, clock, prev, W)
        pending.remove(j)
        q, planned, eid = len(arrived), j[1], j[2]["episode_id"]
        if q >= qhi:
            degraded = True
        elif q <= qlo:
            degraded = False
        mid = planned
        if policy == "closed" and degraded and DOWN[planned] != planned:
            tp = prof.est(planned, C[eid][planned], prev)
            td = prof.est(DOWN[planned], C[eid][DOWN[planned]], prev)
            if tp is not None and td is not None and td < tp - MIN_GAIN:
                mid = DOWN[planned]
        est = prof.est(mid, C[eid][mid], prev)
        r = call(mid, messages_of(j[2]))
        prof.update(mid, C[eid][mid], r["wall_s"] - r["load_s"])
        if est is not None:
            ests.append(est)
            acts.append(r["wall_s"])
        ndown += mid != planned
        clock += r["wall_s"]
        lat[j[0]] = clock - j[3]
        load += r["load_s"]
        sw += prev is not None and mid != prev
        prev = mid
        qsum += score_of(outcomes, eid, mid)
        tag = f"{planned}->{mid}" if mid != planned else planned
        es = f"{est:6.1f}s" if est is not None else "   -  "
        print(f"  [{policy:6s}] #{j[0]:2d} queue {q} {tag:24s} 예상 {es} 실제 {r['wall_s']:6.2f}s")
    l = sorted(lat.values())
    n = len(jobs)
    return {"total": clock, "load": load, "switches": sw, "mean": sum(l) / n,
            "p90": l[int(0.9 * (n - 1))], "max": l[-1], "down": ndown,
            "quality": qsum / n, "corr": corr(ests, acts)}

def main(inp, dec, outp, n, gap, W, qhi, qlo):
    outcomes = {e["episode_id"]: e for e in json.load(open(outp))["episodes"]}
    C = predicted_costs(inp)
    jobs = pick(inp, dec, n)
    res = {p: run(p, jobs, gap, W, qhi, qlo, outcomes, C) for p in ("aging", "closed")}
    print(f"\n도착 간격 {gap}s, 대기 상한 {W}s, 하향 진입 큐>={qhi}, 해제 큐<={qlo}")
    print(f"{'':10s}{'aging':>10s}{'closed':>10s}")
    for k, lab in [("total", "총 시간"), ("load", "교체 시간"), ("mean", "평균 지연"),
                   ("p90", "90% 지연"), ("max", "최대 지연")]:
        print(f"{lab:8s}" + "".join(f"{res[p][k]:9.1f}s" for p in res))
    for k, lab in [("switches", "교체 횟수"), ("down", "하향 건수")]:
        print(f"{lab:7s}" + "".join(f"{res[p][k]:10d}" for p in res))
    print(f"{'평균 품질':7s}" + "".join(f"{res[p]['quality']:10.3f}" for p in res))
    print(f"{'예측 상관':7s}" + "".join(f"{res[p]['corr']:10.3f}" for p in res))

if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3], int(a[4]), float(a[5]), float(a[6]), int(a[7]), int(a[8]))
