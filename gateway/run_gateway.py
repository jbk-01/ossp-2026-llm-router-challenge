
# SPDX-FileCopyrightText: Copyright 2026 뭘했음청년들
# SPDX-License-Identifier: Apache-2.0
"""라우터 결정을 실제 로컬 모델 호출로 실행하는 게이트웨이 (v0).

라우터(제어 평면)가 고른 model_id를 Ollama 모델(데이터 평면)에 대응시켜
실제로 호출하고, 요청별 지연시간·토큰 수·모델 적재 시간을 기록한다.
"""
import json, sys, time, urllib.request
from pathlib import Path

OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL_MAP = {
    "ax31-light": "qwen2.5:0.5b",
    "ax31": "qwen2.5:1.5b",
    "axk1-think": "deepseek-r1:1.5b",
}
MAX_TOKENS = {"ax31-light": 256, "ax31": 256, "axk1-think": 512}
MAX_PROMPT_CHARS = 1500

def messages_of(ep):
    if ep.get("messages"):
        return [{"role": m["role"], "content": m["content"]} for m in ep["messages"]]
    return [{"role": "user", "content": ep["prompt"]}]

def call(model_id, msgs):
    body = json.dumps({
        "model": MODEL_MAP[model_id], "messages": msgs, "stream": False,
        "options": {"num_predict": MAX_TOKENS[model_id]},
    }).encode()
    req = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as r:
        res = json.load(r)
    wall = time.monotonic() - t0
    ns = 1e9
    return {
        "wall_s": round(wall, 3),
        "load_s": round(res.get("load_duration", 0) / ns, 3),
        "prompt_tokens": res.get("prompt_eval_count", 0),
        "output_tokens": res.get("eval_count", 0),
        "gen_s": round(res.get("eval_duration", 0) / ns, 3),
    }

def main(inputs_path, decisions_path, n, out_path):
    eps = {e["episode_id"]: e for e in json.load(open(inputs_path))["episodes"]}
    decs = json.load(open(decisions_path))["decisions"]
    picked = []
    for d in decs:
        ep = eps[d["episode_id"]]
        text = "".join(m["content"] for m in messages_of(ep))
        if len(text) <= MAX_PROMPT_CHARS:
            picked.append((d["model_id"], ep))
        if len(picked) >= n:
            break
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i, (mid, ep) in enumerate(picked, 1):
            rec = {"idx": i, "model_id": mid, **call(mid, messages_of(ep))}
            f.write(json.dumps(rec) + "\n")
            print(f"[{i:2d}/{len(picked)}] {mid:11s} wall {rec['wall_s']:6.2f}s "
                  f"load {rec['load_s']:5.2f}s out {rec['output_tokens']:4d}tok")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])

