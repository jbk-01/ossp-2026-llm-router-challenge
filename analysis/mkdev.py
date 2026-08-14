import json, pandas as pd
from decimal import Decimal
RATE={"ax31-light":(1,4),"ax31":(2.127,8.509),"axk1-think":(6.565,26.26)}
for split in ["train","dev"]:
    o=json.load(open(f"data/{split}/outcomes.json",encoding="utf-8"))
    i=json.load(open(f"data/materialized/{split}/inputs.json",encoding="utf-8"))
    txt={e["episode_id"]: e.get("prompt") or "\n".join(m["content"] for m in e["messages"])
         for e in i["episodes"]}
    rows=[]
    for e in o["episodes"]:
        r={"episode_id":e["episode_id"]}
        for m,(ri,ro) in RATE.items():
            d=e["models"][m]
            r[f"score_{m}"]=float(Decimal(d["score"]))
            r[f"cost_{m}"]=(d["input_tokens"]*ri+d["output_tokens"]*ro)/1e6
            r[f"out_{m}"]=d["output_tokens"]
            r[f"in_{m}"]=d["input_tokens"]
        rows.append(r)
    df=pd.DataFrame(rows)
    df["prompt"]=df["episode_id"].map(txt)
    df["chars"]=df["prompt"].str.len()
    df.to_csv(f"build/{split}.csv",index=False)
    print(split, len(df))
