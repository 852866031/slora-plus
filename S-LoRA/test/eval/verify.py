# verify_overlap.py
import sqlite3, pandas as pd, re

SQLITE = "my_trace.sqlite"  # nsys export my_trace.nsys-rep --type sqlite -o my_trace

conn = sqlite3.connect(SQLITE)

kern = pd.read_sql_query("""
  SELECT start, end, deviceId, contextId, streamId
  FROM CUPTI_ACTIVITY_KIND_KERNEL
  WHERE start IS NOT NULL AND end IS NOT NULL
""", conn)

nvtx = pd.read_sql_query("SELECT start, end, textId FROM NVTX_EVENTS WHERE textId IS NOT NULL", conn)
sid  = pd.read_sql_query("SELECT id, value FROM StringIds", conn)
conn.close()

nvtx = nvtx.merge(sid, left_on="textId", right_on="id", how="left").rename(columns={"value":"text"})
nvtx = nvtx.dropna(subset=["start","end","text"]).copy()

def nvtx_type(s):
    m = re.search(r":([a-z_]+)", str(s).lower())
    return m.group(1) if m else None

nvtx["type"] = nvtx["text"].apply(nvtx_type)
nv_b = nvtx[nvtx["type"]=="backward"][["start","end"]]
nv_d = nvtx[nvtx["type"]=="decode"][["start","end"]]

def overlaps_any(row, intervals):
    s,e = row["start"], row["end"]
    return ((intervals["start"] < e) & (intervals["end"] > s)).any()

kb = kern[kern.apply(overlaps_any, axis=1, args=(nv_b,))].copy()
kd = kern[kern.apply(overlaps_any, axis=1, args=(nv_d,))].copy()

def compress(iv):
    iv = sorted([(int(s),int(e)) for s,e in iv], key=lambda x: x[0])
    out = []
    for s,e in iv:
        if not out or s > out[-1][1]:
            out.append([s,e])
        else:
            out[-1][1] = max(out[-1][1], e)
    return out

B = compress(kb[["start","end"]].to_numpy())
D = compress(kd[["start","end"]].to_numpy())

# overlap between unions (ns)
i=j=0; overlap=0
while i<len(B) and j<len(D):
    s1,e1=B[i]; s2,e2=D[j]
    overlap += max(0, min(e1,e2)-max(s1,s2))
    if e1<e2: i+=1
    else: j+=1

to_ms=lambda ns: ns/1e6
tb = sum(e-s for s,e in B); td = sum(e-s for s,e in D)

print(f"GPU kernel overlap(backward ∩ decode): {to_ms(overlap):.3f} ms")
print(f"Backward total: {to_ms(tb):.3f} ms  ({100*overlap/tb:.1f}% overlapped)" if tb else "Backward total: 0")
print(f"Decode   total: {to_ms(td):.3f} ms  ({100*overlap/td:.1f}% overlapped)" if td else "Decode total: 0")

# Optional: which streams were used?
print("\nTop streams for backward:")
print(kb["streamId"].value_counts().head(5))
print("\nTop streams for decode:")
print(kd["streamId"].value_counts().head(5))