# gpu_plot.py
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import re
import math
import os
import sys
import argparse
import bisect

# ---------------- CONFIG ----------------
BUCKET_MS     = 10                 # utilization bin size (ms)
NUM_SMS       = 128                # your GPU SM count
LANES         = 6
LANE_STEP     = 4                  # y-distance between lanes (percent)
LANE_MIN_SEP  = 0.3                # min time (s) separation to reuse a lane
START_STYLE   = "--"               # annotation start style
END_STYLE     = ":"                # annotation end style
MEM_YTICK_STEP_GB = 2              # right-axis tick step in GB
RIGHT_Y_MAX_GB    = 26             # clamp right axis (GPU=24GB). Change if needed.
MODEL_LOAD_LABEL  = "model_load"   # NVTX instant label to anchor mem-manager
# ----------------------------------------

# --------- mem-manager log regex (ns or wall clock; with Page Size) ----------
NS_RE   = re.compile(r"^\[\s*(?P<ns>\d+)\s*\].*?\|GPU\s+(?P<gpu_used>\d+)\s*/\s*(?P<gpu_max>\d+)\s+CPU\s+(?P<cpu_used>\d+)\|")
WALL_RE = re.compile(r"^\[\s*(?P<ts>[\d\-:\s]+)\s*\].*?\|GPU\s+(?P<gpu_used>\d+)\s*/\s*(?P<gpu_max>\d+)\s+CPU\s+(?P<cpu_used>\d+)\|")
PAGE_SZ_RE = re.compile(r"Page\s*Size\s*(?P<val>[\d.]+)\s*(?P<unit>[KMG]B)", re.IGNORECASE)
UNIT_MULT = {"KB": 1024, "MB": 1024**2, "GB": 1024**3}

def parse_memmgr_log(log_path: str):
    """
    Parse mem-manager log with ns timestamps and '/Page Size ...'.
    Returns DataFrame with columns:
      t_rel_s (memmgr local seconds), used_gb, max_gb
    """
    from datetime import datetime

    rows = []
    last_page_size_bytes = None
    saw_ns = False
    saw_wall = False

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            ps = PAGE_SZ_RE.search(line)
            if ps:
                try:
                    last_page_size_bytes = int(float(ps.group("val")) * UNIT_MULT[ps.group("unit").upper()])
                except Exception:
                    pass

            m = NS_RE.search(line)
            if m:
                saw_ns = True
                rows.append({
                    "ns_rel_raw": int(m.group("ns")),
                    "gpu_used_pages": int(m.group("gpu_used")),
                    "gpu_max_pages": int(m.group("gpu_max")),
                    "cpu_pages": int(m.group("cpu_used")),
                    "page_size_bytes": last_page_size_bytes,
                    "ts": None
                })
                continue

            m = WALL_RE.search(line)
            if m:
                saw_wall = True
                try:
                    ts = datetime.strptime(m.group("ts").strip(), "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue
                rows.append({
                    "ns_rel_raw": None,
                    "gpu_used_pages": int(m.group("gpu_used")),
                    "gpu_max_pages": int(m.group("gpu_max")),
                    "cpu_pages": int(m.group("cpu_used")),
                    "page_size_bytes": last_page_size_bytes,
                    "ts": ts
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Ensure page size known (fallback 128KB if never present)
    if df["page_size_bytes"].isna().any():
        df["page_size_bytes"] = df["page_size_bytes"].fillna(128 * 1024)

    if saw_ns:
        df = df[df["ns_rel_raw"].notna()].copy().sort_values("ns_rel_raw").reset_index(drop=True)
        df["t_rel_s"] = df["ns_rel_raw"] * 1e-9  # local seconds from memmgr start
    else:
        # fallback: wall clock -> seconds from first line
        df = df[df["ts"].notna()].copy().sort_values("ts").reset_index(drop=True)
        t0 = df.loc[0, "ts"]
        df["t_rel_s"] = (df["ts"] - t0).dt.total_seconds()

    # bytes & GB
    df["used_bytes"] = df["gpu_used_pages"] * df["page_size_bytes"]
    df["max_bytes"]  = df["gpu_max_pages"]  * df["page_size_bytes"]
    df["used_gb"]    = df["used_bytes"] / (1024**3)
    df["max_gb"]     = df["max_bytes"]  / (1024**3)
    return df

def modal_max_gb(df: pd.DataFrame) -> float:
    counts = Counter([float(x) for x in df["max_gb"].tolist()])
    if not counts:
        return float(df["max_gb"].max()) if "max_gb" in df else 0.0
    best = max(counts.values())
    candidates = [v for v, c in counts.items() if c == best]
    return float(max(candidates))

def step_lookup(xs: list[float], ys: list[float], t: float) -> float:
    """Return last y value where x <= t in a step series."""
    if not xs:
        return 0.0
    i = bisect.bisect_right(xs, t) - 1
    if i < 0:
        return ys[0]
    return ys[i]

def main():
    ap = argparse.ArgumentParser(description="GPU Utilization + Total Memory + Mem-Manager underlay, anchored at NVTX 'model_load'")
    ap.add_argument("--report_name", help="nsys report base name (no extension)")
    ap.add_argument("--memmgr_log", default= "mem_manager_log.txt", help="mem-manager log path")
    args = ap.parse_args()
    os.system(f"nsys export {args.report_name}.nsys-rep --force-overwrite true --output {args.report_name}.sqlite --type sqlite")

    report_name = args.report_name
    SQLITE_PATH = f"{report_name}.sqlite"

    # Export .nsys-rep -> .sqlite (idempotent)
    if not os.path.exists(SQLITE_PATH):
        os.system(f"nsys export {report_name}.nsys-rep --force-overwrite true --output {report_name}.sqlite --type sqlite")

    # ---------- LOAD FROM SQLITE ----------
    conn = sqlite3.connect(SQLITE_PATH)
    # Kernels
    try:
        kernels_df = pd.read_sql_query("""
            SELECT start, end
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE start IS NOT NULL AND end IS NOT NULL
        """, conn)
        kernels_df = kernels_df.dropna(subset=["start","end"]).astype({"start":"int64","end":"int64"})
    except Exception:
        print("[WARN] CUPTI_ACTIVITY_KIND_KERNEL missing.")
        kernels_df = pd.DataFrame(columns=["start","end"])

    # NVTX events + strings
    nvtx_events = pd.read_sql_query("SELECT start, end, textId, eventType FROM NVTX_EVENTS", conn)
    string_ids  = pd.read_sql_query("SELECT id, value FROM StringIds", conn)

    # Total memory events
    try:
        mem_df = pd.read_sql_query("SELECT start, bytes, memoryOperationType FROM CUDA_GPU_MEMORY_USAGE_EVENTS", conn)
    except Exception:
        print("[WARN] CUDA_GPU_MEMORY_USAGE_EVENTS missing; total memory curve omitted.")
        mem_df = pd.DataFrame(columns=["start","bytes","memoryOperationType"])

    # Optional enum mapping for memory ops
    op_map = {}
    try:
        enum_df = pd.read_sql_query("SELECT * FROM ENUM_CUDA_DEV_MEM_EVENT_OPER", conn)
        if "id" in enum_df.columns:
            name_col = "name" if "name" in enum_df.columns else ("value" if "value" in enum_df.columns else None)
            if name_col:
                op_map = dict(zip(enum_df["id"], enum_df[name_col].astype(str)))
    except Exception:
        pass

    conn.close()

    # ---------- NVTX merge & find 'model_load' ----------
    nvtx_df = nvtx_events.merge(string_ids, how="left", left_on="textId", right_on="id") \
                         .rename(columns={"value":"text"})
    # Find the first 'model_load' instant (case-insensitive contains)
    nvtx_df["text_lc"] = nvtx_df["text"].astype(str).str.lower()
    ml_rows = nvtx_df[nvtx_df["text_lc"].str.contains(MODEL_LOAD_LABEL, na=False)]
    if ml_rows.empty:
        raise SystemExit(f"Could not find NVTX event with label containing '{MODEL_LOAD_LABEL}'.")
    model_load_ns = int(ml_rows.iloc[0]["start"])  # ns since nsys start

    # ---------- Relative time base ----------
    kernel_min = kernels_df["start"].min() if not kernels_df.empty else math.inf
    nvtx_min   = nvtx_df["start"].min()    if not nvtx_df.empty   else math.inf
    mem_min    = mem_df["start"].min()     if not mem_df.empty    else math.inf
    base_time  = min(kernel_min, nvtx_min, mem_min, model_load_ns)
    if base_time is math.inf:
        base_time = 0

    # Rel seconds
    if not kernels_df.empty:
        kernels_df["start_rel"] = (kernels_df["start"] - base_time) / 1e9
        kernels_df["end_rel"]   = (kernels_df["end"]   - base_time) / 1e9
        kernels_df = kernels_df.dropna(subset=["start_rel","end_rel"])

    # NVTX job annotations (optional)
    nvtx_jobs = nvtx_df.dropna(subset=["start","end","text"]).copy()
    nvtx_jobs["start_rel"] = (nvtx_jobs["start"] - base_time) / 1e9
    nvtx_jobs["end_rel"]   = (nvtx_jobs["end"]   - base_time) / 1e9

    def parse_job(text: str):
        m = re.match(r"job[_\-]?(\d+):([A-Za-z_]+)", str(text).lower())
        if not m:
            return "unknown","other"
        jid, jtype = m.groups()
        return f"job_{jid}", jtype

    parsed = nvtx_jobs["text"].apply(parse_job)
    nvtx_jobs["job_id"]   = parsed.apply(lambda x: x[0])
    nvtx_jobs["job_type"] = parsed.apply(lambda x: x[1])
    nvtx_jobs = nvtx_jobs[nvtx_jobs["job_type"].isin({"prefill","decode","backward"})]

    # ---------- Utilization (left) ----------
    BUCKET_S   = BUCKET_MS / 1000.0
    MAX_US_BIN = BUCKET_S * 1e6 * NUM_SMS
    buckets = defaultdict(float)

    if not kernels_df.empty:
        for _, r in kernels_df.iterrows():
            t0, t1 = float(r["start_rel"]), float(r["end_rel"])
            if not (math.isfinite(t0) and math.isfinite(t1)) or t1 <= t0:
                continue
            b0 = int(t0 // BUCKET_S)
            b1 = int(t1 // BUCKET_S)
            for b in range(b0, b1 + 1):
                bs = b * BUCKET_S
                be = bs + BUCKET_S
                overlap = max(0.0, min(t1, be) - max(t0, bs))
                if overlap > 0:
                    buckets[bs] += overlap * 1e6 * NUM_SMS

    timestamps   = sorted(buckets.keys())
    utilizations = [min(100.0, 100.0 * buckets[t] / MAX_US_BIN) for t in timestamps]

    # ---------- Total memory (right) ----------
    mem_t = []
    mem_total_gb = []

    if not mem_df.empty:
        mem_df = mem_df.dropna(subset=["start","bytes"]).copy()
        mem_df["start"] = pd.to_numeric(mem_df["start"], errors="coerce")
        mem_df["bytes"] = pd.to_numeric(mem_df["bytes"], errors="coerce")
        mem_df = mem_df.dropna(subset=["start","bytes"])
        mem_df["t_rel"] = (mem_df["start"] - base_time) / 1e9

        def signed_size(row):
            val = float(row["bytes"])
            op = row.get("memoryOperationType", None)
            name = None
            if op_map and pd.notna(op) and op in op_map:
                name = str(op_map[op]).lower()
            elif pd.notna(op):
                name = str(op).lower()
            if name:
                if any(k in name for k in ["free","dealloc","release"]): return -abs(val)
                if any(k in name for k in ["alloc","malloc","new"]):    return  abs(val)
            return val

        mem_df["signed_size"] = mem_df.apply(signed_size, axis=1)
        mem_df = mem_df.sort_values("t_rel")
        mem_df["total_bytes"] = mem_df["signed_size"].cumsum()
        mem_df["total_gb"] = mem_df["total_bytes"] / (1024**3)

        mem_t = mem_df["t_rel"].tolist()
        mem_total_gb = mem_df["total_gb"].tolist()

        # extend to plot end for a step look
        end_candidates = [mem_t[-1] if mem_t else 0.0]
        if timestamps: end_candidates.append(timestamps[-1])
        if not nvtx_jobs.empty: end_candidates.append(float(nvtx_jobs["end_rel"].max()))
        if not kernels_df.empty: end_candidates.append(float(kernels_df["end_rel"].max()))
        t_end = max(end_candidates) if end_candidates else 0.0

        if mem_t:
            if mem_t[0] > 0:
                mem_t = [0.0] + mem_t
                mem_total_gb = [mem_total_gb[0]] + mem_total_gb
            if t_end > mem_t[-1]:
                mem_t.append(t_end)
                mem_total_gb.append(mem_total_gb[-1])

    # ---------- Mem-manager (align to NVTX 'model_load') ----------
    mm_df = parse_memmgr_log(args.memmgr_log)
    mm_df["cpu_gb"] = (mm_df["cpu_pages"] * mm_df["page_size_bytes"]) / (1024**3)
    mm_t_local = mm_df["t_rel_s"].tolist() if not mm_df.empty else []
    mm_used_gb = mm_df["used_gb"].tolist() if not mm_df.empty else []
    mm_max_gb_value = float(modal_max_gb(mm_df)) if not mm_df.empty else 0.0

    # model_load relative seconds in nsys time base
    model_load_rel_s = (model_load_ns - base_time) / 1e9

    # Align: t_mm_aligned = model_load_rel_s + t_mm_local
    mm_t = [model_load_rel_s + t for t in mm_t_local] if mm_t_local else []

    # Underlay math
    mm_y_base = []
    mm_y_curve = []
    if mm_t and mem_t:
        for t, u in zip(mm_t, mm_used_gb):
            total_gb_t = step_lookup(mem_t, mem_total_gb, t)
            base = total_gb_t - mm_max_gb_value
            mm_y_base.append(base)
            mm_y_curve.append(base + u)

    # ---------- Annotations ----------
    annots = []
    for _, r in nvtx_jobs.iterrows():
        annots.append({"time": float(r["start_rel"]), "job_id": r["job_id"], "job_type": r["job_type"], "kind": "start"})
        annots.append({"time": float(r["end_rel"]),   "job_id": r["job_id"], "job_type": r["job_type"], "kind": "end"})
    annots.sort(key=lambda a: a["time"])

    last_time_in_lane = [-1e12] * LANES
    def assign_lane(t: float) -> int:
        for i in range(LANES):
            if t - last_time_in_lane[i] >= LANE_MIN_SEP:
                last_time_in_lane[i] = t
                return i
        i = min(range(LANES), key=lambda k: last_time_in_lane[k])
        last_time_in_lane[i] = t
        return i

    job_colors = {"prefill":"tab:blue","decode":"tab:orange","backward":"tab:green"}

    # ---------- PLOT ----------
    fig, ax = plt.subplots(figsize=(20, 6))

    # utilization (left)
    if timestamps:
        ax.plot(timestamps, utilizations, color="tab:blue", label="GPU Utilization (%)", linewidth=1.5)

    # total memory (right)
    ax2 = ax.twinx()
    if mem_t:
        ax2.plot(mem_t, mem_total_gb, color="tab:red", linestyle="--", linewidth=1.2, label="GPU Memory (GB)")

    # mem-manager underlay (right)
    if mm_t and mem_t:
        t_mem_end = mem_t[-1]  # last timestamp of the total memory step series

        # If mem-manager stops earlier, append a flat point at t_mem_end
        if mm_t[-1] < t_mem_end:
            base_end = step_lookup(mem_t, mem_total_gb, t_mem_end) - mm_max_gb_value
            last_used = mm_used_gb[-1]  # hold the last mem-manager usage level
            mm_t.append(t_mem_end)
            mm_y_base.append(base_end)
            mm_y_curve.append(base_end + last_used)
    if mm_t and mm_y_curve:
        mm_y_segments = []
        last_color = None
        seg_t, seg_yb, seg_yc = [], [], []

        for t, u_gpu, u_cpu in zip(mm_t, mm_used_gb, mm_df["cpu_gb"].tolist()):
            total_gb_t = step_lookup(mem_t, mem_total_gb, t)   # aligns with nsys memory curve
            base = total_gb_t - mm_max_gb_value
            curve = base + u_gpu
            color = "tab:red" if u_cpu > 0 else "tab:purple"

            if last_color is None:
                seg_t, seg_yb, seg_yc = [t], [base], [curve]
                last_color = color
            elif color == last_color:
                seg_t.append(t); seg_yb.append(base); seg_yc.append(curve)
            else:
                mm_y_segments.append((seg_t, seg_yb, seg_yc, last_color))
                seg_t, seg_yb, seg_yc = [t], [base], [curve]
                last_color = color

        if last_color is not None:
            mm_y_segments.append((seg_t, seg_yb, seg_yc, last_color))

        # --- Plot each segment with its color ---
        for seg_t, seg_yb, seg_yc, color in mm_y_segments:
            label = "Mem-Manager (GPU)" if color == "tab:purple" else "Mem-Manager (swapped)"
            ax2.fill_between(seg_t, seg_yb, seg_yc, alpha=0.25, color=color, label=label)
            ax2.plot(seg_t, seg_yc, color=color, linewidth=1.0)

    # mark model_load
    ax.axvline(model_load_rel_s, color="tab:green", linestyle="-.", linewidth=1.0, alpha=0.8)
    ax.text(model_load_rel_s, 118, "model_load", ha="center", va="top", fontsize=8, color="tab:green",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    # annotations
    y_base_annot = 101
    for a in annots:
        if a["job_type"] == "decode":
            continue
        c   = job_colors.get(a["job_type"], "tab:gray")
        ls  = START_STYLE if a["kind"] == "start" else END_STYLE
        t   = a["time"]
        lane = assign_lane(t)
        y_label = y_base_annot + lane * LANE_STEP

        ax.axvline(t, color=c, linestyle=ls, alpha=0.9, linewidth=1)
        jid_num = a["job_id"].replace("job_", "")
        label = f"j{jid_num}S" if a["kind"] == "start" else f"j{jid_num}E"
        ax.text(t, y_label, label, ha="center", va="bottom",
                color=c, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    # legends
    # Deduplicate legend entries
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc="lower right")
    handles_types = [plt.Line2D([0],[0], color=col, lw=2, linestyle="--", label=typ) for typ, col in job_colors.items()]
    leg1 = ax.legend(handles=handles_types, title="Job Type (annotation color)", loc="upper left")

    handles_lines = []
    if timestamps:
        handles_lines.append(plt.Line2D([0],[0], color="tab:blue", lw=1.5, linestyle="-", label="GPU Utilization (%)"))
    if mem_t:
        handles_lines.append(plt.Line2D([0],[0], color="tab:red", lw=1.2, linestyle="--", label="GPU Memory (GB)"))
    if mm_t and mm_y_curve:
        handles_lines.append(plt.Line2D([0],[0], color="tab:purple", lw=1.0, linestyle="-", label="Mem-Manager (underlay)"))
    if handles_lines:
        ax.add_artist(leg1)
        ax.legend(handles=handles_lines, loc="upper right")
    
    # axes
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("GPU Compute Utilization (%)")
    ax.set_ylim(0, 120)
    ax.set_yticks(range(0, 101, 20))

    ax2.set_ylabel("GPU Memory (GB)")
    if mem_t:
        ax2.set_ylim(0, 32)              # right axis fixed at 0–24 GB
        ax2.set_yticks(range(0, 32, 4))  # e.g. ticks every 4 GB

    ax.set_title("GPU Utilization, Total GPU Memory & Mem-Manager (anchored at NVTX 'model_load')")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("gpu_utilization_memory_memmgr_plot.png", dpi=150)
    print("[OK] wrote gpu_utilization_memory_memmgr_plot.png")
    plt.show()

if __name__ == "__main__":
    main()