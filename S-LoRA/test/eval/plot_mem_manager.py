# mem_manager_mem_plot.py
import argparse
import re
from datetime import datetime
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

# New-style ns timestamp + counts (+ optional "Page Size ...")
NS_RE = re.compile(
    r"^\[\s*(?P<ns>\d+)\s*\].*?\|GPU\s+(?P<gpu_used>\d+)\s*/\s*(?P<gpu_max>\d+)\s+CPU\s+(?P<cpu_used>\d+)\|"
)

# Old-style wall clock timestamp + counts
WALL_RE = re.compile(
    r"^\[\s*(?P<ts>[\d\-:\s]+)\s*\].*?\|GPU\s+(?P<gpu_used>\d+)\s*/\s*(?P<gpu_max>\d+)\s+CPU\s+(?P<cpu_used>\d+)\|"
)

# Page size snippet e.g. "/Page Size 128.00 KB /" or "Page Size 2 MB"
PAGE_SZ_RE = re.compile(r"Page\s*Size\s*(?P<val>[\d.]+)\s*(?P<unit>[KMG]B)", re.IGNORECASE)

UNIT_MULT = {"KB": 1024, "MB": 1024**2, "GB": 1024**3}

def parse_size_to_bytes(s: str) -> int:
    """
    Accepts "128KB", "128 KB", "2 MB", "0.5 GB" (case-insensitive) -> bytes.
    """
    m = re.match(r"^\s*([\d.]+)\s*([KMG]B)\s*$", s.strip(), re.IGNORECASE)
    if not m:
        raise ValueError(f"Unrecognized size: {s}")
    val = float(m.group(1))
    unit = m.group(2).upper()
    return int(val * UNIT_MULT[unit])

def parse_log(path: str, fallback_page_size_bytes):
    rows = []
    saw_ns = False
    saw_wall = False
    last_page_size_bytes = fallback_page_size_bytes  # update when we see a Page Size

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # capture page size if present
            ps = PAGE_SZ_RE.search(line)
            if ps:
                try:
                    last_page_size_bytes = int(float(ps.group("val")) * UNIT_MULT[ps.group("unit").upper()])
                except Exception:
                    pass  # keep previous/fallback

            m_ns = NS_RE.search(line)
            if m_ns:
                saw_ns = True
                rows.append({
                    "ns_rel": int(m_ns.group("ns")),
                    "gpu_used_pages": int(m_ns.group("gpu_used")),
                    "gpu_max_pages": int(m_ns.group("gpu_max")),
                    "cpu_pages": int(m_ns.group("cpu_used")),
                    "page_size_bytes": last_page_size_bytes,
                    "ts": None,
                })
                continue

            m_wall = WALL_RE.search(line)
            if m_wall:
                saw_wall = True
                try:
                    ts = datetime.strptime(m_wall.group("ts").strip(), "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue
                rows.append({
                    "ns_rel": None,
                    "gpu_used_pages": int(m_wall.group("gpu_used")),
                    "gpu_max_pages": int(m_wall.group("gpu_max")),
                    "cpu_pages": int(m_wall.group("cpu_used")),
                    "page_size_bytes": last_page_size_bytes,
                    "ts": ts,
                })

    if not rows:
        raise SystemExit("No parsable lines found. Check the log format or regex.")

    df = pd.DataFrame(rows)

    # Ensure we have a page size; if still missing, fail clearly.
    if df["page_size_bytes"].isna().any():
        missing = df["page_size_bytes"].isna().sum()
        raise SystemExit(
            f"{missing} lines missing page size and no fallback provided. "
            "Pass --fallback-page-size (e.g., 128KB)."
        )

    # Time base
    if saw_ns:
        df = df[df["ns_rel"].notna()].copy().sort_values("ns_rel").reset_index(drop=True)
        df["t_rel_s"] = df["ns_rel"] * 1e-9
        time_basis = "ns"
    else:
        df = df[df["ts"].notna()].copy().sort_values("ts").reset_index(drop=True)
        t0 = df.loc[0, "ts"]
        df["t_rel_s"] = (df["ts"] - t0).dt.total_seconds()
        df["ns_rel"] = (df["t_rel_s"] * 1e9).astype("int64")
        time_basis = "wall"

    # Bytes
    df["gpu_used_bytes"] = df["gpu_used_pages"] * df["page_size_bytes"]
    df["gpu_max_bytes"]  = df["gpu_max_pages"]  * df["page_size_bytes"]
    df["cpu_bytes"]      = df["cpu_pages"]      * df["page_size_bytes"]

    return df, time_basis

def choose_gpu_max_bytes(df: pd.DataFrame) -> int:
    # Pick the modal gpu_max_bytes (or max of tied)
    counts = Counter(df["gpu_max_bytes"].tolist())
    top = counts.most_common()
    if not top:
        return int(df["gpu_max_bytes"].max())
    best = top[0][1]
    candidates = [v for v, c in top if c == best]
    return int(max(candidates))

def human_gib(x_bytes: float) -> float:
    return x_bytes / (1024**3)

def plot(df: pd.DataFrame, out_png: str, label_prefix: str = "", y_unit: str = "GiB"):
    gpu_max_b = choose_gpu_max_bytes(df)

    x = df["t_rel_s"].values
    y_gpu = [human_gib(v) for v in df["gpu_used_bytes"].values]
    y_max = human_gib(gpu_max_b)
    y_cpu_over = [human_gib(gpu_max_b + v) for v in df["cpu_bytes"].values]

    plt.figure(figsize=(14, 5))
    plt.plot(x, y_gpu, label=f"{label_prefix}GPU memory in use", linewidth=2)
    plt.axhline(y_max, linestyle="--", linewidth=1.5, label=f"GPU max ({y_max:.2f} {y_unit})")

    if (df["cpu_bytes"] > 0).any():
        plt.plot(x, y_cpu_over, label=f"{label_prefix}GPU max + CPU (overflow)", linewidth=1.8)

    plt.xlabel("Time (s)")
    plt.ylabel(f"Memory ({y_unit})")
    ttl = f"{label_prefix}Memory Manager Usage (GPU/CPU over time)".strip()
    plt.title(ttl)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Saved plot -> {out_png}")

def main():
    ap = argparse.ArgumentParser(description="Plot mem-manager memory usage from log (ns or wall-clock).")
    ap.add_argument("--log", default="mem_manager_log.txt", help="Path to mem manager log file")
    ap.add_argument("--out", default="mem_manager_mem_usage.png", help="Output PNG filename")
    ap.add_argument("--csv", default="", help="Optional: write parsed timeline to CSV")
    ap.add_argument(
        "--offset-ns",
        type=int,
        default=0,
        help="Optional additive offset (ns) to align with Nsight: t_ns = ns_rel + offset_ns",
    )
    ap.add_argument(
        "--fallback-page-size",
        type=str,
        default="",
        help="Fallback page size if not present in log, e.g. '128KB', '2MB', '1GB'",
    )
    args = ap.parse_args()

    fallback_bytes = None
    if args.fallback_page_size:
        fallback_bytes = parse_size_to_bytes(args.fallback_page_size)

    df, basis = parse_log(args.log, fallback_bytes)

    # Alignment offset to Nsight
    if args.offset_ns != 0:
        df["ns_rel"] = df["ns_rel"].astype("int64") + int(args.offset_ns)
        df["t_rel_s"] = df["ns_rel"] * 1e-9
        label_prefix = "Aligned: "
    else:
        label_prefix = ""

    if args.csv:
        cols = [
            "ns_rel", "t_rel_s",
            "page_size_bytes",
            "gpu_used_pages", "gpu_max_pages", "cpu_pages",
            "gpu_used_bytes", "gpu_max_bytes", "cpu_bytes",
        ]
        df[cols].to_csv(args.csv, index=False)
        print(f"[OK] Wrote CSV -> {args.csv} (time basis: {basis})")

    plot(df, args.out, label_prefix=label_prefix, y_unit="GiB")

if __name__ == "__main__":
    main()