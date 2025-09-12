#!/usr/bin/env python3
# nsys_preclean_stable.py
# Robust pre-clean for Nsight Systems .sqlite
# - Finds job_66:backward and decode jobs 67..71
# - Keeps GPU/CPU activity overlapping those windows + a non-overlap control decode
# - Copies small metadata tables wholesale (no fragile key assumptions)
# - Skips missing tables/columns gracefully
# - VACUUMs correctly (outside transactions) and DETACHes clean DB

import argparse, sqlite3, os, sys
from typing import List, Tuple

# Tables we typically want time-filtered
TABLES_WITH_TIME = [
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_MEMSET",
    "CUPTI_ACTIVITY_KIND_RUNTIME",
    "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
    "OSRT_API",
    "NVTX_EVENTS",
]

# Tables that may lack start/end in some exports; copy wholesale if present
COPY_WHOLE_IF_NO_TIME = [
    "CUDA_GPU_MEMORY_USAGE_EVENTS",
]

# Small metadata tables — copy wholesale if present
META_TABLES_ALWAYS_COPY = [
    "PROCESSES", "ProcessStreams", "ThreadNames",
    "TARGET_INFO_CUDA_CONTEXT_INFO", "TARGET_INFO_CUDA_STREAM",
    "TARGET_INFO_GPU", "TARGET_INFO_SESSION_START_TIME",
    "TARGET_INFO_SYSTEM_ENV", "ANALYSIS_FILE", "META_DATA_CAPTURE",
    "META_DATA_EXPORT", "ANALYSIS_DETAILS", "PROFILER_OVERHEAD",
    "StringIds",  # filtered later if NVTX kept; wholesale copy is still safe
]

# Candidate time column names (Nsight varies across versions)
CAND_START = ["start", "start_ns", "startTime", "startTimestamp"]
CAND_END   = ["end", "end_ns", "endTime", "endTimestamp", "\"end\""]

def table_exists(cur, name, schema="main"):
    cur.execute(f"SELECT COUNT(*) FROM {schema}.sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone()[0] == 1

def colnames(cur, table):
    cur.execute(f"PRAGMA table_info('{table}')")
    return [r[1] for r in cur.fetchall()]

def find_time_cols(cur, table):
    cols = colnames(cur, table)
    s = next((c for c in CAND_START if c in cols), None)
    e = next((c for c in CAND_END   if c in cols), None)
    return s, e

def attach_clean(con, out_path):
    if os.path.exists(out_path):
        os.remove(out_path)
    con.execute("ATTACH DATABASE ? AS clean", (out_path,))
    return "clean"

def ensure_empty_like(con, src, dest, table):
    cur = con.cursor()
    if table_exists(cur, table, schema=dest):
        con.execute(f'DELETE FROM {dest}."{table}"')
    else:
        con.execute(f'CREATE TABLE {dest}."{table}" AS SELECT * FROM {src}."{table}" WHERE 0')

def insert_overlapping(con, src, dest, table, start_col, end_col, windows: List[Tuple[int,int]]):
    preds, params = [], []
    for (ws, we) in windows:
        preds.append(f"(({start_col}) < ? AND ({end_col}) > ?)")
        params.extend([we, ws])
    where = " OR ".join(preds)
    con.execute(f"""
        INSERT INTO {dest}."{table}"
        SELECT * FROM {src}."{table}"
        WHERE {where}
    """, params)

def copy_whole(con, src, dest, table):
    ensure_empty_like(con, src, dest, table)
    con.execute(f'INSERT INTO {dest}."{table}" SELECT * FROM {src}."{table}"')

def dedupe(con, dest, table):
    con.execute(f'CREATE TEMP TABLE tmp_{table} AS SELECT * FROM {dest}."{table}"')
    con.execute(f'DELETE FROM {dest}."{table}"')
    con.execute(f'INSERT INTO {dest}."{table}" SELECT DISTINCT * FROM tmp_{table}')
    con.execute(f'DROP TABLE tmp_{table}')

def main():
    ap = argparse.ArgumentParser(description="Stable Nsight Systems .sqlite pre-cleaner.")
    ap.add_argument("sqlite_path", help="Path to Nsight Systems .sqlite exported DB")
    ap.add_argument("--output", help="Output cleaned sqlite (default: <input>_cleaned.sqlite)")
    ap.add_argument("--pad_ms", type=float, default=0.0, help="Padding added to each window (ms)")
    args = ap.parse_args()

    src_path = args.sqlite_path
    out_path = args.output or (os.path.splitext(src_path)[0] + "_cleaned.sqlite")
    PAD_NS = int(args.pad_ms * 1e6)

    con = sqlite3.connect(src_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Sanity: must have NVTX and StringIds
    for t in ("NVTX_EVENTS", "StringIds"):
        if not table_exists(cur, t):
            print(f"[FATAL] Missing required table: {t}", file=sys.stderr)
            sys.exit(2)

    # Locate backward
    cur.execute("""
        SELECT e.start AS s, e."end" AS e
        FROM NVTX_EVENTS e LEFT JOIN StringIds sids ON e.textId = sids.id
        WHERE COALESCE(e.text, sids.value) GLOB 'job_66:backward*'
        ORDER BY (e."end" - e.start) DESC
        LIMIT 1
    """)
    b = cur.fetchone()
    if not b:
        print("[FATAL] Could not find 'job_66:backward'", file=sys.stderr); sys.exit(2)
    b_start, b_end = int(b["s"]), int(b["e"])

    # Locate decodes 67..71
    cur.execute("""
        SELECT e.start AS s, e."end" AS e, COALESCE(e.text, sids.value) AS label
        FROM NVTX_EVENTS e LEFT JOIN StringIds sids ON e.textId = sids.id
        WHERE COALESCE(e.text, sids.value) GLOB 'job_6[7-9]:decode*'
           OR COALESCE(e.text, sids.value) GLOB 'job_7[0-1]:decode*'
        ORDER BY s
    """)
    dec = cur.fetchall()

    overlapped, nonover = [], []
    for d in dec:
        ds, de, lbl = int(d["s"]), int(d["e"]), d["label"]
        if min(de, b_end) - max(ds, b_start) > 0:
            overlapped.append((ds, de, lbl))
        else:
            nonover.append((ds, de, lbl))

    # Control: prefer job_71 trimmed to after backward + small guard
    ctrl_window = None
    for ds, de, lbl in nonover + [(int(d["s"]), int(d["e"]), d["label"]) for d in dec if "job_71:decode" in d["label"]]:
        safe_start = max(ds, b_end + 100_000)  # +0.1 ms guard
        if safe_start < de:
            ctrl_window = (safe_start, de, lbl); break

    print(f"Backward: {b_start}..{b_end} ({(b_end-b_start)/1e6:.3f} ms)")
    print(f"Overlapped decodes: {len(overlapped)}")
    if ctrl_window:
        print(f"Control decode: {ctrl_window[2]}  {ctrl_window[0]}..{ctrl_window[1]} ({(ctrl_window[1]-ctrl_window[0])/1e6:.3f} ms)")
    else:
        print("[!] No clean non-overlap control decode found; proceeding without control.")

    # Build windows (with optional padding)
    windows: List[Tuple[int,int]] = []
    def pad(s,e): return (max(0, s-PAD_NS), e+PAD_NS)
    windows.append(pad(b_start, b_end))
    for (s,e,_) in overlapped: windows.append(pad(s,e))
    if ctrl_window: windows.append(pad(ctrl_window[0], ctrl_window[1]))

    # Attach clean DB
    dest = attach_clean(con, out_path)

    try:
        # Copy metadata wholesale (safe, small)
            # Copy metadata wholesale (safe, small)
        for t in META_TABLES_ALWAYS_COPY:
            if table_exists(cur, t):
                copy_whole(con, "main", dest, t)

        # --- NEW: copy all ENUM_* tables wholesale ---
        cur.execute("SELECT name FROM main.sqlite_master WHERE type='table' AND name LIKE 'ENUM_%'")
        for (enum_t,) in cur.fetchall():
            ensure_empty_like(con, "main", dest, enum_t)
            con.execute(f'INSERT INTO {dest}."{enum_t}" SELECT * FROM main."{enum_t}"')


        # Time-ranged tables + fallbacks
        copied_nvtx = False
        for t in TABLES_WITH_TIME + COPY_WHOLE_IF_NO_TIME:
            if not table_exists(cur, t):
                continue
            s_col, e_col = find_time_cols(cur, t)
            if s_col and e_col:
                ensure_empty_like(con, "main", dest, t)
                insert_overlapping(con, "main", dest, t, f'"{s_col}"', f'"{e_col}"', windows)
                dedupe(con, dest, t)
                if t == "NVTX_EVENTS":
                    copied_nvtx = True
            else:
                if t in COPY_WHOLE_IF_NO_TIME:
                    copy_whole(con, "main", dest, t)
                    dedupe(con, dest, t)
                else:
                    # silent skip: table lacks time columns and not whitelisted
                    pass

        # Filter StringIds to only those referenced by kept NVTX rows
        # Copy StringIds wholesale so kernel/API names resolve
        if table_exists(cur, "StringIds"):
            ensure_empty_like(con, "main", dest, "StringIds")
            con.execute(f'INSERT INTO {dest}."StringIds" SELECT * FROM main."StringIds"')
            dedupe(con, dest, "StringIds")

        # Commit all writes before VACUUM
        con.commit()

        # VACUUM must run in autocommit; vacuum the attached clean DB
        con.isolation_level = None
        con.execute("VACUUM clean")
        con.execute("DETACH DATABASE clean")
        print(f"\nDone. Wrote: {out_path}")

    except Exception as e:
        # Try to detach clean to avoid leaving the file locked
        try:
            con.isolation_level = None
            con.execute("DETACH DATABASE clean")
        except Exception:
            pass
        print(f"[FATAL] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        con.close()

if __name__ == "__main__":
    main()