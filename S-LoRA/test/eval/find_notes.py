#!/usr/bin/env python3
import argparse, sqlite3, sys, textwrap

def colnames(cur, table):
    cur.execute(f"PRAGMA table_info('{table}')")
    return [r[1] for r in cur.fetchall()]

def select_exists(cur, table):
    cur.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    )
    return cur.fetchone()[0] == 1

def has_col(cols, name):
    return any(c.lower() == name.lower() for c in cols)

def coalesce_label_expr(cols_e, cols_s):
    # Prefer NVTX_EVENTS.text if present; otherwise StringIds.string via textId
    parts = []
    if has_col(cols_e, "text"):
        parts.append("e.text")
    if has_col(cols_e, "textId") and select_stringids:
        if has_col(cols_s, "id") and has_col(cols_s, "string"):
            parts.append("s.string")
    if not parts:
        return "'<no_label_cols>'"
    return "COALESCE(" + ", ".join(parts) + ")"

def fetch_events(cur, label_globs):
    # NVTX_EVENTS has 'text' + 'textId', StringIds has 'value'
    join = "LEFT JOIN StringIds s ON e.textId = s.id"
    label_expr = "COALESCE(e.text, s.value)"

    wheres = " OR ".join([f"{label_expr} GLOB ?" for _ in label_globs])
    params = label_globs

    q = f"""
        SELECT
          e.rowid AS nvtx_id,
          e.start      AS start_ns,
          e."end"      AS end_ns,
          (e."end" - e.start)/1e6 AS duration_ms,
          e.eventType  AS type,
          {label_expr} AS label
        FROM NVTX_EVENTS e
        {join}
        WHERE {wheres}
        ORDER BY start_ns
    """
    cur.execute(q, params)
    return cur.fetchall()

def fmt_ns(ns):
    if ns is None: return "None"
    return f"{ns:,}"

def print_events(title, rows):
    print(f"\n== {title} ({len(rows)} hit{'s' if len(rows)!=1 else ''}) ==")
    if not rows:
        return
    # Summaries
    for r in rows:
        nvtx_id, start_ns, end_ns, dur_ms, typ, label = r
        dur_txt = f"{dur_ms:.3f} ms" if dur_ms is not None else "n/a"
        print(f"- id={nvtx_id:<8} start={fmt_ns(start_ns)}  end={fmt_ns(end_ns)}  dur={dur_txt}  type={typ}  label={label}")
    # Mark/range hint
    maybe_marks = [r for r in rows if r[2] in (None, 0) or (r[3] is not None and r[3] < 0.05)]
    if maybe_marks:
        print("  [!] Some entries look like NVTX *marks* (no/near-zero duration). "
              "If you expected ranges, check how you emit NVTX (push/pop or range start/end).")

def main():
    p = argparse.ArgumentParser(
        description="Probe NVTX labels and time ranges (Nsight Systems .sqlite).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python find_nvtx.py my_trace.sqlite
        """),
    )
    p.add_argument("sqlite_path", help="Path to Nsight Systems .sqlite exported DB")
    args = p.parse_args()

    con = sqlite3.connect(args.sqlite_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Check required tables
    global select_stringids
    have_nvtx = select_exists(cur, "NVTX_EVENTS")
    select_stringids = select_exists(cur, "StringIds")
    if not have_nvtx:
        print("[FATAL] NVTX_EVENTS table not found. Are you sure this is an Nsight Systems export?", file=sys.stderr)
        sys.exit(2)

    # Show schemas briefly
    print("Tables present (subset):")
    for t in ("NVTX_EVENTS", "StringIds", "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY",
              "CUPTI_ACTIVITY_KIND_MEMSET", "CUPTI_ACTIVITY_KIND_RUNTIME", "OSRT_API"):
        print(f"  - {t}: {'yes' if select_exists(cur, t) else 'no'}")

    print("\nNVTX_EVENTS columns:", colnames(cur, "NVTX_EVENTS"))
    if select_stringids:
        print("StringIds columns:", colnames(cur, "StringIds"))

    # 1) Find job_66:backward
    backward = fetch_events(cur, ["job_66:backward*"])
    print_events("job_66:backward", backward)

    # 2) Find decode windows job_67..job_71
    decodes = fetch_events(cur, ["job_6[7-9]:decode*", "job_7[0-1]:decode*"])
    print_events("decode jobs 67..71", decodes)

    # 3) If we have a single backward range, compute overlaps with decode ranges
    if backward:
        # choose the longest matching backward range (common if nested pushes)
        b = max(backward, key=lambda r: (r[3] or -1))
        b_start, b_end = b[1], b[2]
        if b_start is not None and b_end is not None and b_end > b_start:
            print(f"\nBackward window (chosen): start={fmt_ns(b_start)}, end={fmt_ns(b_end)}, "
                  f"duration={(b_end-b_start)/1e6:.3f} ms")
            overlapped, non_overlapped = [], []
            for d in decodes:
                ds, de = d[1], d[2]
                if ds is None or de is None:
                    continue
                overlap = max(0, min(b_end, de) - max(b_start, ds))
                if overlap > 0:
                    overlapped.append((d, overlap))
                else:
                    non_overlapped.append(d)
            print(f"- decode overlapped with backward: {len(overlapped)}")
            for d, ov in overlapped:
                print(f"  > {d[5]}  overlap={(ov/1e6):.3f} ms  decode_dur={(d[3] or 0):.3f} ms")
            # pick one non-overlapped decode closest to the backward end as the control
            if non_overlapped:
                ctrl = min(non_overlapped, key=lambda d: abs((d[1] or 0) - b_end))
                print(f"- control decode (non-overlap, nearest in time): {ctrl[5]} "
                      f"[dur={(ctrl[3] or 0):.3f} ms, start={fmt_ns(ctrl[1])}]")
        else:
            print("[!] Could not determine a valid backward range (missing/invalid start/end).")

    con.close()

if __name__ == "__main__":
    main()