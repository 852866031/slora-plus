import pandas as pd

def extract_minutes_rebase(
    input_csv: str,
    output_csv: str,
    minute_start: int,
    minute_end: int,
):
    """
    Extract rows whose timestamps fall between minute_start (inclusive)
    and minute_end (exclusive), and rebase timestamps so the first row
    starts at 0.000 seconds.

    minute = timestamp_s // 60
    """

    # Load
    df = pd.read_csv(input_csv)

    # Compute minute bucket
    df["minute"] = (df["timestamp_s"] // 60).astype(int)

    # Filter by minute range
    mask = (df["minute"] >= minute_start) & (df["minute"] < minute_end)
    out_df = df[mask].copy()

    if out_df.empty:
        print("No rows found in the specified minute range.")
        return

    # Rebase timestamps
    first_ts = out_df["timestamp_s"].min()
    out_df["timestamp_s"] = out_df["timestamp_s"] - first_ts

    # Drop helper column
    out_df = out_df.drop(columns=["minute"])

    # Save
    out_df.to_csv(output_csv, index=False)

    print(f"Extracted {len(out_df)} rows → {output_csv}")
    print(f"Rebased timestamps: original start {first_ts:.3f}s → new start 0.000s")

extract_minutes_rebase(
    input_csv="timeline_live.csv",
    output_csv="timeline_extracted.csv",
    minute_start=15,
    minute_end=17
)