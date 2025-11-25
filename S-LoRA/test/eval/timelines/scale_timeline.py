from asyncio import subprocess
import os
import pandas as pd
import numpy as np
import random


def apply_smoothing_outside_window(scaled, start, end, radius=2):
    """
    Smooth only the region OUTSIDE the forced pattern [start, end].
    """
    for sec in range(start - radius, end + radius + 1):
        if sec < start or sec > end:  # only outside
            if sec not in scaled.index:
                continue

            # linear soften based on distance from nearest boundary
            dist = 0
            if sec < start:
                dist = (start - sec) / radius
            else:
                dist = (sec - end) / radius

            dist = min(max(dist, 0), 1)
            # dist = 0 → fully pattern value
            # dist = 1 → original value
            nearest = start if sec < start else end

            scaled.loc[sec] = (
                scaled.loc[sec] * dist + scaled.loc[nearest] * (1 - dist)
            )


def find_random_window(scaled, length, below_avg_mask, reserved_mask):
    """
    Choose a random window of given length such that:
      • all indices exist in scaled index
      • the window is NOT reserved
      • the average RPS in the window is below average_scaled
    """
    valid_starts = []

    for start in scaled.index:
        end = start + length - 1
        if end not in scaled.index:
            continue

        # Check reserve conflicts
        if any(reserved_mask.get(i, False) for i in range(start, end + 1)):
            continue

        # Check the "below average" criterion
        window_vals = scaled.loc[start:end]
        if window_vals.mean() < scaled.mean():
            valid_starts.append(start)

    if not valid_starts:
        # fallback: just find ANY available non-reserved window
        for start in scaled.index:
            end = start + length - 1
            if end not in scaled.index:
                continue
            if any(reserved_mask.get(i, False) for i in range(start, end + 1)):
                continue
            valid_starts.append(start)

    if not valid_starts:
        raise RuntimeError("No valid windows available for inserting pattern.")

    return random.choice(valid_starts)


def scale_rps(
    csv_in,
    csv_out,
    target_avg_rps,
    max_rps=None,
    must_have_rps=None,      # list of patterns, e.g. [[0,0,0], [0,1]]
    min_prompt_length=None,
    min_new_tokens=None,
    prompt_noise_frac=0.0,
    new_tokens_noise_frac=0.0,
):
    """
    Final version:
      • Scale RPS to target_avg
      • Apply max_rps cap
      • For each pattern in must_have_rps, insert it:
          - find existing matching window OR
          - randomly place inside a below-average window
          - reserve used seconds
          - no compensation
      • Smooth edges outside forced patterns
      • Apply optional prompt/new token constraints and noise
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_in = os.path.join(current_dir, csv_in)
    csv_out = os.path.join(current_dir, csv_out)

    df = pd.read_csv(csv_in)
    if "timestamp_s" not in df.columns:
        raise ValueError("CSV must contain timestamp_s")

    # ------------------------------------------------------------
    # Step 1 — Compute per-second RPS
    # ------------------------------------------------------------
    df["sec"] = np.floor(df["timestamp_s"]).astype(int)
    counts = df.groupby("sec").size().sort_index()

    original_avg = counts.mean()
    original_min = counts.min()
    original_max = counts.max()

    if original_avg == 0:
        raise ValueError("Timeline has zero RPS.")

    # ------------------------------------------------------------
    # Step 2 — Scale to reach target RPS
    # ------------------------------------------------------------
    scale_factor = target_avg_rps / original_avg
    scaled = counts * scale_factor

    # ------------------------------------------------------------
    # Step 3 — Apply max_rps limit
    # ------------------------------------------------------------
    if max_rps is not None:
        scaled = np.minimum(scaled, max_rps)

    # ------------------------------------------------------------
    # Step 4 — Insert forced patterns (ABS values)
    # ------------------------------------------------------------
    reserved_mask = {}  # seconds already used by earlier patterns

    if must_have_rps:
        for pattern in must_have_rps:

            pattern = list(pattern)
            L = len(pattern)

            # First: try to find an EXACT existing match
            found = False
            for start in scaled.index:
                end = start + L - 1
                if end not in scaled.index:
                    continue
                if any(reserved_mask.get(i, False) for i in range(start, end + 1)):
                    continue

                window = scaled.loc[start:end].round().astype(int).tolist()
                if window == pattern:
                    # Mark reserved
                    for i in range(start, end + 1):
                        reserved_mask[i] = True
                    found = True
                    break
                # skip any window touching reserved indices
                
            if found:
                continue

            # No exact match → choose a random suitable window
            below_avg_mask = scaled < scaled.mean()
            start = find_random_window(scaled, L, below_avg_mask, reserved_mask)
            end = start + L - 1

            # Apply forced pattern
            for i in range(L):
                scaled.loc[start + i] = pattern[i]
                reserved_mask[start + i] = True

            # Smooth outside the pattern
            apply_smoothing_outside_window(scaled, start, end, radius=2)

    # Final rounding
    scaled = scaled.round().astype(int)

    # ------------------------------------------------------------
    # Step 5 — Reconstruct timeline rows
    # ------------------------------------------------------------
    rows = []
    for sec, new_count in scaled.items():
        orig_rows = df[df["sec"] == sec]
        if len(orig_rows) == 0:
            continue

        sampled = orig_rows.sample(
            n=new_count,
            replace=(new_count > len(orig_rows))
        ).copy()

        sampled["timestamp_s"] = sec + np.linspace(0, 0.999, new_count)
        rows.append(sampled)

    df_scaled = pd.concat(rows, ignore_index=True)
    df_scaled = df_scaled.sort_values("timestamp_s").reset_index(drop=True)

    # ------------------------------------------------------------
    # Step 6 — min prompt/new_tokens + noise
    # ------------------------------------------------------------
    if min_prompt_length is not None:
        df_scaled.loc[df_scaled["prompt_length"] < min_prompt_length, "prompt_length"] = min_prompt_length

    if min_new_tokens is not None:
        df_scaled.loc[df_scaled["max_new_tokens"] < min_new_tokens, "max_new_tokens"] = min_new_tokens

    if prompt_noise_frac > 0:
        noise = (np.random.rand(len(df_scaled)) * 2 - 1) * prompt_noise_frac
        df_scaled["prompt_length"] = (
            df_scaled["prompt_length"] * (1 + noise)
        ).round().clip(lower=1)

    if new_tokens_noise_frac > 0:
        noise = (np.random.rand(len(df_scaled)) * 2 - 1) * new_tokens_noise_frac
        df_scaled["max_new_tokens"] = (
            df_scaled["max_new_tokens"] * (1 + noise)
        ).round().clip(lower=1)

    df_scaled["prompt_length"] = df_scaled["prompt_length"].astype(int)
    df_scaled["max_new_tokens"] = df_scaled["max_new_tokens"].astype(int)

    # ------------------------------------------------------------
    # Step 7 — Final stats report
    # ------------------------------------------------------------
    df_scaled["sec"] = np.floor(df_scaled["timestamp_s"]).astype(int)
    final_counts = df_scaled.groupby("sec").size()

    print("\n================= RPS SUMMARY =================")
    print(f"Original avg={original_avg:.2f}, min={original_min}, max={original_max}")
    print(f"Final avg   ={final_counts.mean():.2f}, min={final_counts.min()}, max={final_counts.max()}")
    print(f"Target avg  ={target_avg_rps}")
    print("================================================")

    # Save result
    df_scaled.drop(columns=["sec"]).to_csv(csv_out, index=False)
    print(f"✅ Saved scaled timeline → {csv_out}")
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        plot_script = os.path.join(current_dir, "plot_timeline.py")
        if not os.path.exists(plot_script):
            print(f"⚠️ plot_timeline.py not found at {plot_script}, skipping plot.")
            return
        print(f"📈 Calling plot_timeline.py...")
        os.system(f"python {plot_script} {csv_out}")
        print(f"✅ Plot generated successfully.")
    except Exception as e:
        print(f"⚠️ Unexpected error while plotting: {e}")
    return df_scaled


# Example
if __name__ == "__main__":
    scale_rps(
        "nutanix.csv",
        "nutanix_scaled.csv",
        target_avg_rps=4,
        max_rps=18,
        must_have_rps=[],  # NEW
        min_prompt_length=50,
        min_new_tokens=40,
        prompt_noise_frac=0.3,
        new_tokens_noise_frac=0.3
    )