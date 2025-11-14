import csv
import os
import subprocess

def generate_peak_valley_timeline(
    peak_duration: float,
    valley_duration: float,
    prompt_length: int,
    max_new_tokens: int,
    total_cycles: int,
    output_file: str,
    peak_rps: int = 30,
    valley_rps: int = 1,
    initial_idle: float = 1.0,
):
    """
    Generate a CSV timeline simulating alternating 'peak' and 'valley' request periods,
    starting with an initial idle phase.
    """
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_s", "prompt_length", "max_new_tokens", "second", "index_in_second"])

        time = initial_idle
        total_reqs = 0

        for cycle in range(total_cycles):
             # --- Valley Phase ---
            valley_count = int(valley_duration * valley_rps)
            for i in range(valley_count):
                timestamp = time + i / max(1, valley_rps)
                second = int(timestamp)
                writer.writerow([round(timestamp, 6), prompt_length, max_new_tokens, second, i % max(1, valley_rps)])
                total_reqs += 1
            time += valley_duration
            # --- Peak Phase ---
            peak_count = int(peak_duration * peak_rps)
            for i in range(peak_count):
                timestamp = time + i / peak_rps
                second = int(timestamp)
                writer.writerow([round(timestamp, 6), prompt_length, max_new_tokens, second, i % peak_rps])
                total_reqs += 1
            time += peak_duration

           

    print(f"✅ Generated {output_file} with {total_reqs} requests.")

    # === Automatically call plot_timeline.py ===
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        plot_script = os.path.join(current_dir, "plot_timeline.py")
        if not os.path.exists(plot_script):
            print(f"⚠️ plot_timeline.py not found at {plot_script}, skipping plot.")
            return
        print(f"📈 Calling plot_timeline.py to generate plot...")
        subprocess.run(["python", plot_script, output_file], check=True)
        print(f"✅ Plot generated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to plot timeline: {e}")
    except Exception as e:
        print(f"⚠️ Unexpected error while plotting: {e}")


if __name__ == "__main__":
    # === Example Usage ===
    peak_duration = 3.0
    valley_duration = 2.0
    prompt_length = 200
    max_new_tokens = 80
    total_cycles = 4
    peak_rps = 9
    valley_rps = 1
    initial_idle = 1.0

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(
        current_dir,
        f"tl_pdc_{peak_rps}p_{valley_rps}v_{total_cycles}cyc.csv"
    )

    generate_peak_valley_timeline(
        peak_duration,
        valley_duration,
        prompt_length,
        max_new_tokens,
        total_cycles,
        output_path,
        peak_rps=peak_rps,
        valley_rps=valley_rps,
        initial_idle=initial_idle,
    )