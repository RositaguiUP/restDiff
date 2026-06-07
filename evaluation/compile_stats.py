import argparse
import json
import csv
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Compile validation stats across scenes, floors, and steps.")
    parser.add_argument("--stage", required=True, help="Target stage (e.g., 'warmup')")
    parser.add_argument("--version", required=True, help="Target version (e.g., 'v5.1')")
    parser.add_argument("--steps", type=int, nargs="+", required=True, help="One or more step integers (e.g., 29999 30000)")
    parser.add_argument("--scenes", nargs="*", help="Specific scenes to parse. Parses all if omitted.")
    parser.add_argument("--results-dir", default="results", help="Base results directory")
    parser.add_argument("--output", default="stats_comparison.csv", help="Output CSV path")
    
    args = parser.parse_args()
    results_base = Path(args.results_dir)

    if not results_base.exists():
        print(f"[-] Error: Results directory '{results_base}' does not exist.")
        sys.exit(1)

    if args.scenes:
        scenes = [results_base / scene for scene in args.scenes]
    else:
        scenes = [d for d in results_base.iterdir() if d.is_dir()]

    all_data = []

    for step in args.steps:
        json_filename = f"val_step{step}.json"
        
        for scene_path in scenes:
            scene_name = scene_path.name
            target_dir = scene_path / args.stage / args.version

            if not target_dir.exists():
                continue

            for floor_path in target_dir.iterdir():
                # Avoid utility directories
                if not floor_path.is_dir() or floor_path.name in ["checkpoints", "renders"]:
                    continue

                floor_name = floor_path.name
                # Corrected path layout: floor_path / "stats" / json_filename
                json_path = floor_path / "stats" / json_filename

                if json_path.exists():
                    try:
                        with open(json_path, 'r') as f:
                            stats = json.load(f)

                        row_data = {
                            "scene": scene_name,
                            "stage": args.stage,
                            "version": args.version,
                            "floor": floor_name,
                            "step": step
                        }
                        row_data.update(stats)
                        all_data.append(row_data)
                    except Exception as e:
                        print(f"[-] Error reading {json_path}: {e}")

    if not all_data:
        print("[-] No matching metrics found for the specified configurations.")
        sys.exit(0)

    # Dynamic header tracking
    metadata_cols = ["scene", "stage", "version", "floor", "step"]
    stat_cols = list({k for row in all_data for k in row.keys() if k not in metadata_cols})
    fieldnames = metadata_cols + sorted(stat_cols)

    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)
        
    print(f"[+] Successfully compiled {len(all_data)} rows into '{args.output}'")

if __name__ == "__main__":
    main()
    
    
# python evaluation/compile_stats.py --stage warmup --version v6.0 --steps 29999 --scenes 6VSV7_695_v2 --results-dir ./results --output results/_stats_/metrics_v6.1.csv