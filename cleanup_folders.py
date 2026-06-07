import argparse
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Delete a specific folder inside scenes and floors for a given stage and optional version.")
    parser.add_argument("--stage", required=True, help="Target stage (e.g., 'warmup')")
    parser.add_argument("--version", default=None, help="Target version (e.g., 'v6.0'). If omitted, runs for ALL versions found.")
    parser.add_argument("--target", required=True, help="File or folder name to delete (e.g. 'comparisons' or 'metrics.csv')")
    parser.add_argument("--results-dir", default="results", help="Base results directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview matches without deleting anything")

    args = parser.parse_args()
    results_base = Path(args.results_dir)

    if not results_base.exists():
        print(f"[-] Base results directory '{results_base}' does not exist.")
        return

    target_version_label = args.version if args.version else "[ALL VERSIONS]"

    print("=" * 60)
    print(f"=== {'[DRY RUN MODE]' if args.dry_run else '[EXECUTE DELETION MODE]'} ===")
    print(f"Targeting: {results_base}/<scene>/{args.stage}/{target_version_label}/<floor>/{args.target}")
    print("=" * 60)

    match_count = 0

    # Loop through each scene directory
    for scene_dir in results_base.iterdir():
        if not scene_dir.is_dir():
            continue

        stage_dir = scene_dir / args.stage
        if not stage_dir.exists():
            continue

        # If a version is specified, target only that folder. Otherwise, collect all subdirectories.
        if args.version:
            version_dirs = [stage_dir / args.version]
        else:
            version_dirs = [d for d in stage_dir.iterdir() if d.is_dir()]

        # Loop through the resolved version directories
        for version_dir in version_dirs:
            if not version_dir.exists():
                continue

            # Loop through each floor directory inside the scene version
            for floor_dir in version_dir.iterdir():
                if not floor_dir.is_dir():
                    continue
                
                # Define the exact folder target to destroy
                delete_target = floor_dir / Path(args.target)
                
                if delete_target.exists():
                    match_count += 1
                    if args.dry_run:
                        kind = "DIR" if delete_target.is_dir() else "FILE"
                        print(f"  [WOULD DELETE {kind}] {delete_target}")
                        print(f"  [WOULD DELETE] {delete_target}")
                    else:
                        try:
                            if delete_target.is_dir():
                                shutil.rmtree(delete_target)
                                print(f"  [DELETED DIR ] {delete_target}")
                            else:
                                delete_target.unlink()
                                print(f"  [DELETED FILE] {delete_target}")
                        except Exception as e:
                            print(f"  [ERROR]        Failed to delete {delete_target}: {e}")

    print("=" * 60)
    if args.dry_run:
        print(f"Dry run complete. Found {match_count} target folders to delete.")
        print("Run again without '--dry-run' to permanently delete them.")
    else:
        print(f"Execution complete. Cleaned up {match_count} folders.")
    print("=" * 60)

if __name__ == "__main__":
    main()
    
# python cleanup_folders.py --stage warmup --version v6.0 --target comparisons --dry-run
# python cleanup_folders.py --stage warmup --folder comparisons --dry-run

# python cleanup_folders.py --stage warmup --version v6.0 --target checkpoints/ckpt_warmup_latest.pt --dry-run
# python cleanup_folders.py --stage warmup --version v6.0 --target wandb --dry-run