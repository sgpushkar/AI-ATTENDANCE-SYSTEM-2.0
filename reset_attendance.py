#!/usr/bin/env python3
"""
reset_attendance.py

Usage:
  # dry-run (shows what would be removed)
  python reset_attendance.py --dry

  # remove only logs/uploads/faces/timeline (keeps dataset & encodings)
  python reset_attendance.py --mode logs

  # full wipe (logs + uploads + encodings + dataset)
  python reset_attendance.py --mode full

  # same but without backup (not recommended)
  python reset_attendance.py --mode logs --no-backup
"""
import argparse
import shutil
import zipfile
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent.resolve()
DATASET_DIR = BASE / "dataset"
ENC_DIR = BASE / "encodings"
UPLOADS = BASE / "uploads"
FACES_DIR = UPLOADS / "faces"
ATT_CSV = BASE / "attendance.csv"
TIMELINE_FILE = BASE / "timeline.json"
THUMB_DIR = UPLOADS / "thumbs"
BACKUPS_DIR = BASE / "backups"

def make_backup(target_paths, outdir=BACKUPS_DIR):
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zname = outdir / f"backup_{ts}.zip"
    with zipfile.ZipFile(str(zname), "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in target_paths:
            if not p.exists():
                continue
            if p.is_file():
                zf.write(str(p), arcname=p.name)
            else:
                for fp in sorted(p.rglob("*")):
                    if fp.is_file():
                        zf.write(str(fp), arcname=str(fp.relative_to(BASE)))
    return zname

def remove_path(p):
    if not p.exists():
        return
    if p.is_file():
        p.unlink()
        return
    # directory
    shutil.rmtree(p)

def main(args):
    # define sets
    logs_and_uploads = [ATT_CSV, UPLOADS, TIMELINE_FILE, THUMB_DIR]
    enc_and_index = [ENC_DIR]
    dataset = [DATASET_DIR]

    if args.mode == "logs":
        to_remove = logs_and_uploads
    elif args.mode == "full":
        to_remove = logs_and_uploads + enc_and_index + dataset
    else:
        print("unknown mode:", args.mode)
        return

    # Dry run: list items
    print("MODE:", args.mode)
    print("Targets to remove:")
    for p in to_remove:
        print("  -", p, "(exists)" if p.exists() else "(missing)")

    if args.dry:
        print("\nDry run complete. No changes made.")
        return

    if args.backup:
        print("\nCreating backup zip of targets (this may take a moment)...")
        try:
            z = make_backup(to_remove)
            print("Backup created:", z)
        except Exception as e:
            print("Backup failed:", e)
            if not args.force:
                print("Abort (use --force to continue despite backup failure).")
                return

    # Proceed with deletion
    print("\nDeleting targets...")
    for p in to_remove:
        try:
            if p.exists():
                print("Removing", p)
                remove_path(p)
            else:
                print("Skipping (missing)", p)
        except Exception as e:
            print("Failed to remove", p, ":", e)

    # Also ensure an empty attendance CSV header if logs-only mode wants to keep file but empty
    if args.mode == "logs":
        # recreate a fresh attendance.csv header (empty)
        try:
            with open(ATT_CSV, "w", newline="") as f:
                f.write("name,timestamp,confidence,session_id,image,subject,teacher\n")
            print("Recreated empty attendance.csv header.")
        except Exception as e:
            print("Could not recreate attendance.csv:", e)

    print("\nDone. Project history cleaned for mode:", args.mode)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["logs", "full"], default="logs",
                   help="logs = delete attendance.csv, uploads, faces, timeline; full = also delete encodings & dataset")
    p.add_argument("--dry", action="store_true", help="dry run (show what would be removed)")
    p.add_argument("--no-backup", dest="backup", action="store_false", help="skip backup zip")
    p.add_argument("--force", action="store_true", help="force continue even if backup fails")
    args = p.parse_args()
    main(args)
