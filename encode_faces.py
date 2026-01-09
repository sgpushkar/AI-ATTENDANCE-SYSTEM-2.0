# encode_faces.py — improved standalone encoder (optional)
import os, pickle
from pathlib import Path
import face_recognition
from PIL import Image
import numpy as np

DATASET_DIR = Path("dataset")
OUTPUT_DIR = Path("encodings")
OUTPUT_FILE = OUTPUT_DIR / "known_faces.pkl"
MAX_DIM = 1600

def gather_images(dataset_dir):
    persons = []
    if not dataset_dir.exists():
        print("[WARN] dataset/ not found.")
        return persons
    for person_name in sorted(p for p in os.listdir(dataset_dir) if not p.startswith(".")):
        person_path = dataset_dir / person_name
        if not person_path.is_dir(): continue
        images = [person_path / f for f in sorted(os.listdir(person_path)) if f.lower().endswith((".jpg",".jpeg",".png"))]
        if images:
            persons.append((person_name, images))
    return persons

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    persons = gather_images(DATASET_DIR)
    known_encodings = []
    known_names = []
    if not persons:
        print("[WARN] No persons found.")
        return
    for name, images in persons:
        print(f"[INFO] encoding {name} ({len(images)} images)")
        for img_path in images:
            try:
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    if max(im.size) > MAX_DIM:
                        ratio = MAX_DIM / max(im.size)
                        im = im.resize((int(im.width*ratio), int(im.height*ratio)), Image.LANCZOS)
                    img_np = np.array(im)
                boxes = face_recognition.face_locations(img_np, model="hog")
                encs = face_recognition.face_encodings(img_np, boxes)
                if encs:
                    known_encodings.append(encs[0])
                    known_names.append(name)
                else:
                    print("  [WARN] no face in", img_path)
            except Exception as e:
                print("  [WARN] skip", img_path, e)
    if not known_encodings:
        print("[ERROR] No encodings produced.")
        return
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)
    print(f"[DONE] saved {len(known_encodings)} encodings to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
