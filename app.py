# app.py - COMPLETE FIXED VERSION
# Clean Flask app with MediaPipe + FaceNet, subjects & teachers support.
# Patched: timeline, encode_status, reverify, realtime, manual mark, notifications, safer uploads, streaks
# ADDED: Authentication system

import os, csv, json, base64
from io import BytesIO
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for,
    send_file, flash, jsonify, current_app, session
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
import numpy as np
import cv2
import sqlite3

# ML libs
import mediapipe as mp
import torch
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms

# notifications
import smtplib
from email.mime.text import MIMEText
try:
    from twilio.rest import Client as TwilioClient
    _have_twilio = True
except Exception:
    _have_twilio = False

BASE = Path(__file__).parent.resolve()
DATASET_DIR = BASE / "dataset"
ENC_DIR = BASE / "encodings"
UPLOADS = BASE / "uploads"
FACES_DIR = UPLOADS / "faces"
ATT_CSV = BASE / "attendance.csv"

for d in (DATASET_DIR, ENC_DIR, UPLOADS, FACES_DIR):
    d.mkdir(parents=True, exist_ok=True)

THUMB_DIR = UPLOADS / "thumbs"
THUMB_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.secret_key = os.environ.get("ATTEND_SECRET", "change-me-very-secret-key-12345")

# safety configs
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024   # 8 MB limit for uploads

SUBJECTS_FILE = ENC_DIR / "subjects.json"
TEACHERS_FILE = ENC_DIR / "teachers.json"
INDEX_FILE = ENC_DIR / "index.json"
TIMELINE_FILE = BASE / "timeline.json"

# config
REVERIFY_THRESH = float(os.environ.get("REVERIFY_THRESH", "0.60"))  # below this -> ask reverify

# Database connection
def get_db():
    conn = sqlite3.connect('attendance.db')
    conn.row_factory = sqlite3.Row
    return conn

# Initialize authentication tables
def init_auth():
    conn = get_db()
    cur = conn.cursor()
    
    # Create users table if not exists
    cur.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT CHECK(role IN ('admin','teacher','student')) NOT NULL,
        full_name TEXT,
        department TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Check if we need to add default users
    cur.execute("SELECT COUNT(*) as count FROM users")
    if cur.fetchone()[0] == 0:
        # Add default users
        default_users = [
            ('admin', generate_password_hash('admin123'), 'admin', 'System Administrator', 'Administration'),
            ('teacher1', generate_password_hash('teacher123'), 'teacher', 'John Teacher', 'Computer Science'),
            ('student1', generate_password_hash('student123'), 'student', 'Alice Student', 'Computer Science')
        ]
        
        for username, pwd_hash, role, full_name, dept in default_users:
            try:
                cur.execute(
                    "INSERT INTO users (username, password_hash, role, full_name, department) VALUES (?, ?, ?, ?, ?)",
                    (username, pwd_hash, role, full_name, dept)
                )
            except sqlite3.IntegrityError:
                pass
    
    conn.commit()
    conn.close()

# Authentication decorators
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_role' not in session:
                flash('Please log in first', 'error')
                return redirect(url_for('login'))
            
            if session['user_role'] not in roles:
                flash('You do not have permission to access this page', 'error')
                return redirect(url_for('index'))
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def is_ajax():
    """Return True if request appears to be an AJAX/XHR request."""
    return request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json

def load_subjects():
    if SUBJECTS_FILE.exists():
        try:
            data = json.loads(SUBJECTS_FILE.read_text())
            if isinstance(data, list):
                return data
        except:
            pass
    default = ["General"]
    SUBJECTS_FILE.write_text(json.dumps(default))
    return default

def save_subjects(lst):
    SUBJECTS_FILE.write_text(json.dumps(lst))

def load_teachers():
    if TEACHERS_FILE.exists():
        try:
            data = json.loads(TEACHERS_FILE.read_text())
            if isinstance(data, dict):
                return data
        except:
            pass
    base = {}
    TEACHERS_FILE.write_text(json.dumps(base))
    return base

def save_teachers(dct):
    TEACHERS_FILE.write_text(json.dumps(dct))

def allowed_file(filename):
    return filename and "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED

def _ensure_att_csv():
    if not ATT_CSV.exists():
        with open(ATT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "timestamp", "confidence", "session_id", "image", "subject", "teacher"])

def _read_attendance_rows():
    rows = []
    if not ATT_CSV.exists():
        return rows
    with open(ATT_CSV, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for r in reader:
            if not r:
                continue
            while len(r) < 7:
                r.append("")
            rows.append(tuple(r[:7]))
    return rows

def _get_all_students():
    if not DATASET_DIR.exists():
        return []
    return [p.name for p in sorted(DATASET_DIR.iterdir()) if p.is_dir()]

def _date_str_from_ts(ts_str):
    try:
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%Y-%m-%d")
    except:
        try:
            return ts_str.split()[0]
        except:
            return ""

# Timeline helpers
def _append_timeline_entry(timestamp, photo_name, marked, faces_found, subject="", teacher=""):
    entry = {
        "ts": timestamp,
        "photo": photo_name,
        "marked": int(marked),
        "faces_found": int(faces_found),
        "subject": subject or "General",
        "teacher": teacher or ""
    }
    data = []
    if TIMELINE_FILE.exists():
        try:
            data = json.loads(TIMELINE_FILE.read_text())
        except:
            data = []
    data.insert(0, entry)  # newest first
    data = data[:100]
    TIMELINE_FILE.write_text(json.dumps(data, indent=2))

def _save_thumbnail_for_photo(photo_path, maxw=320):
    """Create small thumbnail for timeline UI."""
    try:
        img = Image.open(str(photo_path)).convert("RGB")
        img.thumbnail((maxw, maxw), Image.LANCZOS)
        fname = f"thumb_{photo_path.name}"
        out = THUMB_DIR / fname
        img.save(str(out), format="JPEG", quality=70)
        # return URL-ish path relative to UPLOADS
        return f"thumbs/{fname}"
    except Exception:
        current_app.logger.exception("save thumbnail")
        return None

# Model init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
facenet_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.45)

def mediapipe_detect_boxes(image_rgb):
    results = mp_face.process(image_rgb)
    boxes = []
    if not results or not results.detections:
        return boxes
    h, w, _ = image_rgb.shape
    for det in results.detections:
        bb = det.location_data.relative_bounding_box
        x1 = int(bb.xmin * w); y1 = int(bb.ymin * h)
        x2 = int((bb.xmin + bb.width) * w); y2 = int((bb.ymin + bb.height) * h)
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(w-1, x2); y2 = min(h-1, y2)
        boxes.append([x1, y1, x2, y2])
    return boxes

def crop_face_from_box(img_rgb, box, expand=0.15):
    # img_rgb may be numpy array
    if isinstance(img_rgb, Image.Image):
        img_arr = np.array(img_rgb)
    else:
        img_arr = img_rgb
    h, w, _ = img_arr.shape
    x1, y1, x2, y2 = box
    bw = x2 - x1; bh = y2 - y1
    pad_x = int(bw * expand); pad_y = int(bh * expand)
    x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x); y2 = min(h, y2 + pad_y)
    crop = img_arr[y1:y2, x1:x2]
    return Image.fromarray(crop)

def image_to_embedding(pil_img):
    img_t = facenet_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = facenet(img_t)
    emb = emb.cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
    emb = emb / norm
    return emb[0]

def cosine_similarity(a, b):
    if b.size == 0:
        return np.array([])
    a = a.reshape(1, -1)
    num = np.dot(b, a.T).squeeze(1)
    den = (np.linalg.norm(b, axis=1) * np.linalg.norm(a)) + 1e-10
    return num / den

def save_student_embeddings(name, embeddings):
    fname = ENC_DIR / f"{secure_filename(name)}.npy"
    np.save(str(fname), embeddings.astype(np.float32))
    idx = {}
    if INDEX_FILE.exists():
        try:
            idx = json.loads(INDEX_FILE.read_text())
        except:
            idx = {}
    idx[name] = str(fname.name)
    INDEX_FILE.write_text(json.dumps(idx))

def load_all_embeddings():
    if not INDEX_FILE.exists():
        return [], np.zeros((0, 512), dtype=np.float32)
    idx = json.loads(INDEX_FILE.read_text())
    names = []
    vecs = []
    for name, fname in idx.items():
        path = ENC_DIR / fname
        if path.exists():
            arr = np.load(str(path))
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            names.extend([name] * arr.shape[0])
            vecs.append(arr)
    if vecs:
        vecs = np.vstack(vecs)
    else:
        vecs = np.zeros((0, 512), dtype=np.float32)
    return names, vecs

# ==================== AUTHENTICATION ROUTES ====================

@app.route("/", methods=["GET", "POST"])
def login():
    # If already logged in, redirect to index
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Username and password required', 'error')
            return render_template('login.html')
        
        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['user_role'] = user['role']
            session['full_name'] = user['full_name'] or user['username']
            session['department'] = user['department'] or ''
            
            flash(f'Welcome back, {session["full_name"]}!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
            return render_template('login.html')
    
    return render_template('login.html')

@app.route("/logout")
def logout():
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))

# ==================== PROTECTED ROUTES ====================

@app.route("/index")
@login_required
def index():
    student_count = sum(1 for _ in DATASET_DIR.iterdir() if _.is_dir()) if DATASET_DIR.exists() else 0
    attendance_count = 0
    if ATT_CSV.exists():
        try:
            with open(ATT_CSV, newline="") as f:
                attendance_count = sum(1 for _ in csv.reader(f)) - 1
                if attendance_count < 0:
                    attendance_count = 0
        except:
            attendance_count = 0
    uploads_count = sum(1 for _ in UPLOADS.glob("*.jpg"))
    last = sorted(UPLOADS.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    latest = last[0].name if last else None
    enc_exists = INDEX_FILE.exists()

    subjects = load_subjects()
    teachers = list(load_teachers().keys())
    top_streaks = compute_streaks() if 'compute_streaks' in globals() else []

    return render_template("index.html",
                           student_count=student_count,
                           attendance_count=attendance_count,
                           uploads_count=uploads_count,
                           latest=latest,
                           enc_exists=enc_exists,
                           subjects=subjects,
                           teachers=teachers,
                           top_streaks=top_streaks)

@app.route("/capture")
@login_required
@role_required('admin', 'teacher')
def capture_page():
    subjects = load_subjects()
    teachers = list(load_teachers().keys())
    return render_template("capture.html", subjects=subjects, teachers=teachers)

@app.route("/students", methods=["GET", "POST"])
@login_required
@role_required('admin')
def students():
    if request.method == "POST":
        name = request.form.get("student_name", "").strip()
        if not name:
            if is_ajax():
                return jsonify(success=False, error="student_name required"), 400
            flash("Student name required", "error")
            return redirect(url_for("students"))
        student_dir = DATASET_DIR / secure_filename(name)
        student_dir.mkdir(exist_ok=True)
        saved = 0
        files = request.files.getlist("images") or [request.files.get("file")]
        for f in files:
            if not f:
                continue
            fname = secure_filename(f.filename)
            if fname == "":
                fname = f"{name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            if allowed_file(fname):
                dest = student_dir / fname
                if dest.suffix == "":
                    dest = dest.with_suffix(".jpg")
                f.save(dest)
                saved += 1
        if is_ajax():
            return jsonify(success=True, saved=saved, student=name)
        flash(f"Saved {saved} image(s) for {name}", "success")
        return redirect(url_for("students"))
    students = []
    if DATASET_DIR.exists():
        for p in sorted(DATASET_DIR.iterdir()):
            if p.is_dir():
                count = sum(1 for _ in p.glob("*") if _.is_file())
                students.append({"name": p.name, "count": count})
    return render_template("students.html", students=students)

@app.route("/students/delete/<name>", methods=["POST"])
@login_required
@role_required('admin')
def delete_student(name):
    folder = DATASET_DIR / name
    if folder.exists() and folder.is_dir():
        for f in folder.glob("*"):
            try:
                f.unlink()
            except:
                pass
        try:
            folder.rmdir()
        except:
            pass
        enc = ENC_DIR / f"{secure_filename(name)}.npy"
        if enc.exists():
            try:
                enc.unlink()
            except:
                pass
            if INDEX_FILE.exists():
                try:
                    idx = json.loads(INDEX_FILE.read_text())
                    if name in idx:
                        idx.pop(name)
                    INDEX_FILE.write_text(json.dumps(idx))
                except:
                    pass
        flash(f"Deleted {name}", "success")
    else:
        flash("Student not found", "error")
    return redirect(url_for("students"))

@app.route("/subjects", methods=["GET", "POST"])
@login_required
@role_required('admin')
def subjects_page():
    if request.method == "POST":
        action = request.form.get("action")
        name = (request.form.get("subject_name") or "").strip()
        subjects = load_subjects()
        if action == "add" and name:
            if name in subjects:
                flash("Subject already exists.", "error")
            else:
                subjects.append(name)
                save_subjects(subjects)
                flash(f"Added subject {name}.", "success")
        return redirect(url_for("subjects_page"))
    subjects = load_subjects()
    return render_template("subjects.html", subjects=subjects)

@app.route("/subjects/delete/<name>", methods=["POST"])
@login_required
@role_required('admin')
def delete_subject(name):
    subjects = load_subjects()
    if name in subjects:
        subjects.remove(name)
        if not subjects:
            subjects = ["General"]
        save_subjects(subjects)
        flash(f"Deleted subject {name}", "success")
    else:
        flash("Subject not found", "error")
    return redirect(url_for("subjects_page"))

@app.route("/teachers", methods=["GET", "POST"])
@login_required
@role_required('admin')
def teachers_page():
    if request.method == "POST":
        action = request.form.get("action")
        tname = (request.form.get("teacher_name") or "").strip()
        subj = (request.form.get("teacher_subject") or "").strip()
        teachers = load_teachers()
        if action == "add" and tname:
            if tname in teachers:
                flash("Teacher exists", "error")
            else:
                teachers[tname] = [subj] if subj else []
                save_teachers(teachers)
                flash(f"Added teacher {tname}", "success")
        elif action == "assign" and tname and subj:
            if tname not in teachers:
                teachers[tname] = []
            if subj not in teachers[tname]:
                teachers[tname].append(subj)
                save_teachers(teachers)
                flash(f"Assigned {subj} to {tname}", "success")
        return redirect(url_for("teachers_page"))
    teachers = load_teachers()
    subjects = load_subjects()
    return render_template("teachers.html", teachers=teachers, subjects=subjects)

@app.route("/teachers/delete/<name>", methods=["POST"])
@login_required
@role_required('admin')
def delete_teacher(name):
    teachers = load_teachers()
    if name in teachers:
        teachers.pop(name)
        save_teachers(teachers)
        flash(f"Deleted {name}", "success")
    else:
        flash("Teacher not found", "error")
    return redirect(url_for("teachers_page"))

@app.route("/encode", methods=["POST"])
@login_required
@role_required('admin')
def encode():
    all_students = [p for p in sorted(DATASET_DIR.iterdir()) if p.is_dir()] if DATASET_DIR.exists() else []
    if not all_students:
        flash("No students found in dataset. Add student images first.", "error")
        return redirect(url_for("index"))
    summary = {}
    for person in all_students:
        name = person.name
        emb_list = []
        for img_path in sorted(person.glob("*")):
            if not allowed_file(img_path.name):
                continue
            try:
                pil = Image.open(str(img_path)).convert("RGB")
                img_rgb = np.array(pil)
                boxes = mediapipe_detect_boxes(img_rgb)
                if not boxes:
                    continue
                boxes_sorted = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
                face_pil = crop_face_from_box(img_rgb, boxes_sorted[0])
                emb = image_to_embedding(face_pil)
                emb_list.append(emb)
            except Exception:
                current_app.logger.exception("encode error")
                continue
        if emb_list:
            arr = np.vstack(emb_list)
            save_student_embeddings(name, arr)
            summary[name] = arr.shape[0]
    if not summary:
        flash("No embeddings created. Make sure dataset images contain detectable faces.", "error")
        return redirect(url_for("index"))
    flash(f"Encoded {len(summary)} students.", "success")
    return redirect(url_for("index"))

@app.route("/encode_status")
@login_required
def encode_status():
    """
    Simple encode status endpoint for frontend to poll.
    We estimate progress as (encoded students / total students) * 100.
    """
    total_students = 0
    encoded_count = 0

    if DATASET_DIR.exists():
        total_students = sum(1 for _ in DATASET_DIR.iterdir() if _.is_dir())

    if INDEX_FILE.exists():
        try:
            idx = json.loads(INDEX_FILE.read_text())
            encoded_count = len(idx)
        except Exception:
            encoded_count = 0

    if total_students == 0:
        return jsonify(message="No students", progress=0)

    progress = int((encoded_count / total_students) * 100) if total_students > 0 else 0
    message = "Encoded" if encoded_count >= total_students and total_students > 0 else "Encoding"
    return jsonify(message=message, progress=progress, encoded=encoded_count, total=total_students)

@app.route("/upload_photo", methods=["POST"])
@login_required
def upload_photo():
    # single file upload (explicit file field)
    if request.files and request.files.get("file"):
        f = request.files["file"]
        if allowed_file(f.filename):
            fname = secure_filename(f.filename)
            path = UPLOADS / fname
            f.save(path)
            if is_ajax():
                return jsonify(success=True, filename=fname)
            flash("Uploaded photo saved.", "success")
            return redirect(url_for("index"))
        else:
            if is_ajax():
                return jsonify(success=False, error="Invalid file type"), 400
            flash("Invalid file type.", "error")
            return redirect(url_for("capture_page"))

    # multiple images for student upload (from students form)
    if request.files and request.files.getlist("images"):
        saved = 0
        name = request.form.get("student_name", "unknown")
        student_dir = DATASET_DIR / secure_filename(name)
        student_dir.mkdir(exist_ok=True)
        for f in request.files.getlist("images"):
            if not f:
                continue
            fname = secure_filename(f.filename) or f"{name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            if allowed_file(fname):
                dest = student_dir / fname
                if dest.suffix == "":
                    dest = dest.with_suffix(".jpg")
                f.save(dest)
                saved += 1
        if is_ajax():
            return jsonify(success=True, saved=saved, student=name)
        flash(f"Saved {saved} image(s) for {name}", "success")
        return redirect(url_for("students"))

    # -------------------------
    # handle dataURL / base64 uploads (from preview / webcam)
    # Accepts:
    #  - FormData with field 'image' or 'imageData' (text data-url)
    #  - application/x-www-form-urlencoded with imageData
    #  - JSON payload with image / imageData
    # -------------------------
    data_url = None
    if request.form:
        data_url = request.form.get("imageData") or request.form.get("image")

    if not data_url and request.files and request.files.get("image"):
        try:
            data_url = request.files.get("image").read().decode("utf-8")
        except Exception:
            data_url = None

    if not data_url and request.is_json:
        js = request.get_json(silent=True) or {}
        data_url = js.get("imageData") or js.get("image")

    if data_url:
        try:
            if "," in data_url and data_url.startswith("data:"):
                header, encoded = data_url.split(",", 1)
            else:
                encoded = data_url
            data = base64.b64decode(encoded)
            img = Image.open(BytesIO(data)).convert("RGB")
            fname = f"class_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            path = UPLOADS / fname
            img.save(path, format="JPEG", quality=90)
            if is_ajax():
                return jsonify(success=True, filename=fname)
            flash("Captured photo saved.", "success")
            return redirect(url_for("index"))
        except Exception as e:
            current_app.logger.exception("upload_photo error")
            if is_ajax():
                return jsonify(success=False, error=str(e)), 400
            flash("Failed to save image.", "error")
            return redirect(url_for("capture_page"))

    if is_ajax():
        return jsonify(success=False, error="no image provided"), 400
    flash("No image provided.", "error")
    return redirect(url_for("capture_page"))

@app.route("/detect_frame", methods=["POST"])
@login_required
def detect_frame():
    data_url = request.form.get("imageData") or (request.get_json(silent=True) or {}).get("imageData")
    if not data_url:
        return jsonify(error="no image"), 400
    try:
        if "," in data_url:
            _, encoded = data_url.split(",", 1)
        else:
            encoded = data_url
        raw = base64.b64decode(encoded)
        img = Image.open(BytesIO(raw)).convert("RGB")
        img_rgb = np.array(img)
        boxes = mediapipe_detect_boxes(img_rgb)
        return jsonify(success=True, boxes=boxes, count=len(boxes))
    except Exception:
        current_app.logger.exception("detect_frame error")
        return jsonify(error="failed to detect"), 500

@app.route("/recognize_ajax", methods=["POST"])
@login_required
@role_required('admin', 'teacher')
def recognize_ajax():
    photos = sorted(UPLOADS.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not photos:
        return jsonify(error="No class photo found."), 400
    photo_path = photos[0]
    if not INDEX_FILE.exists():
        return jsonify(error="Encodings not found. Run Encode first."), 400

    names_db, vecs_db = load_all_embeddings()
    if vecs_db.size == 0:
        return jsonify(error="No known embeddings. Run Encode."), 400

    img_bgr = cv2.imread(str(photo_path))
    if img_bgr is None:
        return jsonify(error="Failed to read photo."), 400
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes = mediapipe_detect_boxes(img_rgb)
    annotated = img_bgr.copy()
    detections = []
    marked = 0

    try:
        if request.is_json:
            T = float(request.json.get("threshold", 0.55))
        else:
            T = float(request.form.get("threshold", 0.55))
    except:
        T = 0.55

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    faces_saved = 0
    low_conf = []

    for box in boxes:
        try:
            face_pil = crop_face_from_box(img_rgb, box)
            emb = image_to_embedding(face_pil)
            sims = cosine_similarity(emb, vecs_db)
            if sims.size == 0:
                name = "Unknown"; conf = 0.0
            else:
                best_idx = int(np.argmax(sims)); best_sim = float(sims[best_idx])
                name = names_db[best_idx] if best_sim >= T else "Unknown"
                conf = best_sim if best_sim >= 0 else 0.0

            x1, y1, x2, y2 = box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (16, 185, 129), 2)
            label = f"{name} {conf:.2f}"
            cv2.putText(annotated, label, (x1, max(12, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (16, 185, 129), 1)

            # save face thumbnail
            face_dir = FACES_DIR / session_id / (secure_filename(name) or "unknown")
            face_dir.mkdir(parents=True, exist_ok=True)
            thumb_name = f"{secure_filename(name)}_{datetime.now().strftime('%H%M%S')}.jpg"
            face_pil.save(str(face_dir / thumb_name), format="JPEG", quality=80)
            faces_saved += 1

            detections.append({"name": name, "confidence": float(conf), "thumb": f"faces/{session_id}/{secure_filename(name)}/{thumb_name}"})
            if name != "Unknown":
                marked += 1

            # low confidence detection (for reverify)
            is_low = float(conf) < REVERIFY_THRESH
            if is_low:
                low_conf.append({"thumb": f"faces/{session_id}/{secure_filename(name)}/{thumb_name}", "suggested_name": name, "confidence": float(conf)})
        except Exception:
            current_app.logger.exception("recognize_ajax face error")
            continue

    annotated_path = UPLOADS / f"annotated_{photo_path.name}"
    cv2.imwrite(str(annotated_path), annotated)

    retval, buf = cv2.imencode('.jpg', annotated)
    if not retval:
        return jsonify(error="Failed to encode annotated image."), 500
    annotated_b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    annotated_dataurl = "data:image/jpeg;base64," + annotated_b64

    # timeline
    try:
        thumb_rel = _save_thumbnail_for_photo(photo_path)
        _append_timeline_entry(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), photo_path.name, marked, len(boxes))
    except Exception:
        current_app.logger.exception("timeline")

    return jsonify(success=True, annotated_b64=annotated_dataurl, detections=detections, marked=marked, faces_found=len(boxes), faces_saved=faces_saved, low_conf=low_conf)

@app.route("/recognize", methods=["POST"])
@login_required
@role_required('admin', 'teacher')
def recognize_and_mark():
    photos = sorted(UPLOADS.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not photos:
        flash("No class photo found.", "error")
        return redirect(url_for("index"))
    photo_path = photos[0]
    if not INDEX_FILE.exists():
        flash("Encodings not found. Run Encode first.", "error")
        return redirect(url_for("index"))

    names_db, vecs_db = load_all_embeddings()
    if vecs_db.size == 0:
        flash("No known embeddings. Run Encode.", "error")
        return redirect(url_for("index"))

    img_bgr = cv2.imread(str(photo_path))
    if img_bgr is None:
        flash("Failed to read photo.", "error")
        return redirect(url_for("index"))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes = mediapipe_detect_boxes(img_rgb)
    annotated = img_bgr.copy()
    marked = 0
    try:
        T = float(request.form.get("threshold", 0.55))
    except:
        T = 0.55

    subject = request.form.get("subject", "General") or "General"
    teacher = request.form.get("teacher", "") or ""
    _ensure_att_csv()
    session_id = datetime.now().strftime("%Y-%m-%d")
    
    # Get user mapping for better name matching
    conn = get_db()
    users = conn.execute("SELECT username, full_name FROM users WHERE role = 'student'").fetchall()
    conn.close()
    
    # Create a mapping dictionary
    user_mapping = {}
    for user in users:
        if user['full_name']:
            user_mapping[user['full_name'].lower()] = user['full_name']
        user_mapping[user['username'].lower()] = user['username']
    
    for box in boxes:
        try:
            face_pil = crop_face_from_box(img_rgb, box)
            emb = image_to_embedding(face_pil)
            sims = cosine_similarity(emb, vecs_db)
            if sims.size == 0:
                name = "Unknown"; conf = 0.0
            else:
                best_idx = int(np.argmax(sims)); best_sim = float(sims[best_idx])
                name = names_db[best_idx] if best_sim >= T else "Unknown"
                conf = best_sim if best_sim >= 0 else 0.0

            x1, y1, x2, y2 = box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (16, 185, 129), 2)
            cv2.putText(annotated, f"{name} {conf:.2f}", (x1, max(12, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (16, 185, 129), 1)

            face_dir = FACES_DIR / session_id / (secure_filename(name) or "unknown")
            face_dir.mkdir(parents=True, exist_ok=True)
            thumb_name = f"{secure_filename(name)}_{datetime.now().strftime('%H%M%S')}.jpg"
            face_pil.save(str(face_dir / thumb_name), format="JPEG", quality=80)

            if name != "Unknown":
                marked += 1
                
                # Try to map the recognized name to a user's full name
                display_name = name
                for user_full_name in user_mapping.values():
                    if name.lower() in user_full_name.lower() or user_full_name.lower() in name.lower():
                        display_name = user_full_name
                        break
                
                with open(ATT_CSV, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([display_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"{conf:.3f}", session_id, photo_path.name, subject, teacher])
        except Exception:
            current_app.logger.exception("recognize_and_mark error")
            continue

    annotated_path = UPLOADS / f"annotated_{photo_path.name}"
    cv2.imwrite(str(annotated_path), annotated)

    # timeline
    try:
        thumb_rel = _save_thumbnail_for_photo(photo_path)
        _append_timeline_entry(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), photo_path.name, marked, len(boxes), subject=subject, teacher=teacher)
    except Exception:
        current_app.logger.exception("timeline")

    flash(f"Recognition done. Found {len(boxes)} faces; marked {marked} present for subject {subject}.", "success")
    return redirect(url_for("attendance"))

import calendar

def compute_period_summary(days, subject_filter=None, teacher_filter=None):
    rows = _read_attendance_rows()
    today = datetime.now().date()
    start_date = today - timedelta(days=days - 1)
    dates_list = [(start_date + timedelta(days=i)) for i in range(days)]
    dates_sorted = [d.strftime("%Y-%m-%d") for d in dates_list]
    week_days = [(d.strftime("%Y-%m-%d"), calendar.day_abbr[d.weekday()]) for d in dates_list]

    attendance_map = {}
    for name, ts, conf, sess, img, subj, teacher in rows:
        if subject_filter and subject_filter != "All" and subj != subject_filter:
            continue
        if teacher_filter and teacher_filter != "All" and teacher != teacher_filter:
            continue
        date_s = _date_str_from_ts(ts)
        if not date_s:
            continue
        if date_s < dates_sorted[0] or date_s > dates_sorted[-1]:
            continue
        attendance_map.setdefault(name, set()).add(date_s)

    student_attendance = {student: set() for student in _get_all_students()}
    for name, dates_attended in attendance_map.items():
        student_attendance.setdefault(name, set()).update(dates_attended)

    total_sessions = len(dates_sorted)
    summary_rows = []
    for student in sorted(student_attendance.keys(), key=lambda x: x.lower()):
        dates_attended = student_attendance.get(student, set())
        presents = sum(1 for d in dates_sorted if d in dates_attended)
        percent = (presents / total_sessions * 100.0) if total_sessions > 0 else 0.0
        per_date = {d: ("P" if d in dates_attended else "A") for d in dates_sorted}
        summary_rows.append({
            "name": student,
            "presents": presents,
            "total_sessions": total_sessions,
            "percent": round(percent, 2),
            "per_date": per_date
        })

    defaulters = [dict(name=r["name"], percent=r["percent"], presents=r["presents"], total=r["total_sessions"]) for r in summary_rows if r["percent"] < 75.0]

    inconsistent = []
    for r in summary_rows:
        if r["presents"] > 0 and r["presents"] < r["total_sessions"]:
            inconsistent.append(dict(name=r["name"], presents=r["presents"], total=r["total_sessions"], percent=r["percent"]))
    return week_days, summary_rows, defaulters, inconsistent

def compute_streaks():
    rows = _read_attendance_rows()
    dates_by_name = {}
    for name, ts, conf, sess, img, subj, teacher in rows:
        ds = _date_str_from_ts(ts)
        if not ds: continue
        dates_by_name.setdefault(name, set()).add(ds)
    today = datetime.now().date()
    streaks = []
    for name, dates in dates_by_name.items():
        streak = 0
        day = today
        while True:
            if day.strftime("%Y-%m-%d") in dates:
                streak += 1
                day = day - timedelta(days=1)
            else:
                break
        streaks.append((name, streak))
    streaks.sort(key=lambda x: x[1], reverse=True)
    top5 = [{'name': s[0], 'streak': s[1]} for s in streaks[:5]]
    return top5

def todays_present_set(subject_filter=None, teacher_filter=None):
    rows = _read_attendance_rows()
    today = datetime.now().strftime("%Y-%m-%d")
    present = set()
    for name, ts, conf, sess, img, subj, teacher in rows:
        date_s = _date_str_from_ts(ts)
        if date_s != today:
            continue
        if subject_filter and subject_filter != "All" and subj != subject_filter:
            continue
        if teacher_filter and teacher_filter != "All" and teacher != teacher_filter:
            continue
        present.add(name)
    return present

@app.route("/attendance")
@login_required
def attendance():
    selected_subject = request.args.get("subject", "All")
    selected_teacher = request.args.get("teacher", "All")
    subjects = ["All"] + load_subjects()
    teachers = ["All"] + list(load_teachers().keys())

    raw_rows = _read_attendance_rows()
    rows = []
    
    # If student, only show their own attendance
    if session.get('user_role') == 'student':
        # Get the student's actual name from the database
        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE id = ?", (session['user_id'],)
        ).fetchone()
        conn.close()
        
        # Try to match the student's full name or username
        student_name = user['full_name'] or user['username']
        
        # Also check if there's a dataset folder with their name
        all_students = _get_all_students()
        matched_dataset_name = None
        
        # Try exact match first
        for ds_name in all_students:
            if ds_name.lower() == student_name.lower():
                matched_dataset_name = ds_name
                break
        
        # Try partial match
        if not matched_dataset_name:
            for ds_name in all_students:
                if student_name.lower() in ds_name.lower() or ds_name.lower() in student_name.lower():
                    matched_dataset_name = ds_name
                    break
        
        # Use the matched name or the student's full name
        search_names = []
        if matched_dataset_name:
            search_names.append(matched_dataset_name)
        search_names.append(student_name)
        
        # Filter rows
        for name, ts, conf, sess, img, subj, teacher in raw_rows:
            if any(search_name.lower() in name.lower() or name.lower() in search_name.lower() for search_name in search_names):
                try:
                    conf_f = float(conf)
                except:
                    conf_f = 0.0
                rows.append([name, ts, conf_f, sess, img, subj, teacher])
    else:
        # Admin/teacher see all
        for name, ts, conf, sess, img, subj, teacher in raw_rows:
            try:
                conf_f = float(conf)
            except:
                conf_f = 0.0
            rows.append([name, ts, conf_f, sess, img, subj, teacher])

    photos = sorted(UPLOADS.glob("annotated_*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    annotated = photos[0].name if photos else None

    # For students, don't show the filter dropdowns
    if session.get('user_role') == 'student':
        subjects = ["All"]
        teachers = ["All"]

    week_dates, week_summary, week_defaulters, week_inconsistent = compute_period_summary(
        7,
        subject_filter=(None if selected_subject == "All" else selected_subject),
        teacher_filter=(None if selected_teacher == "All" else selected_teacher)
    )
    month_dates, month_summary, month_defaulters, month_inconsistent = compute_period_summary(
        30,
        subject_filter=(None if selected_subject == "All" else selected_subject),
        teacher_filter=(None if selected_teacher == "All" else selected_teacher)
    )

    overall_week_dates, overall_week_summary, overall_week_defaulters, overall_week_inconsistent = compute_period_summary(7, subject_filter=None, teacher_filter=None)
    overall_month_dates, overall_month_summary, overall_month_defaulters, overall_month_inconsistent = compute_period_summary(30, subject_filter=None, teacher_filter=None)

    thumbnails_map = {}
    if FACES_DIR.exists():
        sessions = sorted([p for p in FACES_DIR.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        for s in sessions:
            for student_folder in s.iterdir():
                if student_folder.is_dir():
                    files = sorted(student_folder.glob("*"), key=lambda f: f.stat().st_mtime, reverse=True)
                    if files:
                        rel = f"faces/{s.name}/{student_folder.name}/{files[0].name}"
                        if student_folder.name not in thumbnails_map:
                            thumbnails_map[student_folder.name] = rel

    return render_template(
        "attendance.html",
        rows=rows,
        annotated=annotated,
        subjects=subjects,
        teachers=teachers,
        selected_subject=selected_subject,
        selected_teacher=selected_teacher,
        week_dates=week_dates,
        week_summary=week_summary,
        week_defaulters=week_defaulters,
        week_inconsistent=week_inconsistent,
        month_summary=month_summary,
        month_defaulters=month_defaulters,
        month_inconsistent=month_inconsistent,
        overall_week_defaulters=overall_week_defaulters,
        overall_month_defaulters=overall_month_defaulters,
        thumbnails_map=thumbnails_map
    )

@app.route("/download_weekly_sheet")
@login_required
@role_required('admin', 'teacher')
def download_weekly_sheet():
    subject = request.args.get("subject", "All")
    teacher = request.args.get("teacher", "All")
    subj_filter = None if subject == "All" else subject
    t_filter = None if teacher == "All" else teacher
    week_days, summary_rows, _, _ = compute_period_summary(7, subject_filter=subj_filter, teacher_filter=t_filter)

    headers = ["name"] + [f"{wd[1]} ({wd[0]})" for wd in week_days] + ["Presents", "Total", "Percent"]
    if subj_filter:
        headers.append("Subject")
    if t_filter:
        headers.append("Teacher")
    lines = [",".join(headers)]
    for r in summary_rows:
        row_cells = [r["name"]] + [r["per_date"].get(d, "A") for d, _ in week_days] + [str(r["presents"]), str(r["total_sessions"]), f"{r['percent']:.2f}"]
        if subj_filter:
            row_cells.append(subj_filter)
        if t_filter:
            row_cells.append(t_filter)
        safe_row = []
        for c in row_cells:
            if isinstance(c, str) and "," in c:
                safe_row.append(f'"{c}"')
            else:
                safe_row.append(str(c))
        lines.append(",".join(safe_row))
    csv_bytes = "\n".join(lines).encode("utf-8")
    safe_subject = secure_filename(subject) if subject else "All"
    safe_teacher = secure_filename(teacher) if teacher else "All"
    fname = f"weekly_attendance_{safe_subject}_{safe_teacher}_{datetime.now().strftime('%Y%m%d')}.csv"
    return send_file(BytesIO(csv_bytes), mimetype="text/csv", download_name=fname, as_attachment=True)

@app.route("/download_image/<path:name>")
@login_required
def download_image(name):
    # name can be "thumbs/thumb_xxx.jpg" or "annotated_xxx.jpg" or normal upload
    path = UPLOADS / name
    if not path.exists():
        flash("Image not found.", "error")
        return redirect(url_for("attendance"))
    return send_file(str(path), mimetype="image/jpeg")

# Reverify endpoint
@app.route("/reverify", methods=["POST"])
@login_required
def reverify():
    """Accepts a dataURL (image) and a target name to attempt re-embedding and mark present if confident."""
    js = request.get_json(silent=True) or {}
    data_url = js.get("image") or js.get("imageData")
    target = js.get("target") or None
    subject = js.get("subject") or "General"
    teacher = js.get("teacher") or ""
    if not data_url or not target:
        return jsonify(success=False, error="missing"), 400
    try:
        if "," in data_url and data_url.startswith("data:"):
            _, encoded = data_url.split(",", 1)
        else:
            encoded = data_url
        raw = base64.b64decode(encoded)
        img = Image.open(BytesIO(raw)).convert("RGB")
        boxes = mediapipe_detect_boxes(np.array(img))
        if not boxes:
            return jsonify(success=False, error="no face detected"), 400
        face_pil = crop_face_from_box(np.array(img), boxes[0])
        emb = image_to_embedding(face_pil)
        names_db, vecs_db = load_all_embeddings()
        sims = cosine_similarity(emb, vecs_db)
        if sims.size == 0:
            return jsonify(success=False, error="no known embeddings"), 400
        best_idx = int(np.argmax(sims)); best_sim = float(sims[best_idx])
        recognized = names_db[best_idx] if best_sim >= REVERIFY_THRESH else "Unknown"
        session_id = datetime.now().strftime("%Y-%m-%d")
        face_dir = FACES_DIR / session_id / secure_filename(recognized)
        face_dir.mkdir(parents=True, exist_ok=True)
        thumb_name = f"reverify_{secure_filename(target)}_{datetime.now().strftime('%H%M%S')}.jpg"
        face_pil.save(str(face_dir / thumb_name), format="JPEG", quality=80)
        if recognized == target and best_sim >= REVERIFY_THRESH:
            _ensure_att_csv()
            with open(ATT_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([recognized, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"{best_sim:.3f}", session_id, "", subject, teacher])
            return jsonify(success=True, recognized=recognized, confidence=best_sim, marked=True)
        return jsonify(success=True, recognized=recognized, confidence=best_sim, marked=False)
    except Exception:
        current_app.logger.exception("reverify")
        return jsonify(success=False, error="server"), 500

# Realtime & manual mark endpoints + template
@app.route("/realtime")
@login_required
@role_required('admin', 'teacher')
def realtime():
    students = _get_all_students()
    thumbnails = {}
    if FACES_DIR.exists():
        sessions = sorted([p for p in FACES_DIR.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        for s in sessions:
            for student_folder in s.iterdir():
                if student_folder.is_dir():
                    files = sorted(student_folder.glob("*"), key=lambda f: f.stat().st_mtime, reverse=True)
                    if files and student_folder.name not in thumbnails:
                        thumbnails[student_folder.name] = f"faces/{s.name}/{student_folder.name}/{files[0].name}"
    present = list(todays_present_set())
    return render_template("realtime.html", students=students, present=present, thumbnails=thumbnails)

@app.route("/realtime_status")
@login_required
def realtime_status():
    present = list(todays_present_set())
    return jsonify(success=True, present=present)

@app.route("/attendance/mark_present", methods=["POST"])
@login_required
@role_required('admin', 'teacher')
def mark_present_manual():
    name = request.form.get("name") or (request.get_json(silent=True) or {}).get("name")
    if not name:
        return jsonify(success=False, error="missing name"), 400
    _ensure_att_csv()
    session_id = datetime.now().strftime("%Y-%m-%d")
    with open(ATT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"{1.0:.3f}", session_id, "", "Manual", ""])
    return jsonify(success=True, name=name)

# Notification helpers
def send_email(to_addr, subject, body):
    host = os.environ.get("SMTP_HOST")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    pwd = os.environ.get("SMTP_PASS")
    from_addr = os.environ.get("NOTIFY_FROM", user)
    if not host or not user or not pwd:
        current_app.logger.warning("SMTP not configured")
        return False
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = to_addr
        s = smtplib.SMTP(host, port, timeout=10)
        s.starttls()
        s.login(user, pwd)
        s.sendmail(from_addr, [to_addr], msg.as_string())
        s.quit()
        return True
    except Exception:
        current_app.logger.exception("send_email")
        return False

def send_whatsapp(to_number, message):
    sid = os.environ.get("TWILIO_ACCOUNT_SID")
    tok = os.environ.get("TWILIO_AUTH_TOKEN")
    from_wh = os.environ.get("TWILIO_WHATSAPP_FROM")
    if not sid or not tok or not from_wh or not _have_twilio:
        current_app.logger.warning("Twilio not configured or not installed")
        return False
    try:
        client = TwilioClient(sid, tok)
        client.messages.create(body=message, from_=from_wh, to=to_number)
        return True
    except Exception:
        current_app.logger.exception("send_whatsapp")
        return False

@app.route("/notify_absent", methods=["POST"])
@login_required
def notify_absent():
    # body: name, contact_email, contact_whatsapp (optional)
    js = request.get_json(silent=True) or {}
    name = js.get("name")
    if not name:
        return jsonify(success=False, error="missing"), 400
    to_email = js.get("email")
    to_wh = js.get("whatsapp")  # e.g. 'whatsapp:+91...'
    body = f"Your ward {name} was absent today ({datetime.now().strftime('%Y-%m-%d')})."
    email_ok = send_email(to_email, "Attendance alert", body) if to_email else False
    wh_ok = send_whatsapp(to_wh, body) if to_wh else False
    return jsonify(success=True, email=email_ok, whatsapp=wh_ok)

@app.route("/timeline")
@login_required
def timeline():
    data = []
    if TIMELINE_FILE.exists():
        try:
            data = json.loads(TIMELINE_FILE.read_text())
        except:
            data = []
    return jsonify(success=True, timeline=data)

# User management routes (admin only)
@app.route("/users")
@login_required
@role_required('admin')
def users_page():
    conn = get_db()
    users = conn.execute("SELECT * FROM users ORDER BY role, username").fetchall()
    conn.close()
    return render_template("users.html", users=users)

@app.route("/users/add", methods=["POST"])
@login_required
@role_required('admin')
def add_user():
    username = request.form.get("username")
    password = request.form.get("password")
    role = request.form.get("role")
    full_name = request.form.get("full_name")
    department = request.form.get("department")
    
    if not username or not password or not role:
        flash("Username, password and role are required", "error")
        return redirect(url_for("users_page"))
    
    password_hash = generate_password_hash(password)
    
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, role, full_name, department) VALUES (?, ?, ?, ?, ?)",
            (username, password_hash, role, full_name, department)
        )
        conn.commit()
        flash(f"User {username} added successfully", "success")
    except sqlite3.IntegrityError:
        flash(f"Username {username} already exists", "error")
    finally:
        conn.close()
    
    return redirect(url_for("users_page"))

@app.route("/users/delete/<int:user_id>", methods=["POST"])
@login_required
@role_required('admin')
def delete_user(user_id):
    # Prevent deleting yourself
    if user_id == session.get('user_id'):
        flash("Cannot delete your own account", "error")
        return redirect(url_for("users_page"))
    
    conn = get_db()
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    
    flash("User deleted successfully", "success")
    return redirect(url_for("users_page"))

if __name__ == "__main__":
    # Initialize authentication on startup
    init_auth()
    app.run(debug=True, host="127.0.0.1", port=5000)