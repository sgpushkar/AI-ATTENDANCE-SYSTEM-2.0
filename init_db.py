import sqlite3

DB_NAME = "attendance.db"

conn = sqlite3.connect(DB_NAME)
cur = conn.cursor()

# ---------------- USERS ----------------
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT CHECK(role IN ('admin','teacher','student')) NOT NULL,
    department TEXT,
    full_name TEXT,
    email TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# ---------------- SUBJECTS ----------------
cur.execute("""
CREATE TABLE IF NOT EXISTS subjects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_name TEXT NOT NULL,
    department TEXT NOT NULL,
    teacher_id INTEGER,
    FOREIGN KEY (teacher_id) REFERENCES users(id)
)
""")

# ---------------- STUDENTS ----------------
cur.execute("""
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    roll_no TEXT UNIQUE,
    name TEXT,
    department TEXT
)
""")

# ---------------- ATTENDANCE ----------------
cur.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    subject_id INTEGER,
    date TEXT,
    status TEXT CHECK(status IN ('present','absent')),
    FOREIGN KEY (student_id) REFERENCES students(id),
    FOREIGN KEY (subject_id) REFERENCES subjects(id)
)
""")

conn.commit()
conn.close()

print("✅ Database tables created successfully")
print("ℹ️  Default users will be created automatically on first run:")
print("   - admin / admin123")
print("   - teacher1 / teacher123")
print("   - student1 / student123")