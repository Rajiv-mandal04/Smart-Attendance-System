import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from PIL import Image, ImageEnhance, ImageFilter


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

STUDENT_PATH = BASE_DIR / "data" / "students.csv"
ATTENDANCE_PATH = BASE_DIR / "attendance" / "attendance.xlsx"
TRAINER_PATH = BASE_DIR / "trainer" / "trainer.yml"
DATASET_DIR = BASE_DIR / "dataset"
CASCADE_PATH = (
    BASE_DIR
    / "haarcascade"
    / "haarcascade_frontalface_default.xml"
)


# ============================================================
# CONSTANTS
# ============================================================

IST = ZoneInfo("Asia/Kolkata")

RECOGNITION_THRESHOLD = 70

ATTENDANCE_COOLDOWN_MINUTES = 60

FACE_MIN_SIZE = (60, 60)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

STUDENT_PATH.parent.mkdir(parents=True, exist_ok=True)
ATTENDANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
TRAINER_PATH.parent.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: #0b1120;
        color: #f8fafc;
    }

    /* ---------------- SIDEBAR ---------------- */

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        border-right: 1px solid #1f2937;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 1rem 0.9rem;
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    .side-brand {
        text-align: center;
        padding: 0.4rem 0 1.1rem 0;
        border-bottom: 1px solid #1f2937;
        margin-bottom: 1rem;
    }

    .side-brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 0.5px;
        margin: 0;
    }

    .side-brand-sub {
        font-size: 0.72rem;
        color: #94a3b8;
        letter-spacing: 2px;
        margin-top: 0.25rem;
        font-weight: 500;
    }

    .side-section-label {
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.8px;
        font-weight: 700;
        margin: 0.6rem 0 0.4rem 0.4rem;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 0.25rem;
        display: flex;
        flex-direction: column;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 0.75rem 0.95rem;
        margin: 0.15rem 0;
        transition: all 0.25s ease;
        cursor: pointer;
        display: flex;
        align-items: center;
        width: 100%;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(96, 165, 250, 0.10);
        border-color: rgba(96, 165, 250, 0.35);
        transform: translateX(2px);
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label p {
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px;
        margin: 0;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, rgba(96,165,250,0.20), rgba(167,139,250,0.20));
        border-color: rgba(96, 165, 250, 0.55);
        box-shadow: 0 6px 18px rgba(96,165,250,0.18);
    }

    .side-info {
        background: rgba(59, 130, 246, 0.10);
        border: 1px solid rgba(59, 130, 246, 0.35);
        border-radius: 12px;
        padding: 0.75rem 0.9rem;
        font-size: 0.85rem;
        color: #bfdbfe;
        margin: 0.7rem 0;
        line-height: 1.5;
    }

    .side-rule {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.30);
        border-radius: 12px;
        padding: 0.75rem 0.9rem;
        font-size: 0.85rem;
        color: #fcd34d;
        line-height: 1.5;
        margin: 0.7rem 0;
    }

    .side-clock {
        background: linear-gradient(135deg, rgba(96,165,250,0.14), rgba(167,139,250,0.14));
        border: 1px solid rgba(96,165,250,0.30);
        border-radius: 12px;
        padding: 0.85rem 0.9rem;
        text-align: center;
        margin-top: 0.7rem;
    }

    .side-clock-time {
        font-size: 1.45rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: 1px;
        line-height: 1.1;
    }

    .side-clock-date {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 0.3rem;
        letter-spacing: 0.4px;
    }

    /* ---------------- MAIN AREA ---------------- */

    .main-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .stat-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 20px;
        min-height: 120px;
    }

    .stat-title {
        color: #94a3b8;
        font-size: 14px;
    }

    .stat-value {
        color: #f8fafc;
        font-size: 30px;
        font-weight: 800;
        margin-top: 8px;
    }

    .success-box {
        background: #052e1b;
        border: 1px solid #166534;
        padding: 15px;
        border-radius: 12px;
        color: #bbf7d0;
    }

    .warning-box {
        background: #3b2500;
        border: 1px solid #92400e;
        padding: 15px;
        border-radius: 12px;
        color: #fde68a;
    }

    .info-box {
        background: #0c2948;
        border: 1px solid #1d4ed8;
        padding: 15px;
        border-radius: 12px;
        color: #bfdbfe;
    }

    .face-result {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 20px;
        margin-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UTILITY
# ============================================================

def now_ist():
    """Current Indian Standard Time."""
    return datetime.now(IST).replace(tzinfo=None)


def normalize_column_name(column):
    """Normalize dataframe column names."""
    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


# ============================================================
# STUDENTS
# ============================================================

def load_students():
    """
    Loads students.csv.
    Supports headered TSV, headerless TSV, and comma CSV.
    """

    if not STUDENT_PATH.exists():
        return pd.DataFrame(columns=["rollno", "name", "branch"])

    try:
        df = pd.read_csv(
            STUDENT_PATH,
            sep=None,
            engine="python",
            dtype=str,
        )
    except Exception:
        try:
            df = pd.read_csv(
                STUDENT_PATH,
                sep="\t",
                header=None,
                names=["rollno", "name", "branch"],
                dtype=str,
            )
        except Exception:
            return pd.DataFrame(columns=["rollno", "name", "branch"])

    if df.empty:
        return pd.DataFrame(columns=["rollno", "name", "branch"])

    normalized = {
        col: normalize_column_name(col)
        for col in df.columns
    }

    df = df.rename(columns=normalized)

    expected = {"rollno", "name", "branch"}

    if not expected.issubset(set(df.columns)):
        try:
            df = pd.read_csv(
                STUDENT_PATH,
                sep="\t",
                header=None,
                names=["rollno", "name", "branch"],
                dtype=str,
            )
        except Exception:
            return pd.DataFrame(columns=["rollno", "name", "branch"])

    for col in ["rollno", "name", "branch"]:
        if col not in df.columns:
            df[col] = ""

    df = df[["rollno", "name", "branch"]].copy()
    df = df.fillna("")

    for col in ["rollno", "name", "branch"]:
        df[col] = df[col].astype(str).str.strip()

    df = df[df["rollno"].str.lower() != "rollno"]

    df = df[
        ~(
            (df["rollno"] == "")
            & (df["name"] == "")
            & (df["branch"] == "")
        )
    ]

    return df.reset_index(drop=True)


# ============================================================
# SAVE STUDENT
# ============================================================

def save_student(rollno, name, branch):

    rollno = str(rollno).strip()
    name = str(name).strip()
    branch = str(branch).strip()

    students = load_students()

    new_row = pd.DataFrame(
        [
            {
                "rollno": rollno,
                "name": name,
                "branch": branch,
            }
        ]
    )

    students = pd.concat([students, new_row], ignore_index=True)

    students.to_csv(
        STUDENT_PATH,
        sep="\t",
        index=False,
    )


# ============================================================
# ATTENDANCE
# ============================================================

def load_attendance():

    if not ATTENDANCE_PATH.exists():
        return pd.DataFrame(
            columns=[
                "roll no",
                "name",
                "branch",
                "date",
                "time",
                "status",
            ]
        )

    try:
        df = pd.read_excel(ATTENDANCE_PATH)
    except Exception:
        return pd.DataFrame(
            columns=[
                "roll no",
                "name",
                "branch",
                "date",
                "time",
                "status",
            ]
        )

    if df.empty:
        return pd.DataFrame(
            columns=[
                "roll no",
                "name",
                "branch",
                "date",
                "time",
                "status",
            ]
        )

    mapping = {}

    for col in df.columns:

        normalized = normalize_column_name(col)

        if normalized in ["rollno", "rollnumber", "studentid", "id"]:
            mapping[col] = "roll no"

        elif normalized in ["name", "studentname"]:
            mapping[col] = "name"

        elif normalized in ["branch", "department"]:
            mapping[col] = "branch"

        elif normalized in ["date", "attendancedate"]:
            mapping[col] = "date"

        elif normalized in ["time", "attendancetime"]:
            mapping[col] = "time"

        elif normalized in ["status", "attendance"]:
            mapping[col] = "status"

    df = df.rename(columns=mapping)

    required = ["roll no", "name", "branch", "date", "time", "status"]

    for col in required:
        if col not in df.columns:
            df[col] = ""

    df = df[required].copy()
    df = df.fillna("")

    return df


# ============================================================
# SAVE ATTENDANCE
# ============================================================

def save_attendance_record(rollno, name, branch, timestamp):

    attendance = load_attendance()

    new_row = pd.DataFrame(
        [
            {
                "roll no": str(rollno),
                "name": str(name),
                "branch": str(branch),
                "date": timestamp.strftime("%Y-%m-%d"),
                "time": timestamp.strftime("%H:%M:%S"),
                "status": "Present",
            }
        ]
    )

    attendance = pd.concat([attendance, new_row], ignore_index=True)

    attendance.to_excel(ATTENDANCE_PATH, index=False)


# ============================================================
# CHECK 1 HOUR RULE
# ============================================================

def can_mark_attendance(rollno):

    attendance = load_attendance()

    if attendance.empty:
        return True, None

    records = attendance[
        attendance["roll no"].astype(str).str.strip()
        ==
        str(rollno).strip()
    ]

    if records.empty:
        return True, None

    latest = records.iloc[-1]

    date_value = str(latest["date"]).strip()
    time_value = str(latest["time"]).strip()

    try:
        latest_datetime = datetime.strptime(
            f"{date_value} {time_value}",
            "%Y-%m-%d %H:%M:%S",
        )
    except Exception:
        return True, None

    current_time = now_ist()

    difference = current_time - latest_datetime

    if difference < timedelta(minutes=ATTENDANCE_COOLDOWN_MINUTES):

        remaining = (
            timedelta(minutes=ATTENDANCE_COOLDOWN_MINUTES)
            - difference
        )

        return False, remaining

    return True, None


# ============================================================
# MARK ATTENDANCE
# ============================================================

def mark_attendance(rollno, name, branch):

    current_time = now_ist()

    allowed, remaining = can_mark_attendance(rollno)

    if not allowed:

        minutes = int(remaining.total_seconds() // 60)
        seconds = int(remaining.total_seconds() % 60)

        return (
            False,
            f"Attendance already marked. "
            f"Try again after {minutes}m {seconds}s."
        )

    save_attendance_record(rollno, name, branch, current_time)

    return (
        True,
        f"Attendance marked successfully at "
        f"{current_time.strftime('%I:%M:%S %p')} IST"
    )


# ============================================================
# FACE CASCADE
# ============================================================

@st.cache_resource
def load_face_cascade():

    if not CASCADE_PATH.exists():
        return None

    cascade = cv2.CascadeClassifier(str(CASCADE_PATH))

    if cascade.empty():
        return None

    return cascade


# ============================================================
# LBPH MODEL
# ============================================================

@st.cache_resource
def load_recognizer():

    if not TRAINER_PATH.exists():
        return None

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(str(TRAINER_PATH))
        return recognizer
    except Exception:
        return None


# ============================================================
# FACE DETECTION
# ============================================================

def detect_faces(image):

    cascade = load_face_cascade()

    if cascade is None:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=7,
        minSize=FACE_MIN_SIZE,
    )

    valid_faces = []

    for x, y, w, h in faces:
        area = w * h
        if area >= 4000:
            valid_faces.append((x, y, w, h))

    if not valid_faces:

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(40, 40),
        )

        for x, y, w, h in faces:
            if w * h >= 3000:
                valid_faces.append((x, y, w, h))

    return valid_faces


# ============================================================
# SELECT LARGEST FACE
# ============================================================

def detect_single_face(image):

    faces = detect_faces(image)

    if not faces:
        return None

    largest_face = max(faces, key=lambda item: item[2] * item[3])

    return largest_face


# ============================================================
# AUGMENTATION
# ============================================================

def create_training_samples(face_gray, output_dir, rollno):

    output_dir.mkdir(parents=True, exist_ok=True)

    samples = []

    samples.append(face_gray)
    samples.append(cv2.flip(face_gray, 1))
    samples.append(cv2.GaussianBlur(face_gray, (3, 3), 0))

    brighter = cv2.convertScaleAbs(face_gray, alpha=1.0, beta=15)
    samples.append(brighter)

    darker = cv2.convertScaleAbs(face_gray, alpha=1.0, beta=-15)
    samples.append(darker)

    contrast = cv2.convertScaleAbs(face_gray, alpha=1.15, beta=0)
    samples.append(contrast)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(face_gray, -1, kernel)
    samples.append(sharpened)

    start_index = len(list(output_dir.glob("*.jpg"))) + 1

    for index, sample in enumerate(samples, start=start_index):
        filename = f"User.{rollno}.{index}.jpg"
        filepath = output_dir / filename
        cv2.imwrite(str(filepath), sample)

    return len(samples)


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8,
    )

    faces = []
    ids = []

    if not DATASET_DIR.exists():
        return False, "Dataset folder not found."

    student_folders = [
        folder
        for folder in DATASET_DIR.iterdir()
        if folder.is_dir() and folder.name.isdigit()
    ]

    if not student_folders:
        return False, "No student face dataset found."

    for student_folder in student_folders:

        try:
            student_id = int(student_folder.name)
        except ValueError:
            continue

        image_files = list(student_folder.glob("*.jpg"))
        image_files += list(student_folder.glob("*.jpeg"))
        image_files += list(student_folder.glob("*.png"))

        for image_path in image_files:

            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            image = cv2.resize(image, (200, 200))
            image = cv2.equalizeHist(image)

            faces.append(image)
            ids.append(student_id)

    if not faces:
        return False, "No valid face images found."

    try:
        recognizer.train(faces, np.array(ids))
        recognizer.save(str(TRAINER_PATH))
        load_recognizer.clear()
        return True, f"Model trained successfully using {len(faces)} images."
    except Exception as e:
        return False, f"Training failed: {e}"


# ============================================================
# RECOGNIZE FACE
# ============================================================

def recognize_face(image):

    students = load_students()

    if students.empty:
        return (
            None, None, None, image,
            "No registered students found.",
        )

    recognizer = load_recognizer()

    if recognizer is None:
        return (
            None, None, None, image,
            "Trained model not found. Please register a student first.",
        )

    face = detect_single_face(image)

    if face is None:
        return (None, None, None, image, "No face detected.")

    x, y, w, h = face

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    face_gray = gray[y:y + h, x:x + w]

    if face_gray.size == 0:
        return (None, None, None, image, "Invalid face region.")

    face_gray = cv2.resize(face_gray, (200, 200))

    try:
        predicted_id, confidence = recognizer.predict(face_gray)
    except Exception as e:
        return (None, None, None, image, f"Recognition error: {e}")

    annotated = image.copy()

    # Unknown face
    if confidence > RECOGNITION_THRESHOLD:

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(
            annotated, "Unknown",
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
        )

        return (None, None, confidence, annotated, "Face not recognized.")

    student_rows = students[
        students["rollno"].astype(str).str.strip()
        ==
        str(predicted_id).strip()
    ]

    if student_rows.empty:

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(
            annotated, "Unknown",
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
        )

        return (None, None, confidence, annotated, "Recognized ID is not registered.")

    student = student_rows.iloc[0]

    rollno = str(student["rollno"]).strip()
    name = str(student["name"]).strip()

    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)

    label = f"{name} | Roll: {rollno}"

    cv2.rectangle(
        annotated,
        (x, max(y - 42, 0)),
        (x + w, y),
        (0, 255, 0),
        -1,
    )

    cv2.putText(
        annotated, label,
        (x + 5, max(y - 12, 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2,
    )

    return (rollno, name, confidence, annotated, "Face recognized.")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="side-brand">
            <div class="side-brand-title">🎓 Smart Attendance</div>
            <div class="side-brand-sub">AI · FACE · REALTIME</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="side-section-label">Navigation</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "🏠  Dashboard",
            "📝  Register Student",
            "📷  Mark Attendance",
            "👥  Students",
            "📊  Attendance Records",
        ],
        label_visibility="collapsed",
    )

    page_clean = page.replace("  ", " ").strip()

    st.markdown("---")

    _now = now_ist()

    st.markdown(
        f"""
        <div class="side-clock">
            <div class="side-clock-time">{_now.strftime('%I:%M:%S %p')}</div>
            <div class="side-clock-date">{_now.strftime('%A, %d %b %Y')} · IST</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="side-info">
            📍 Attendance timezone: <b>India (IST)</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="side-rule">
            ⏱️ One student can mark attendance only <b>once per hour</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DASHBOARD
# ============================================================

if page_clean == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">Smart Attendance System</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">Face recognition based student attendance management</div>',
        unsafe_allow_html=True,
    )

    students = load_students()
    attendance = load_attendance()

    total_students = len(students)
    total_records = len(attendance)

    today = now_ist().strftime("%Y-%m-%d")

    today_records = attendance[attendance["date"].astype(str) == today]
    today_count = len(today_records)

    unique_today = (
        today_records["roll no"].astype(str).nunique()
        if not today_records.empty else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-title">Registered Students</div>
                <div class="stat-value">{total_students}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-title">Today's Attendance</div>
                <div class="stat-value">{unique_today}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-title">Total Records</div>
                <div class="stat-value">{total_records}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-title">Current Time</div>
                <div class="stat-value">{now_ist().strftime("%I:%M %p")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 System Status")

        if TRAINER_PATH.exists():
            st.success("✅ Face recognition model available")
        else:
            st.warning("⚠️ Face model not trained yet")

        if STUDENT_PATH.exists():
            st.success("✅ Student database available")
        else:
            st.warning("⚠️ Student database not created")

        if ATTENDANCE_PATH.exists():
            st.success("✅ Attendance database available")
        else:
            st.info("ℹ️ Attendance file will be created automatically")

    with col2:
        st.subheader("🕐 Current Indian Time")

        st.info(now_ist().strftime("%A, %d %B %Y — %I:%M:%S %p IST"))

        st.markdown(
            """
            **Attendance Rule**

            A student can mark attendance only once
            within a 1-hour period.
            """
        )


# ============================================================
# REGISTER STUDENT
# ============================================================

elif page_clean == "📝 Register Student":

    st.markdown(
        '<div class="main-title">Register New Student</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">Capture a face image and create a recognition model</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Student Information")

        rollno = st.text_input("Roll Number", placeholder="Example: 101")
        name = st.text_input("Student Name", placeholder="Example: Rajiv Kumar")
        branch = st.text_input("Branch", placeholder="Example: CSE")

    with col2:
        st.subheader("Capture Face")

        camera_image = st.camera_input("Take a clear photo")

    st.markdown("---")

    register_button = st.button(
        "🚀 Register Student",
        type="primary",
        width="stretch",
    )

    if register_button:

        rollno = rollno.strip()
        name = name.strip()
        branch = branch.strip()

        if not rollno:
            st.error("Please enter Roll Number.")
            st.stop()

        if not rollno.isdigit():
            st.error("Roll Number must contain numbers only.")
            st.stop()

        if not name:
            st.error("Please enter student name.")
            st.stop()

        if not branch:
            st.error("Please enter branch.")
            st.stop()

        if camera_image is None:
            st.error("Please capture your face photo.")
            st.stop()

        students = load_students()

        if (students["rollno"].astype(str).str.strip() == rollno).any():
            st.error(f"Roll Number {rollno} is already registered.")
            st.stop()

        try:
            image_bytes = camera_image.getvalue()
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        except Exception as e:
            st.error(f"Unable to read camera image: {e}")
            st.stop()

        if image is None:
            st.error("Invalid camera image.")
            st.stop()

        face = detect_single_face(image)

        if face is None:
            st.error("❌ No face detected in the captured photo.")
            st.warning(
                "Try again with better lighting and "
                "keep your face closer to the camera."
            )
            st.stop()

        x, y, w, h = face

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        face_gray = gray[y:y + h, x:x + w]

        if face_gray.size == 0:
            st.error("Unable to extract face.")
            st.stop()

        face_gray = cv2.resize(face_gray, (200, 200))

        student_folder = DATASET_DIR / rollno
        student_folder.mkdir(parents=True, exist_ok=True)

        sample_count = create_training_samples(face_gray, student_folder, rollno)

        with st.spinner("Training face recognition model..."):
            trained, message = train_model()

        if not trained:
            st.error(message)
            st.warning(
                "Face images were saved, "
                "but student registration was not completed."
            )
            st.stop()

        try:
            save_student(rollno, name, branch)
        except Exception as e:
            st.error(f"Unable to save student: {e}")
            st.stop()

        st.success(f"🎉 {name} registered successfully!")

        st.info(
            f"Roll No: {rollno} | "
            f"Branch: {branch} | "
            f"Training Images: {sample_count}"
        )

        preview = image.copy()

        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv2.putText(
            preview,
            f"{name} | Roll: {rollno}",
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

        st.image(
            cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
            caption="Registered Face",
            width="stretch",
        )


# ============================================================
# MARK ATTENDANCE
# ============================================================

elif page_clean == "📷 Mark Attendance":

    st.markdown(
        '<div class="main-title">Mark Attendance</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">Capture your face to verify identity and mark attendance</div>',
        unsafe_allow_html=True,
    )

    if not TRAINER_PATH.exists():
        st.warning("⚠️ Face recognition model is not available.")
        st.info("Register at least one student first.")
        st.stop()

    camera_image = st.camera_input("📷 Capture Face")

    if camera_image is not None:

        image_bytes = camera_image.getvalue()
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Unable to process captured image.")
            st.stop()

        with st.spinner("Recognizing face..."):
            (rollno, name, confidence, annotated, message) = recognize_face(image)

        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            caption="Face Recognition Result",
            width="stretch",
        )

        if rollno is None:
            st.error(f"❌ {message}")
        else:
            st.success(f"✅ Recognized: {name} (Roll No: {rollno})")

            if confidence is not None:
                st.caption(f"Recognition distance: {confidence:.2f}")

            students = load_students()

            student_rows = students[
                students["rollno"].astype(str).str.strip()
                ==
                str(rollno).strip()
            ]

            if not student_rows.empty:

                student = student_rows.iloc[0]
                branch = str(student["branch"]).strip()

                marked, attendance_message = mark_attendance(rollno, name, branch)

                if marked:
                    st.success(f"🎉 {attendance_message}")
                    st.info(
                        f"Student: {name}\n\n"
                        f"Roll No: {rollno}\n\n"
                        f"Branch: {branch}\n\n"
                        f"Time: {now_ist().strftime('%I:%M:%S %p')} IST"
                    )
                else:
                    st.warning(f"⏳ {attendance_message}")


# ============================================================
# STUDENTS
# ============================================================

elif page_clean == "👥 Students":

    st.markdown(
        '<div class="main-title">Registered Students</div>',
        unsafe_allow_html=True,
    )

    students = load_students()

    if students.empty:
        st.info("No students registered yet.")
    else:
        display_students = students.rename(
            columns={
                "rollno": "Roll No",
                "name": "Name",
                "branch": "Branch",
            }
        )

        st.dataframe(
            display_students,
            width="stretch",
            hide_index=True,
        )

        st.caption(f"Total registered students: {len(students)}")


# ============================================================
# ATTENDANCE RECORDS
# ============================================================

elif page_clean == "📊 Attendance Records":

    st.markdown(
        '<div class="main-title">Attendance Records</div>',
        unsafe_allow_html=True,
    )

    attendance = load_attendance()

    if attendance.empty:
        st.info("No attendance records available.")
    else:
        display_attendance = attendance.rename(
            columns={
                "roll no": "Roll No",
                "name": "Name",
                "branch": "Branch",
                "date": "Date",
                "time": "Time",
                "status": "Status",
            }
        )

        display_attendance = (
            display_attendance.iloc[::-1]
            .reset_index(drop=True)
        )

        st.dataframe(
            display_attendance,
            width="stretch",
            hide_index=True,
        )

        st.caption(f"Total attendance records: {len(attendance)}")

        today = now_ist().strftime("%Y-%m-%d")

        today_data = attendance[attendance["date"].astype(str) == today]

        st.markdown("---")
        st.subheader("📅 Today's Attendance")

        if today_data.empty:
            st.info("No attendance marked today.")
        else:
            today_display = today_data.rename(
                columns={
                    "roll no": "Roll No",
                    "name": "Name",
                    "branch": "Branch",
                    "date": "Date",
                    "time": "Time",
                    "status": "Status",
                }
            )

            st.dataframe(
                today_display.iloc[::-1],
                width="stretch",
                hide_index=True,
            )
