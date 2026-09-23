import os
import shutil
from datetime import datetime, timedelta

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


# Page configuration
st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Dark professional theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0b1020 0%, #1a1f3a 55%, #0d1224 100%);
        color: #e8ecf3;
    }

    section[data-testid="stSidebar"] {
        background: #0f1428;
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }

    section[data-testid="stSidebar"] * {
        color: #e8ecf3 !important;
    }

    /* Radio as cards */
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        background: transparent;
        border-radius: 12px;
        padding: 0.65rem 0.9rem;
        margin: 0.2rem 0;
        border: 1px solid transparent;
        transition: all 0.25s ease;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(0, 198, 255, 0.08);
        border-color: rgba(0, 198, 255, 0.25);
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > label p {
        font-size: 0.95rem;
        font-weight: 500;
        letter-spacing: 0.3px;
    }

    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: 0.4px;
    }

    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4fc3f7, #7c4dff, #ec407a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        text-align: center;
        color: #8b93a7;
        font-size: 0.9rem;
        margin-bottom: 2rem;
        letter-spacing: 2px;
        font-weight: 500;
    }

    .metric-card {
        background: linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.2rem 1.3rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    }

    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(79,195,247,0.4);
        box-shadow: 0 10px 26px rgba(79,195,247,0.15);
    }

    .metric-label {
        font-size: 0.72rem;
        color: #8b93a7;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        font-weight: 600;
    }

    .metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4fc3f7, #7c4dff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 0.25rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #2979ff, #7c4dff);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
        transition: all 0.25s ease;
        box-shadow: 0 4px 14px rgba(41,121,255,0.35);
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(124,77,255,0.45);
    }

    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stNumberInput > div > div > input {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
    }

    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .brand-box {
        text-align: center;
        padding: 0.5rem 0 1rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 1rem;
    }

    .brand-title {
        font-size: 1.15rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4fc3f7, #7c4dff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .brand-sub {
        font-size: 0.7rem;
        color: #6b7280;
        letter-spacing: 1.5px;
        margin-top: 0.2rem;
    }

    .side-stat {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 0.75rem 0.9rem;
        margin-bottom: 0.6rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .side-stat-label {
        font-size: 0.78rem;
        color: #8b93a7;
        font-weight: 500;
    }

    .side-stat-value {
        font-size: 1rem;
        font-weight: 700;
        color: #4fc3f7;
    }

    .clock-box {
        background: linear-gradient(135deg, rgba(41,121,255,0.12), rgba(124,77,255,0.12));
        border: 1px solid rgba(79,195,247,0.25);
        border-radius: 12px;
        padding: 0.8rem 0.9rem;
        text-align: center;
        margin-top: 0.4rem;
    }

    .clock-time {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 1px;
    }

    .clock-date {
        font-size: 0.72rem;
        color: #8b93a7;
        margin-top: 0.15rem;
        letter-spacing: 0.5px;
    }

    .note {
        font-size: 0.72rem;
        color: #6b7280;
        line-height: 1.5;
        padding: 0.6rem 0.4rem;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin-top: 0.6rem;
    }

    .info-badge {
        display: inline-block;
        background: rgba(79,195,247,0.1);
        color: #4fc3f7;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        border: 1px solid rgba(79,195,247,0.25);
        margin: 0.4rem 0;
    }

    hr {
        border-color: rgba(255,255,255,0.06);
    }
</style>
""", unsafe_allow_html=True)


# Paths and constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ATTEND_DIR = os.path.join(BASE_DIR, "attendance")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")
HAAR_DIR = os.path.join(BASE_DIR, "haarcascade")

STUDENT_PATH = os.path.join(DATA_DIR, "students.csv")
ATTENDANCE_PATH = os.path.join(ATTEND_DIR, "attendance.csv")
TRAINER_PATH = os.path.join(TRAINER_DIR, "trainer.yml")
CASCADE_PATH = os.path.join(HAAR_DIR, "haarcascade_frontalface_default.xml")

STUDENT_COLUMNS = ["RollNo", "Name", "Branch"]
ATTENDANCE_COLUMNS = ["RollNo", "Name", "Branch", "Date", "Time", "Timestamp", "Status"]

CONF_THRESHOLD = 85
NUM_CAPTURE_IMAGES = 60
ONE_HOUR = timedelta(hours=1)


# Time helpers
def now():
    return datetime.now()


def now_ts():
    return now().strftime("%Y-%m-%d %H:%M:%S")


def today_str():
    return now().strftime("%Y-%m-%d")


# Student data
def load_students():
    if not os.path.exists(STUDENT_PATH):
        return pd.DataFrame(columns=STUDENT_COLUMNS)
    try:
        df = pd.read_csv(STUDENT_PATH, dtype={"RollNo": str})
        for c in STUDENT_COLUMNS:
            if c not in df.columns:
                df[c] = ""
        return df[STUDENT_COLUMNS]
    except Exception:
        return pd.DataFrame(columns=STUDENT_COLUMNS)


def save_students(df):
    df.to_csv(STUDENT_PATH, index=False)


def add_student(rollno, name, branch):
    df = load_students()
    if (df["RollNo"].astype(str) == str(rollno)).any():
        return False, "This roll number is already registered."
    new_row = pd.DataFrame([{"RollNo": str(rollno), "Name": name, "Branch": branch}])
    df = pd.concat([df, new_row], ignore_index=True)
    save_students(df)
    return True, "Student saved."


# Attendance data
def load_attendance():
    if not os.path.exists(ATTENDANCE_PATH):
        return pd.DataFrame(columns=ATTENDANCE_COLUMNS)
    try:
        df = pd.read_csv(ATTENDANCE_PATH, dtype={"RollNo": str})
        for c in ATTENDANCE_COLUMNS:
            if c not in df.columns:
                df[c] = ""
        return df[ATTENDANCE_COLUMNS]
    except Exception:
        return pd.DataFrame(columns=ATTENDANCE_COLUMNS)


def save_attendance(df):
    df.to_csv(ATTENDANCE_PATH, index=False)


def build_last_seen_cache():
    df = load_attendance()
    cache = {}
    for _, row in df.iterrows():
        try:
            dt = datetime.strptime(str(row["Timestamp"]), "%Y-%m-%d %H:%M:%S")
        except Exception:
            continue
        r = str(row["RollNo"])
        if r not in cache or dt > cache[r]:
            cache[r] = dt
    return cache


def mark_attendance_row(rollno, name, branch, cache):
    rollno = str(rollno)
    ts = now()
    last = cache.get(rollno)

    if last is not None and (ts - last) < ONE_HOUR:
        remaining = ONE_HOUR - (ts - last)
        mins = int(remaining.total_seconds() // 60)
        secs = int(remaining.total_seconds() % 60)
        return "reverified", f"{name} already marked. Wait {mins}m {secs}s."

    df = load_attendance()
    new_row = {
        "RollNo": rollno, "Name": name, "Branch": branch,
        "Date": ts.strftime("%Y-%m-%d"), "Time": ts.strftime("%H:%M:%S"),
        "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "Status": "Present",
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_attendance(df)
    cache[rollno] = ts
    return "success", f"{name} marked present at {new_row['Time']}."


def get_total_unique_students():
    students_df = load_students()
    attendance_df = load_attendance()
    ids = set()
    if not students_df.empty:
        ids.update(students_df["RollNo"].astype(str).dropna().unique())
    if not attendance_df.empty:
        ids.update(attendance_df["RollNo"].astype(str).dropna().unique())
    ids.discard("")
    ids.discard("nan")
    return len(ids)


def reset_all_data():
    save_students(pd.DataFrame(columns=STUDENT_COLUMNS))
    save_attendance(pd.DataFrame(columns=ATTENDANCE_COLUMNS))
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    os.makedirs(DATASET_DIR, exist_ok=True)
    if os.path.exists(TRAINER_PATH):
        os.remove(TRAINER_PATH)


# Face recognition helpers
def get_cascade():
    path = CASCADE_PATH if os.path.exists(CASCADE_PATH) else (
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return cv2.CascadeClassifier(path)


def open_camera():
    """Try multiple backends - important on Windows."""
    # On Windows, try DSHOW first
    if os.name == "nt":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        return cap
    return cv2.VideoCapture(0)


# Browser-based face capture (uses Streamlit's camera_input)
def process_browser_frame(image, rollno, num_target=NUM_CAPTURE_IMAGES):
    """Takes a PIL image from browser camera, detects face, saves crop."""
    cascade = get_cascade()
    frame = np.array(image.convert("RGB"))
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        return None, "No face detected. Look straight at the camera."

    student_dir = os.path.join(DATASET_DIR, str(rollno))
    os.makedirs(student_dir, exist_ok=True)

    x, y, w, h = faces[0]
    face_img = gray[y:y + h, x:x + w]
    face_img = cv2.resize(face_img, (200, 200))

    existing = len([f for f in os.listdir(student_dir) if f.endswith(".jpg")])
    idx = existing + 1
    save_path = os.path.join(student_dir, f"{idx}.jpg")
    cv2.imwrite(save_path, face_img)

    # Draw box on preview image
    preview = frame.copy()
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return preview, f"Captured image {idx}"


# Desktop OpenCV capture (fallback - opens separate window)
def capture_faces_desktop(rollno, num_images=NUM_CAPTURE_IMAGES):
    cascade = get_cascade()
    student_dir = os.path.join(DATASET_DIR, str(rollno))
    os.makedirs(student_dir, exist_ok=True)

    cam = open_camera()
    if not cam.isOpened():
        return 0, "Could not open the webcam."

    count = 0
    try:
        while count < num_images:
            ok, frame = cam.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.2, 5)

            for (x, y, w, h) in faces:
                count += 1
                face_img = gray[y:y + h, x:x + w]
                cv2.imwrite(os.path.join(student_dir, f"{count}.jpg"), face_img)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Captured {count}/{num_images}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                break

            cv2.imshow("Registering Face - press ESC to stop early", frame)
            if cv2.waitKey(1) & 0xFF == 27 or count >= num_images:
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

    return count, "Done."


def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = [], []

    if not os.path.exists(DATASET_DIR):
        return False, "No dataset found."

    for folder in os.listdir(DATASET_DIR):
        if not folder.isdigit():
            continue
        folder_path = os.path.join(DATASET_DIR, folder)
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".jpg", ".png")):
                img = cv2.imread(os.path.join(folder_path, fname), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (200, 200))
                faces.append(img)
                ids.append(int(folder))

    if not faces:
        return False, "No face images to train on yet."

    recognizer.train(faces, np.array(ids))
    recognizer.write(TRAINER_PATH)
    return True, f"Model trained on {len(faces)} images from {len(set(ids))} student(s)."


# Browser-based recognition (single shot from camera_input)
def recognize_from_browser_frame(image):
    """Detect faces in a browser frame, mark attendance."""
    if not os.path.exists(TRAINER_PATH):
        return [], "No trained model found."

    cascade = get_cascade()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_PATH)

    students = load_students()
    cache = build_last_seen_cache()

    frame = np.array(image.convert("RGB"))
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        return [], "No face detected in the frame."

    results = []
    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (200, 200))
        rollno, conf = recognizer.predict(face_img)

        if conf < CONF_THRESHOLD:
            row = students[students["RollNo"].astype(str) == str(rollno)]
            if not row.empty:
                name = row.iloc[0]["Name"]
                branch = row.iloc[0]["Branch"]
                status, msg = mark_attendance_row(rollno, name, branch, cache)
                results.append((status, msg))
            else:
                results.append(("unknown", "Unknown face - not in database."))
        else:
            results.append(("unknown", f"Unknown face (conf: {conf:.1f})"))

    return results, "Done."


# Desktop recognition (opens cv2 window)
def run_recognition_desktop():
    if not os.path.exists(TRAINER_PATH):
        return "no_model", []

    cascade = get_cascade()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_PATH)

    students = load_students()
    cache = build_last_seen_cache()

    cam = open_camera()
    if not cam.isOpened():
        return "no_camera", []

    session_log = []
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.2, 5)

            for (x, y, w, h) in faces:
                face_img = gray[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, (200, 200))
                rollno, conf = recognizer.predict(face_img)

                if conf < CONF_THRESHOLD:
                    row = students[students["RollNo"].astype(str) == str(rollno)]
                    if not row.empty:
                        name = row.iloc[0]["Name"]
                        branch = row.iloc[0]["Branch"]
                        status, msg = mark_attendance_row(rollno, name, branch, cache)
                        color = (0, 255, 0) if status == "success" else (0, 200, 255)
                        label = f"{name} ({status})"
                        if not session_log or session_log[-1] != msg:
                            session_log.append(msg)
                    else:
                        label, color = "Unknown", (0, 0, 255)
                else:
                    label, color = "Unknown", (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Mark Attendance - press ESC to stop", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

    return "done", session_log


# Sidebar
with st.sidebar:
    st.markdown("""
        <div class="brand-box">
            <div class="brand-title">🎓 Smart Attendance</div>
            <div class="brand-sub">AI · REAL-TIME · PRO</div>
        </div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "Navigation",
        ["🏠  Dashboard", "📝  Register Student", "📸  Mark Attendance",
         "📊  Attendance Records", "👥  Students", "ℹ️  About"],
        label_visibility="collapsed"
    )
    menu_clean = menu.replace("  ", " ").strip()

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    total_students_sidebar = get_total_unique_students()
    total_records_sidebar = len(load_attendance())

    st.markdown(f"""
        <div class="side-stat">
            <span class="side-stat-label">Registered Students</span>
            <span class="side-stat-value">{total_students_sidebar}</span>
        </div>
        <div class="side-stat">
            <span class="side-stat-label">Attendance Records</span>
            <span class="side-stat-value">{total_records_sidebar}</span>
        </div>
    """, unsafe_allow_html=True)

    now_sidebar = now()
    st.markdown(f"""
        <div class="clock-box">
            <div class="clock-time">{now_sidebar.strftime('%I:%M:%S %p')}</div>
            <div class="clock-date">{now_sidebar.strftime('%A, %d %b %Y')}</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="note">
            ⏱️ A student can only get a new Present entry once the previous
            one is more than 1 hour old.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    with st.expander("⚙️ Settings"):
        st.caption("This permanently deletes all students, face images, "
                    "the trained model, and attendance history.")
        confirm = st.checkbox("I understand this cannot be undone")
        if st.button("🗑️ Reset All Data", key="reset_btn", disabled=not confirm):
            reset_all_data()
            st.success("All data cleared. Starting fresh!")
            st.rerun()


# Header
st.markdown('<div class="main-title">Smart Attendance System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-POWERED · REAL-TIME · PROFESSIONAL</div>', unsafe_allow_html=True)


# Dashboard
if menu_clean == "🏠 Dashboard":
    attendance_df = load_attendance()
    today = today_str()
    today_df = (
        attendance_df[attendance_df["Date"].astype(str) == today]
        if not attendance_df.empty else pd.DataFrame()
    )

    total_students = get_total_unique_students()
    total_records = len(attendance_df)
    present_today = today_df["RollNo"].astype(str).nunique() if not today_df.empty else 0
    absent_today = max(total_students - present_today, 0)
    attendance_rate = round((present_today / total_students) * 100, 2) if total_students > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in zip(
        (c1, c2, c3, c4),
        ("Total Students", "Present Today", "Absent Today", "Attendance Rate"),
        (total_students, present_today, absent_today, f"{attendance_rate}%"),
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📅 Today's Attendance Log")

    if today_df.empty:
        st.info("No attendance marked yet today. Go to 'Mark Attendance' to start.")
    else:
        st.dataframe(today_df.sort_values("Timestamp", ascending=False),
                     use_container_width=True, hide_index=True)


# Register Student
elif menu_clean == "📝 Register Student":
    st.markdown("### 📝 Register a New Student")
    st.markdown(
        '<span class="info-badge">📸 Capture face images from your browser camera, '
        'then train the model</span>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        rollno = st.text_input("Roll No (must be numeric, used as the ID)")
        name = st.text_input("Full Name")
    with col2:
        branch = st.text_input("Branch / Department")

    st.markdown("---")

    # Camera mode selector
    mode = st.radio(
        "Camera Mode",
        ["🌐 Browser Camera (recommended)", "🖥️ Desktop Window (OpenCV)"],
        horizontal=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Browser mode ----
    if mode.startswith("🌐"):
        st.info("Click 'Take Photo' for each face image you want to capture. "
                f"Aim for at least {NUM_CAPTURE_IMAGES} images from different angles.")

        if not (rollno and rollno.isdigit()):
            st.warning("Enter a numeric Roll No first.")
        else:
            student_dir = os.path.join(DATASET_DIR, str(rollno))
            os.makedirs(student_dir, exist_ok=True)
            existing = len([f for f in os.listdir(student_dir) if f.endswith(".jpg")])

            st.markdown(f"**Captured so far:** `{existing}` / `{NUM_CAPTURE_IMAGES}`")

            photo = st.camera_input("Camera", key="reg_camera")

            if photo is not None:
                image = Image.open(photo)
                preview, msg = process_browser_frame(image, rollno)

                if preview is None:
                    st.error(msg)
                else:
                    st.image(preview, caption=msg, use_container_width=True)
                    new_count = existing + 1
                    st.success(f"✅ Saved image {new_count}. Keep going!")

            if existing >= 5:
                if st.button("🧠 Train Model Now"):
                    if not (rollno and name and branch):
                        st.error("Please fill Roll No, Name and Branch to register.")
                    else:
                        ok, msg = add_student(rollno, name, branch)
                        if ok or "already registered" in msg:
                            with st.spinner("Training model..."):
                                trained, train_msg = train_model()
                            if trained:
                                st.success(f"✅ {name} registered. {train_msg}")
                                st.balloons()
                            else:
                                st.warning(train_msg)
                        else:
                            st.warning(msg)

    # ---- Desktop mode ----
    else:
        st.warning("⚠️ This opens a separate OpenCV window. If it doesn't appear, "
                   "switch to Browser Camera mode.")
        st.info("Make sure no other app is using your webcam.")

        if st.button("📸 Open Camera & Capture Faces"):
            if not (rollno and name and branch):
                st.error("Please fill in Roll No, Name and Branch.")
            elif not rollno.isdigit():
                st.error("Roll No must be numeric.")
            else:
                ok, msg = add_student(rollno, name, branch)
                if not ok:
                    st.warning(msg)
                else:
                    with st.spinner("Opening webcam... (ESC to stop early)"):
                        count, cap_msg = capture_faces_desktop(rollno)

                    if count == 0:
                        st.error(f"No face images captured. {cap_msg} Student record removed.")
                        df = load_students()
                        df = df[df["RollNo"].astype(str) != str(rollno)]
                        save_students(df)
                    else:
                        st.success(f"Captured {count} face images for {name}.")
                        with st.spinner("Training recognition model..."):
                            trained, train_msg = train_model()
                        if trained:
                            st.success(f"✅ {name} registered and model trained. {train_msg}")
                            st.balloons()
                        else:
                            st.warning(train_msg)


# Mark Attendance
elif menu_clean == "📸 Mark Attendance":
    st.markdown("### 📸 Mark Attendance")
    st.markdown(
        '<span class="info-badge">⏱️ 1 hour gap required between two Present entries '
        'for the same student</span>',
        unsafe_allow_html=True
    )

    students_df = load_students()
    if students_df.empty:
        st.warning("⚠️ No students registered yet. Go to 'Register Student' first.")
    elif not os.path.exists(TRAINER_PATH):
        st.warning("⚠️ No trained model found. Register at least one student first.")
    else:
        mode = st.radio(
            "Camera Mode",
            ["🌐 Browser Camera (recommended)", "🖥️ Desktop Window (OpenCV)"],
            horizontal=True,
            key="mark_mode"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Browser mode
        if mode.startswith("🌐"):
            st.info("Take a photo - faces in the frame will be recognized and marked.")
            photo = st.camera_input("Camera", key="mark_camera")

            if photo is not None:
                image = Image.open(photo)
                with st.spinner("Recognizing faces..."):
                    results, msg = recognize_from_browser_frame(image)

                if not results:
                    st.warning(msg)
                else:
                    for status, line in results:
                        if status == "success":
                            st.success(f"✅ {line}")
                        elif status == "reverified":
                            st.warning(f"⏳ {line}")
                        else:
                            st.error(f"❌ {line}")

        # Desktop mode
        else:
            st.warning("⚠️ Opens a separate OpenCV window. Press ESC to stop.")
            if st.button("▶️ Start Camera & Recognize"):
                with st.spinner("Camera running - press ESC to stop..."):
                    status, log = run_recognition_desktop()

                if status == "no_camera":
                    st.error("Could not open the webcam.")
                elif status == "no_model":
                    st.error("No trained model found.")
                elif not log:
                    st.info("Session ended - no attendance was newly marked.")
                else:
                    st.success("Session ended. Results:")
                    for line in log:
                        st.write(f"- {line}")


# Attendance Records
elif menu_clean == "📊 Attendance Records":
    st.markdown("### 📊 Attendance Records")
    df = load_attendance()

    if df.empty:
        st.info("No attendance records yet.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            dates = ["All"] + sorted(df["Date"].astype(str).unique().tolist(), reverse=True)
            date_filter = st.selectbox("Filter by Date", dates)
        with col2:
            branches = ["All"] + sorted(df["Branch"].astype(str).dropna().unique().tolist())
            branch_filter = st.selectbox("Filter by Branch", branches)
        with col3:
            statuses = ["All"] + sorted(df["Status"].astype(str).dropna().unique().tolist())
            status_filter = st.selectbox("Filter by Status", statuses)

        filtered = df.copy()
        if date_filter != "All":
            filtered = filtered[filtered["Date"].astype(str) == date_filter]
        if branch_filter != "All":
            filtered = filtered[filtered["Branch"].astype(str) == branch_filter]
        if status_filter != "All":
            filtered = filtered[filtered["Status"].astype(str) == status_filter]

        st.markdown(f"**{len(filtered)}** records found")
        st.dataframe(filtered.sort_values("Timestamp", ascending=False),
                     use_container_width=True, hide_index=True)

        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv,
                            file_name=f"attendance_{today_str()}.csv", mime="text/csv")


# Students
elif menu_clean == "👥 Students":
    st.markdown("### 👥 Registered Students")
    df = load_students()

    if df.empty:
        st.info("No students registered yet. Go to 'Register Student' to add one.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


# About
elif menu_clean == "ℹ️ About":
    st.markdown("### ℹ️ About This System")
    st.markdown(f"""
    **Smart Attendance System** is an AI-powered attendance solution that:

    - 📸 Uses real face recognition (Haar cascade + LBPH) to mark attendance
    - 🕒 Uses your system's real current time for every entry
    - ⏱️ Blocks a student from being marked Present twice within 1 hour
    - 📊 Gives a real-time dashboard and filterable records
    - 💾 Supports CSV export

    ---

    **Current Server Time:** `{now_ts()}`
    """)


# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align:center; color:#6b7280; font-size:0.82rem; padding:0.8rem 0;">
        🎓 Smart Attendance System · Built with ❤️ using Streamlit
        <br>
        <span style="font-size:0.72rem;">Last refreshed: {now_ts()}</span>
    </div>
    """,
    unsafe_allow_html=True
)
