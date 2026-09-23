import os
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import pandas as pd
import streamlit as st
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

STUDENTS_FILE = BASE_DIR / "data" / "students.csv"
ATTENDANCE_FILE = BASE_DIR / "attendance" / "attendance.xlsx"
TRAINER_FILE = BASE_DIR / "trainer" / "trainer.yml"
DATASET_DIR = BASE_DIR / "dataset"
HAAR_FILE = (
    BASE_DIR
    / "haarcascade"
    / "haarcascade_frontalface_default.xml"
)

STUDENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
ATTENDANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
TRAINER_FILE.parent.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IST = ZoneInfo("Asia/Kolkata")

RECOGNITION_THRESHOLD = 70
ATTENDANCE_COOLDOWN_MINUTES = 60
FACE_MIN_SIZE = (60, 60)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL
       ===================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(45, 65, 95, 0.20),
                transparent 35%
            ),
            #0b0f14;
        color: #f5f7fa;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    h1, h2, h3 {
        color: #f8fafc !important;
    }

    p, label {
        color: #b8c0cc;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        min-width: 285px;
        max-width: 285px;
        background:
            linear-gradient(
                180deg,
                #101722 0%,
                #0b1119 55%,
                #080d14 100%
            );
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] > div {
        padding: 1.2rem 1rem 1rem 1rem;
    }

    .sidebar-brand {
        padding: 10px 8px 18px 8px;
    }

    .sidebar-logo {
        width: 48px;
        height: 48px;
        border-radius: 14px;
        background: linear-gradient(
            135deg,
            #2563eb,
            #4f46e5
        );
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 25px;
        margin-bottom: 13px;
        box-shadow:
            0 8px 25px rgba(37, 99, 235, 0.25);
    }

    .sidebar-title {
        font-size: 20px;
        font-weight: 700;
        letter-spacing: -0.3px;
        color: #f8fafc;
        line-height: 1.2;
    }

    .sidebar-subtitle {
        margin-top: 7px;
        color: #8994a3;
        font-size: 12px;
        line-height: 1.55;
    }

    .sidebar-section {
        color: #64748b;
        text-transform: uppercase;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.2px;
        padding: 8px 11px 7px 11px;
    }

    /* Radio navigation */

    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 5px;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        width: 100%;
        min-height: 47px;
        padding: 10px 12px;
        border-radius: 11px;
        color: #aeb8c6;
        background: transparent;
        border: 1px solid transparent;
        transition: all 0.18s ease;
        cursor: pointer;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"]
    label:hover {
        background: rgba(255,255,255,0.055);
        border-color: rgba(255,255,255,0.06);
        color: #ffffff;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"]
    label:has(input:checked) {
        background:
            linear-gradient(
                90deg,
                rgba(37,99,235,0.22),
                rgba(79,70,229,0.12)
            );
        border-color: rgba(59,130,246,0.28);
        color: #ffffff;
        box-shadow:
            inset 3px 0 0 #3b82f6;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"]
    label p {
        font-size: 13px !important;
        font-weight: 600;
        color: inherit !important;
        margin: 0;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"]
    label div[data-testid="stMarkdownContainer"] {
        padding: 0;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"]
    input {
        display: none;
    }

    .sidebar-divider {
        height: 1px;
        background: rgba(255,255,255,0.07);
        margin: 17px 6px;
    }

    .sidebar-status {
        margin: 9px 5px;
        padding: 12px 13px;
        border-radius: 12px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.06);
    }

    .status-top {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #dce4ed;
        font-size: 12px;
        font-weight: 650;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22c55e;
        box-shadow: 0 0 9px rgba(34,197,94,0.7);
        display: inline-block;
    }

    .status-text {
        color: #758195;
        font-size: 10px;
        line-height: 1.5;
        margin-top: 5px;
    }

    .sidebar-footer {
        margin: 24px 6px 4px 6px;
        padding-top: 15px;
        border-top: 1px solid rgba(255,255,255,0.06);
        color: #566274;
        font-size: 10px;
        line-height: 1.6;
        text-align: center;
    }


    /* =====================================================
       CARDS
       ===================================================== */

    .stat-card {
        background:
            linear-gradient(
                145deg,
                rgba(24,32,45,0.96),
                rgba(13,18,26,0.96)
            );
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 19px;
        min-height: 125px;
        box-shadow:
            0 8px 25px rgba(0,0,0,0.15);
    }

    .stat-label {
        color: #8792a2;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 9px;
    }

    .stat-value {
        color: #f8fafc;
        font-size: 29px;
        font-weight: 750;
        line-height: 1.1;
    }

    .stat-sub {
        color: #667386;
        font-size: 10px;
        margin-top: 8px;
    }

    .page-header {
        margin-bottom: 24px;
    }

    .page-title {
        font-size: 30px;
        font-weight: 750;
        color: #f8fafc;
        margin-bottom: 4px;
    }

    .page-subtitle {
        color: #7f8b9c;
        font-size: 13px;
    }

    .info-card {
        background: rgba(17,24,34,0.85);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 18px;
    }

    .success-box {
        padding: 13px 16px;
        border-radius: 11px;
        background: rgba(34,197,94,0.08);
        border: 1px solid rgba(34,197,94,0.20);
        color: #86efac;
    }

    .warning-box {
        padding: 13px 16px;
        border-radius: 11px;
        background: rgba(245,158,11,0.08);
        border: 1px solid rgba(245,158,11,0.20);
        color: #fcd34d;
    }

    .danger-box {
        padding: 13px 16px;
        border-radius: 11px;
        background: rgba(239,68,68,0.08);
        border: 1px solid rgba(239,68,68,0.20);
        color: #fca5a5;
    }

    .camera-box {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 15px;
        padding: 12px;
        background: #0d131c;
    }

    /* Buttons */

    .stButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 650;
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* Inputs */

    div[data-baseweb="input"] {
        background: #111822;
        border-color: rgba(255,255,255,0.08);
    }

    div[data-baseweb="select"] > div {
        background: #111822;
        border-color: rgba(255,255,255,0.08);
    }

    /* Tables */

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Hide Streamlit branding */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def now_ist():
    """
    Returns current India Standard Time.
    """
    return datetime.now(IST).replace(tzinfo=None)


def normalize_column_name(column):
    return (
        str(column)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


# ============================================================
# STUDENTS
# ============================================================

def load_students():
    """
    Loads students.csv.
    Supports:
    - tab-separated file
    - comma-separated file
    - header/no-header formats
    """

    if not STUDENTS_FILE.exists():
        return pd.DataFrame(
            columns=["rollno", "name", "branch"]
        )

    try:
        df = pd.read_csv(
            STUDENTS_FILE,
            sep=None,
            engine="python",
            dtype=str
        )
    except Exception:
        try:
            df = pd.read_csv(
                STUDENTS_FILE,
                sep="\t",
                dtype=str
            )
        except Exception:
            return pd.DataFrame(
                columns=["rollno", "name", "branch"]
            )

    if df.empty:
        return pd.DataFrame(
            columns=["rollno", "name", "branch"]
        )

    # Normalize columns
    normalized = {
        col: normalize_column_name(col)
        for col in df.columns
    }

    df.rename(columns=normalized, inplace=True)

    # Detect standard columns
    rename_map = {}

    for col in df.columns:

        if col in ["roll no", "rollno", "roll number", "id"]:
            rename_map[col] = "rollno"

        elif col in ["name", "student name", "student"]:
            rename_map[col] = "name"

        elif col in ["branch", "department", "dept"]:
            rename_map[col] = "branch"

    df.rename(columns=rename_map, inplace=True)

    # Headerless file fallback
    if not {"rollno", "name", "branch"}.issubset(df.columns):

        try:
            raw = pd.read_csv(
                STUDENTS_FILE,
                sep="\t",
                header=None,
                dtype=str
            )

            if raw.shape[1] >= 3:

                raw = raw.iloc[:, :3]
                raw.columns = [
                    "rollno",
                    "name",
                    "branch"
                ]

                df = raw

        except Exception:
            pass

    for col in ["rollno", "name", "branch"]:
        if col not in df.columns:
            df[col] = ""

    df = df[
        ["rollno", "name", "branch"]
    ].fillna("")

    df["rollno"] = df["rollno"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df["branch"] = df["branch"].astype(str).str.strip()

    return df


def save_student(rollno, name, branch):
    """
    Saves student in existing students.csv.
    """

    df = load_students()

    new_row = pd.DataFrame(
        [{
            "rollno": str(rollno).strip(),
            "name": str(name).strip(),
            "branch": str(branch).strip()
        }]
    )

    df = pd.concat(
        [df, new_row],
        ignore_index=True
    )

    df.to_csv(
        STUDENTS_FILE,
        sep="\t",
        index=False
    )


# ============================================================
# ATTENDANCE
# ============================================================

def load_attendance():
    """
    Loads attendance Excel file.
    """

    if not ATTENDANCE_FILE.exists():
        return pd.DataFrame(
            columns=[
                "roll no",
                "name",
                "branch",
                "date",
                "time",
                "status"
            ]
        )

    try:
        df = pd.read_excel(
            ATTENDANCE_FILE
        )
    except Exception:
        return pd.DataFrame(
            columns=[
                "roll no",
                "name",
                "branch",
                "date",
                "time",
                "status"
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
                "status"
            ]
        )

    rename_map = {}

    for col in df.columns:

        normalized = normalize_column_name(col)

        if normalized in [
            "roll no",
            "rollno",
            "roll number"
        ]:
            rename_map[col] = "roll no"

        elif normalized in [
            "name",
            "student name"
        ]:
            rename_map[col] = "name"

        elif normalized in [
            "branch",
            "department"
        ]:
            rename_map[col] = "branch"

        elif normalized == "date":
            rename_map[col] = "date"

        elif normalized == "time":
            rename_map[col] = "time"

        elif normalized == "status":
            rename_map[col] = "status"

    df.rename(columns=rename_map, inplace=True)

    required = [
        "roll no",
        "name",
        "branch",
        "date",
        "time",
        "status"
    ]

    for col in required:
        if col not in df.columns:
            df[col] = ""

    return df[required]


def save_attendance_record(
    rollno,
    name,
    branch,
    timestamp,
    status="Present"
):
    """
    Saves attendance record to Excel.
    """

    df = load_attendance()

    new_record = pd.DataFrame(
        [{
            "roll no": str(rollno),
            "name": str(name),
            "branch": str(branch),
            "date": timestamp.strftime("%Y-%m-%d"),
            "time": timestamp.strftime("%H:%M:%S"),
            "status": status
        }]
    )

    df = pd.concat(
        [df, new_record],
        ignore_index=True
    )

    df.to_excel(
        ATTENDANCE_FILE,
        index=False
    )


def can_mark_attendance(rollno):
    """
    Strict 1-hour attendance rule.
    """

    df = load_attendance()

    if df.empty:
        return True, None

    matching = df[
        df["roll no"].astype(str).str.strip()
        == str(rollno).strip()
    ].copy()

    if matching.empty:
        return True, None

    latest = matching.iloc[-1]

    try:
        date_value = str(latest["date"]).strip()
        time_value = str(latest["time"]).strip()

        last_time = datetime.strptime(
            f"{date_value} {time_value}",
            "%Y-%m-%d %H:%M:%S"
        )

    except Exception:
        return True, None

    current_time = now_ist()

    difference = (
        current_time - last_time
    )

    if difference < timedelta(
        minutes=ATTENDANCE_COOLDOWN_MINUTES
    ):
        remaining = (
            timedelta(
                minutes=ATTENDANCE_COOLDOWN_MINUTES
            )
            - difference
        )

        return False, remaining

    return True, None


def mark_attendance(
    rollno,
    name,
    branch
):
    """
    Marks attendance at the exact moment
    the function is called.
    """

    timestamp = now_ist()

    allowed, remaining = can_mark_attendance(
        rollno
    )

    if not allowed:
        return False, remaining, timestamp

    save_attendance_record(
        rollno=rollno,
        name=name,
        branch=branch,
        timestamp=timestamp,
        status="Present"
    )

    return True, None, timestamp


# ============================================================
# OPENCV / FACE MODEL
# ============================================================

@st.cache_resource
def load_face_cascade():
    cascade = cv2.CascadeClassifier(
        str(HAAR_FILE)
    )

    if cascade.empty():
        return None

    return cascade


@st.cache_resource
def load_recognizer():
    """
    LBPH face recognizer.
    """

    if not hasattr(
        cv2,
        "face"
    ):
        return None

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )

    if TRAINER_FILE.exists():

        try:
            recognizer.read(
                str(TRAINER_FILE)
            )
        except Exception:
            pass

    return recognizer


# ============================================================
# FACE DETECTION
# ============================================================

def detect_faces(image):
    """
    Detect faces with Haar Cascade.
    Uses histogram equalization and fallback detection.
    """

    cascade = load_face_cascade()

    if cascade is None:
        return []

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=7,
        minSize=FACE_MIN_SIZE
    )

    valid_faces = []

    for (x, y, w, h) in faces:

        area = w * h

        if area >= 4000:
            valid_faces.append(
                (x, y, w, h)
            )

    # Fallback
    if not valid_faces:

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(40, 40)
        )

        for (x, y, w, h) in faces:

            area = w * h

            if area >= 3000:
                valid_faces.append(
                    (x, y, w, h)
                )

    return valid_faces


def detect_single_face(image):
    """
    Selects the largest detected face.

    This avoids false 'multiple faces' errors caused
    by overlapping Haar detections.
    """

    faces = detect_faces(image)

    if not faces:
        return None

    faces = sorted(
        faces,
        key=lambda box: box[2] * box[3],
        reverse=True
    )

    return faces[0]


# ============================================================
# TRAINING DATA AUGMENTATION
# ============================================================

def create_training_samples(
    face_image,
    student_folder
):
    """
    Creates multiple training images from one captured face.
    """

    student_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    image = Image.fromarray(
        cv2.cvtColor(
            face_image,
            cv2.COLOR_BGR2RGB
        )
    )

    samples = []

    # Original
    samples.append(
        image
    )

    # Flip
    samples.append(
        image.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )
    )

    # Blur
    samples.append(
        image.filter(
            ImageFilter.GaussianBlur(
                radius=1
            )
        )
    )

    # Brightness
    samples.append(
        ImageEnhance.Brightness(
            image
        ).enhance(1.15)
    )

    samples.append(
        ImageEnhance.Brightness(
            image
        ).enhance(0.85)
    )

    # Contrast
    samples.append(
        ImageEnhance.Contrast(
            image
        ).enhance(1.2)
    )

    # Sharpness
    samples.append(
        ImageEnhance.Sharpness(
            image
        ).enhance(1.5)
    )

    existing = list(
        student_folder.glob("*.jpg")
    )

    start_index = len(existing)

    for index, sample in enumerate(samples):

        filename = (
            student_folder
            / f"User.{start_index + index + 1}.jpg"
        )

        sample.save(
            filename,
            quality=95
        )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():
    """
    Trains LBPH model using dataset/<rollno>/ images.
    """

    if not hasattr(
        cv2,
        "face"
    ):
        return False, (
            "OpenCV contrib package is required."
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )

    detector = load_face_cascade()

    if detector is None:
        return False, (
            "Haar Cascade file not found."
        )

    faces = []
    ids = []

    for folder in DATASET_DIR.iterdir():

        if not folder.is_dir():
            continue

        try:
            student_id = int(
                folder.name
            )
        except ValueError:
            continue

        for image_path in folder.glob(
            "*.jpg"
        ):

            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                continue

            image = cv2.equalizeHist(
                image
            )

            faces.append(
                image
            )

            ids.append(
                student_id
            )

    if not faces:
        return False, (
            "No training images found."
        )

    recognizer.train(
        faces,
        np.array(ids)
    )

    recognizer.write(
        str(TRAINER_FILE)
    )

    # Clear cached model so next recognition
    # loads the latest model.
    load_recognizer.clear()

    return True, (
        f"Model trained successfully with "
        f"{len(faces)} images."
    )


# ============================================================
# RECOGNITION
# ============================================================

def recognize_face(image):
    """
    Detect and recognize the largest face.

    Returns:
        annotated_image,
        recognized_student,
        confidence
    """

    students = load_students()

    recognizer = load_recognizer()

    if recognizer is None:
        return image, None, None

    face = detect_single_face(
        image
    )

    if face is None:
        return image, None, None

    x, y, w, h = face

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(
        gray
    )

    face_roi = gray[
        y:y+h,
        x:x+w
    ]

    try:
        student_id, confidence = (
            recognizer.predict(
                face_roi
            )
        )
    except Exception:
        return image, None, None

    annotated = image.copy()

    # LBPH lower distance = better
    if confidence <= RECOGNITION_THRESHOLD:

        matching = students[
            students["rollno"].astype(str).str.strip()
            == str(student_id).strip()
        ]

        if not matching.empty:

            student = matching.iloc[0]

            name = str(
                student["name"]
            )

            rollno = str(
                student["rollno"]
            )

            label = (
                f"{name} | Roll: {rollno}"
            )

            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                (0, 220, 100),
                3
            )

            # Label background
            (text_w, text_h), baseline = (
                cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    2
                )
            )

            label_y = max(
                y - 12,
                text_h + 15
            )

            cv2.rectangle(
                annotated,
                (
                    x,
                    label_y - text_h - 12
                ),
                (
                    x + text_w + 12,
                    label_y + baseline - 5
                ),
                (0, 130, 65),
                -1
            )

            cv2.putText(
                annotated,
                label,
                (x + 6, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            return (
                annotated,
                student.to_dict(),
                confidence
            )

    # Unknown face
    label = "Unknown Face"

    cv2.rectangle(
        annotated,
        (x, y),
        (x + w, y + h),
        (0, 80, 255),
        3
    )

    cv2.putText(
        annotated,
        label,
        (x, max(y - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 80, 255),
        2,
        cv2.LINE_AA
    )

    return (
        annotated,
        None,
        confidence
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">

            <div class="sidebar-logo">
                🎓
            </div>

            <div class="sidebar-title">
                Smart Attendance
            </div>

            <div class="sidebar-subtitle">
                AI-powered face recognition<br>
                attendance management
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-section">
            Navigation
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📝 Register Student",
            "📷 Mark Attendance",
            "👥 Students",
            "📊 Attendance Records",
        ],
        label_visibility="collapsed"
    )

    st.markdown(
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-status">

            <div class="status-top">
                <span class="status-dot"></span>
                System Online
            </div>

            <div class="status-text">
                Face recognition system is ready
                for attendance verification.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-status">

            <div class="status-top">
                🇮🇳 India Standard Time
            </div>

            <div class="status-text">
                Attendance timestamps are recorded
                using IST timezone.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-footer">
            Smart Attendance System<br>
            AI Face Recognition • v1.0
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    students = load_students()
    attendance = load_attendance()

    current_time = now_ist()

    today = current_time.strftime(
        "%Y-%m-%d"
    )

    today_attendance = attendance[
        attendance["date"].astype(str)
        == today
    ]

    unique_today = (
        today_attendance["roll no"]
        .astype(str)
        .nunique()
    )

    st.markdown(
        """
        <div class="page-header">

            <div class="page-title">
                Dashboard
            </div>

            <div class="page-subtitle">
                Smart attendance monitoring
                and face recognition overview
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">
                    TOTAL STUDENTS
                </div>
                <div class="stat-value">
                    {len(students)}
                </div>
                <div class="stat-sub">
                    Registered students
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">
                    TODAY'S ATTENDANCE
                </div>
                <div class="stat-value">
                    {unique_today}
                </div>
                <div class="stat-sub">
                    Unique students today
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">
                    TOTAL RECORDS
                </div>
                <div class="stat-value">
                    {len(attendance)}
                </div>
                <div class="stat-sub">
                    Attendance history
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        model_status = (
            "Ready"
            if TRAINER_FILE.exists()
            else "Not trained"
        )

        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">
                    FACE MODEL
                </div>
                <div class="stat-value"
                     style="font-size:23px;">
                    {model_status}
                </div>
                <div class="stat-sub">
                    LBPH recognition
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    left, right = st.columns(
        [1.35, 1]
    )

    with left:

        st.markdown(
            """
            <div class="info-card">

                <h3 style="margin-top:0;">
                    System Status
                </h3>

                <p>
                    Your Smart Attendance System is
                    running with browser-based camera
                    capture and LBPH face recognition.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        if TRAINER_FILE.exists():

            st.markdown(
                """
                <div class="success-box">
                    ✓ Face recognition model is available
                    and ready for attendance verification.
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="warning-box">
                    ⚠ Face recognition model is not
                    available yet. Register a student
                    to train the model.
                </div>
                """,
                unsafe_allow_html=True
            )

    with right:

        st.markdown(
            f"""
            <div class="info-card">

                <h3 style="margin-top:0;">
                    Current Time
                </h3>

                <div style="
                    font-size:30px;
                    font-weight:700;
                    color:#f8fafc;
                    margin-top:12px;
                ">
                    {current_time.strftime("%I:%M:%S %p")}
                </div>

                <div style="
                    color:#758195;
                    font-size:12px;
                    margin-top:5px;
                ">
                    {current_time.strftime("%d %B %Y")}
                    • India Standard Time
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# REGISTER STUDENT
# ============================================================

elif page == "📝 Register Student":

    st.markdown(
        """
        <div class="page-header">

            <div class="page-title">
                Register Student
            </div>

            <div class="page-subtitle">
                Add a new student and create face
                recognition training data.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    left, right = st.columns(
        [0.9, 1.1]
    )

    with left:

        st.markdown(
            "### Student Information"
        )

        rollno = st.text_input(
            "Roll Number",
            placeholder="Enter roll number"
        )

        name = st.text_input(
            "Student Name",
            placeholder="Enter full name"
        )

        branch = st.text_input(
            "Branch",
            placeholder="e.g. CSE"
        )

        st.markdown(
            """
            <div class="warning-box">
                📌 Keep only one person in front of
                the camera while registering.
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            "### Capture Face"

        )

        camera_image = st.camera_input(
            "Take a clear face photo",
            key="register_camera"
        )

    if st.button(
        "Register Student",
        type="primary",
        width="stretch"
    ):

        if not rollno.strip():
            st.error(
                "Please enter roll number."
            )
            st.stop()

        if not name.strip():
            st.error(
                "Please enter student name."
            )
            st.stop()

        if not branch.strip():
            st.error(
                "Please enter branch."
            )
            st.stop()

        if camera_image is None:
            st.error(
                "Please capture a face image."
            )
            st.stop()

        students = load_students()

        if (
            students["rollno"]
            .astype(str)
            .str.strip()
            .eq(rollno.strip())
            .any()
        ):
            st.error(
                "This roll number is already registered."
            )
            st.stop()

        # Convert uploaded image
        file_bytes = np.asarray(
            bytearray(
                camera_image.getvalue()
            ),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:
            st.error(
                "Unable to process captured image."
            )
            st.stop()

        face = detect_single_face(
            image
        )

        if face is None:

            st.error(
                "No face detected. "
                "Please capture a clearer image "
                "with your face visible."
            )

            st.image(
                image,
                channels="BGR",
                width="stretch"
            )

            st.stop()

        x, y, w, h = face

        face_crop = image[
            y:y+h,
            x:x+w
        ]

        # Draw face rectangle
        preview = image.copy()

        cv2.rectangle(
            preview,
            (x, y),
            (x + w, y + h),
            (0, 220, 100),
            3
        )

        cv2.putText(
            preview,
            f"{name} | Roll: {rollno}",
            (x, max(y - 12, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 100),
            2,
            cv2.LINE_AA
        )

        student_folder = (
            DATASET_DIR
            / str(rollno).strip()
        )

        create_training_samples(
            face_crop,
            student_folder
        )

        with st.spinner(
            "Training face recognition model..."
        ):

            success, message = train_model()

        if not success:

            st.error(message)
            st.stop()

        save_student(
            rollno=rollno,
            name=name,
            branch=branch
        )

        st.success(
            f"✓ {name} registered successfully."
        )

        st.image(
            preview,
            channels="BGR",
            caption=(
                "Detected face — "
                "training data created"
            ),
            width="stretch"
        )

        st.info(
            "Face recognition model has been "
            "updated successfully."
        )


# ============================================================
# MARK ATTENDANCE
# ============================================================

elif page == "📷 Mark Attendance":

    st.markdown(
        """
        <div class="page-header">

            <div class="page-title">
                Mark Attendance
            </div>

            <div class="page-subtitle">
                Capture a face and verify the student
                using the trained recognition model.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    if not TRAINER_FILE.exists():

        st.warning(
            "No trained face model found. "
            "Please register at least one student first."
        )

        st.stop()

    camera_image = st.camera_input(
        "Capture face for attendance",
        key="attendance_camera"
    )

    if camera_image is not None:

        file_bytes = np.asarray(
            bytearray(
                camera_image.getvalue()
            ),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:

            st.error(
                "Unable to process camera image."
            )

            st.stop()

        annotated, student, confidence = (
            recognize_face(
                image
            )
        )

        st.image(
            annotated,
            channels="BGR",
            caption="Face recognition result",
            width="stretch"
        )

        if student is None:

            if confidence is None:

                st.error(
                    "No face detected. "
                    "Please capture your face clearly."
                )

            else:

                st.error(
                    "Face detected, but the student "
                    "could not be recognized."
                )

            st.stop()

        name = str(
            student["name"]
        )

        rollno = str(
            student["rollno"]
        )

        branch = str(
            student["branch"]
        )

        st.markdown(
            f"""
            <div class="success-box">

                <strong>✓ Student Recognized</strong><br><br>

                Name:
                <strong>{name}</strong><br>

                Roll Number:
                <strong>{rollno}</strong><br>

                Branch:
                <strong>{branch}</strong>

            </div>
            """,
            unsafe_allow_html=True
        )

        if confidence is not None:

            st.caption(
                f"Recognition distance: "
                f"{confidence:.2f}"
            )

        st.write("")

        if st.button(
            "✓ Confirm & Mark Attendance",
            type="primary",
            width="stretch"
        ):

            success, remaining, timestamp = (
                mark_attendance(
                    rollno,
                    name,
                    branch
                )
            )

            if success:

                st.success(
                    f"Attendance marked successfully "
                    f"at {timestamp.strftime('%I:%M:%S %p')} IST."
                )

            else:

                total_seconds = int(
                    remaining.total_seconds()
                )

                minutes = (
                    total_seconds // 60
                )

                seconds = (
                    total_seconds % 60
                )

                st.warning(
                    f"Attendance already marked. "
                    f"Please wait approximately "
                    f"{minutes} min {seconds} sec "
                    f"before marking again."
                )


# ============================================================
# STUDENTS
# ============================================================

elif page == "👥 Students":

    st.markdown(
        """
        <div class="page-header">

            <div class="page-title">
                Students
            </div>

            <div class="page-subtitle">
                View all registered students.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    students = load_students()

    if students.empty:

        st.info(
            "No students registered yet."
        )

    else:

        st.markdown(
            f"""
            <div class="info-card">
                Total Registered Students:
                <strong>{len(students)}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        display_students = students.copy()

        display_students.columns = [
            "Roll Number",
            "Name",
            "Branch"
        ]

        st.dataframe(
            display_students,
            width="stretch",
            hide_index=True
        )


# ============================================================
# ATTENDANCE RECORDS
# ============================================================

elif page == "📊 Attendance Records":

    st.markdown(
        """
        <div class="page-header">

            <div class="page-title">
                Attendance Records
            </div>

            <div class="page-subtitle">
                Complete attendance history and
                today's attendance records.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    attendance = load_attendance()

    if attendance.empty:

        st.info(
            "No attendance records available."
        )

    else:

        # Today's records
        current_date = now_ist().strftime(
            "%Y-%m-%d"
        )

        today_records = attendance[
            attendance["date"].astype(str)
            == current_date
        ].copy()

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">
                        TOTAL RECORDS
                    </div>
                    <div class="stat-value">
                        {len(attendance)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">
                        TODAY
                    </div>
                    <div class="stat-value">
                        {len(today_records)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:
            unique_today = (
                today_records["roll no"]
                .astype(str)
                .nunique()
            )

            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">
                        UNIQUE STUDENTS
                    </div>
                    <div class="stat-value">
                        {unique_today}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        tab1, tab2 = st.tabs(
            [
                "📅 Today's Attendance",
                "📋 All Records"
            ]
        )

        with tab1:

            if today_records.empty:

                st.info(
                    "No attendance marked today."
                )

            else:

                display_today = today_records.copy()

                display_today = display_today[
                    [
                        "roll no",
                        "name",
                        "branch",
                        "date",
                        "time",
                        "status"
                    ]
                ]

                display_today.columns = [
                    "Roll Number",
                    "Name",
                    "Branch",
                    "Date",
                    "Time",
                    "Status"
                ]

                st.dataframe(
                    display_today,
                    width="stretch",
                    hide_index=True
                )

        with tab2:

            display_all = attendance.copy()

            display_all = display_all[
                [
                    "roll no",
                    "name",
                    "branch",
                    "date",
                    "time",
                    "status"
                ]
            ]

            display_all.columns = [
                "Roll Number",
                "Name",
                "Branch",
                "Date",
                "Time",
                "Status"
            ]

            st.dataframe(
                display_all,
                width="stretch",
                hide_index=True
            )
