import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from streamlit_autorefresh import st_autorefresh


# Page configuration
st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STUDENT_PATH = os.path.join(BASE_DIR, "data", "students.csv")
ATTENDANCE_PATH = os.path.join(BASE_DIR, "attendance", "attendance.xlsx")
ATTENDANCE_CSV_PATH = os.path.join(BASE_DIR, "attendance", "attendance.csv")
TRAINER_PATH = os.path.join(BASE_DIR, "trainer", "trainer.yml")
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
CASCADE_PATH = os.path.join(
    BASE_DIR,
    "haarcascade",
    "haarcascade_frontalface_default.xml"
)

os.makedirs(os.path.dirname(STUDENT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(ATTENDANCE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(TRAINER_PATH), exist_ok=True)
os.makedirs(DATASET_PATH, exist_ok=True)


# India timezone
IST = ZoneInfo("Asia/Kolkata")


def now_ist():
    return datetime.now(IST).replace(tzinfo=None)


# Auto refresh for live clock
st_autorefresh(
    interval=1000,
    key="smart_attendance_clock"
)


# Custom styling
st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(30, 80, 110, 0.22),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #07111f 0%,
                #0b1625 50%,
                #101827 100%
            );
        color: #f5f7fa;
    }

    [data-testid="stSidebar"] {
        background: #07111f;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] * {
        color: #f5f7fa;
    }

    .brand {
        text-align: center;
        padding: 8px 0 25px 0;
    }

    .brand-icon {
        font-size: 42px;
    }

    .brand-title {
        font-size: 22px;
        font-weight: 800;
        margin-top: 5px;
    }

    .brand-subtitle {
        font-size: 12px;
        color: #91a0b5;
        margin-top: 3px;
    }

    .hero {
        background:
            linear-gradient(
                135deg,
                rgba(22, 39, 61, 0.95),
                rgba(10, 25, 42, 0.95)
            );
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 28px;
        margin-bottom: 22px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.25);
    }

    .hero-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 7px;
    }

    .hero-text {
        color: #9eacbd;
        font-size: 15px;
    }

    .metric-card {
        background: rgba(18, 32, 49, 0.95);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 17px;
        padding: 20px;
        min-height: 125px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }

    .metric-label {
        color: #93a2b7;
        font-size: 13px;
        margin-bottom: 9px;
    }

    .metric-value {
        font-size: 29px;
        font-weight: 800;
    }

    .metric-small {
        color: #6f8097;
        font-size: 12px;
        margin-top: 6px;
    }

    .section-card {
        background: rgba(15, 29, 45, 0.95);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 18px;
        padding: 22px;
        margin-top: 20px;
    }

    .result-card {
        border-radius: 17px;
        padding: 20px;
        margin-top: 18px;
        border: 1px solid rgba(255,255,255,0.1);
        background: rgba(20, 34, 50, 0.96);
    }

    .recognized-card {
        border-left: 5px solid #20d695;
    }

    .warning-card {
        border-left: 5px solid #f0b429;
    }

    .error-card {
        border-left: 5px solid #ff5c69;
    }

    .result-title {
        font-size: 20px;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .result-row {
        display: flex;
        justify-content: space-between;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        padding: 9px 0;
    }

    .result-key {
        color: #91a0b5;
    }

    .result-value {
        font-weight: 700;
    }

    .live-clock {
        text-align: right;
        color: #9baabd;
        font-size: 13px;
        margin-bottom: 15px;
    }

    .status-online {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 20px;
        background: rgba(32,214,149,0.12);
        color: #20d695;
        font-size: 12px;
        font-weight: 700;
    }

    .info-box {
        background: rgba(30, 45, 63, 0.75);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 15px;
        color: #aab7c8;
        font-size: 13px;
    }

    .footer {
        text-align: center;
        color: #64758a;
        font-size: 12px;
        margin-top: 35px;
        padding-bottom: 20px;
    }

    div[data-testid="stCameraInput"] {
        border-radius: 15px;
        overflow: hidden;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 42px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# Student file helpers
def load_students():
    if not os.path.exists(STUDENT_PATH):
        return pd.DataFrame(columns=["rollno", "name", "branch"])

    try:
        df = pd.read_csv(
            STUDENT_PATH,
            sep=None,
            engine="python"
        )
    except Exception:
        try:
            df = pd.read_csv(STUDENT_PATH, sep="\t")
        except Exception:
            return pd.DataFrame(columns=["rollno", "name", "branch"])

    if df.empty:
        return pd.DataFrame(columns=["rollno", "name", "branch"])

    df.columns = [
        str(col).strip().lower().replace(" ", "_")
        for col in df.columns
    ]

    column_map = {}

    for col in df.columns:
        if col in ["rollno", "roll_no", "roll", "student_id", "id"]:
            column_map[col] = "rollno"

        elif col in ["name", "student_name", "fullname", "full_name"]:
            column_map[col] = "name"

        elif col in [
            "branch",
            "department",
            "dept",
            "course",
            "stream"
        ]:
            column_map[col] = "branch"

    df = df.rename(columns=column_map)

    for required in ["rollno", "name", "branch"]:
        if required not in df.columns:
            df[required] = ""

    df = df[["rollno", "name", "branch"]].copy()

    df["rollno"] = df["rollno"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df["branch"] = df["branch"].astype(str).str.strip()

    df = df[
        (df["rollno"] != "") &
        (df["rollno"].str.lower() != "nan")
    ]

    return df.reset_index(drop=True)


def save_students(df):
    df = df[["rollno", "name", "branch"]].copy()

    df.to_csv(
        STUDENT_PATH,
        sep="\t",
        index=False
    )


# Attendance helpers
def load_attendance():
    df = None

    if os.path.exists(ATTENDANCE_PATH):
        try:
            df = pd.read_excel(ATTENDANCE_PATH)
        except Exception:
            df = None

    if df is None and os.path.exists(ATTENDANCE_CSV_PATH):
        try:
            df = pd.read_csv(ATTENDANCE_CSV_PATH)
        except Exception:
            df = None

    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "rollno",
                "name",
                "branch",
                "date",
                "time",
                "timestamp",
                "status"
            ]
        )

    df.columns = [
        str(col).strip().lower().replace(" ", "_")
        for col in df.columns
    ]

    column_map = {}

    for col in df.columns:
        if col in ["roll_no", "roll", "rollno", "student_id", "id"]:
            column_map[col] = "rollno"

        elif col in ["name", "student_name", "fullname"]:
            column_map[col] = "name"

        elif col in [
            "branch",
            "department",
            "dept",
            "course",
            "stream"
        ]:
            column_map[col] = "branch"

        elif col in ["date", "attendance_date"]:
            column_map[col] = "date"

        elif col in ["time", "attendance_time"]:
            column_map[col] = "time"

        elif col in [
            "timestamp",
            "datetime",
            "date_time",
            "attendance_timestamp"
        ]:
            column_map[col] = "timestamp"

        elif col in ["status", "attendance_status"]:
            column_map[col] = "status"

    df = df.rename(columns=column_map)

    for required in [
        "rollno",
        "name",
        "branch",
        "date",
        "time",
        "timestamp",
        "status"
    ]:
        if required not in df.columns:
            df[required] = ""

    df = df[
        [
            "rollno",
            "name",
            "branch",
            "date",
            "time",
            "timestamp",
            "status"
        ]
    ].copy()

    df["rollno"] = df["rollno"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df["branch"] = df["branch"].astype(str).str.strip()
    df["status"] = df["status"].astype(str).str.strip()

    return df.reset_index(drop=True)


def save_attendance(df):
    df = df[
        [
            "rollno",
            "name",
            "branch",
            "date",
            "time",
            "timestamp",
            "status"
        ]
    ].copy()

    df.to_excel(
        ATTENDANCE_PATH,
        index=False
    )


# Face recognition system
@st.cache_resource
def load_face_system():

    cascade = cv2.CascadeClassifier(CASCADE_PATH)

    recognizer = None

    if hasattr(cv2, "face"):
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        if os.path.exists(TRAINER_PATH):
            try:
                recognizer.read(TRAINER_PATH)
            except Exception:
                recognizer = cv2.face.LBPHFaceRecognizer_create()

    return cascade, recognizer


# Face detection
def detect_faces(image, cascade):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(70, 70)
    )

    valid_faces = []

    for x, y, w, h in faces:

        area = w * h

        if area >= 5000:
            valid_faces.append(
                (x, y, w, h)
            )

    return gray, valid_faces


# Attendance marking
def mark_attendance(
    rollno,
    name,
    branch
):

    attendance_df = load_attendance()

    current = now_ist()

    if not attendance_df.empty:

        attendance_df["parsed_timestamp"] = pd.to_datetime(
            attendance_df["timestamp"],
            errors="coerce"
        )

        student_records = attendance_df[
            attendance_df["rollno"].astype(str).str.strip()
            == str(rollno).strip()
        ]

        student_records = student_records[
            student_records["parsed_timestamp"].notna()
        ]

        if not student_records.empty:

            last_attendance = student_records[
                "parsed_timestamp"
            ].max()

            if pd.notna(last_attendance):

                last_attendance = last_attendance.to_pydatetime()

                if last_attendance.tzinfo is not None:
                    last_attendance = last_attendance.astimezone(
                        IST
                    ).replace(tzinfo=None)

                difference = current - last_attendance

                if difference < timedelta(hours=1):

                    remaining = timedelta(hours=1) - difference

                    minutes = int(
                        remaining.total_seconds() // 60
                    )

                    seconds = int(
                        remaining.total_seconds() % 60
                    )

                    attendance_df.drop(
                        columns=["parsed_timestamp"],
                        inplace=True,
                        errors="ignore"
                    )

                    return {
                        "marked": False,
                        "message": (
                            f"Attendance already marked. "
                            f"Try again after {minutes}m {seconds}s."
                        ),
                        "time": current
                    }

        attendance_df.drop(
            columns=["parsed_timestamp"],
            inplace=True,
            errors="ignore"
        )

    new_record = {
        "rollno": str(rollno),
        "name": str(name),
        "branch": str(branch),
        "date": current.strftime("%Y-%m-%d"),
        "time": current.strftime("%I:%M:%S %p"),
        "timestamp": current.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "Present"
    }

    attendance_df = pd.concat(
        [
            attendance_df,
            pd.DataFrame([new_record])
        ],
        ignore_index=True
    )

    save_attendance(attendance_df)

    return {
        "marked": True,
        "message": "Attendance marked successfully.",
        "time": current
    }


# Train/update LBPH model
def train_new_student(rollno):

    student_dir = os.path.join(
        DATASET_PATH,
        str(rollno)
    )

    image_paths = []

    if os.path.exists(student_dir):

        for filename in os.listdir(student_dir):

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                image_paths.append(
                    os.path.join(
                        student_dir,
                        filename
                    )
                )

    if not image_paths:
        return False, "No face samples found."

    if not hasattr(cv2, "face"):
        return False, "OpenCV face module is unavailable."

    images = []
    labels = []

    for image_path in image_paths:

        img = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            continue

        img = cv2.equalizeHist(img)
        img = cv2.resize(img, (200, 200))

        images.append(img)
        labels.append(int(rollno))

    if not images:
        return False, "Face images could not be loaded."

    try:

        if os.path.exists(TRAINER_PATH):

            recognizer = cv2.face.LBPHFaceRecognizer_create()

            recognizer.read(TRAINER_PATH)

            recognizer.update(
                images,
                np.array(labels, dtype=np.int32)
            )

        else:

            recognizer = cv2.face.LBPHFaceRecognizer_create()

            recognizer.train(
                images,
                np.array(labels, dtype=np.int32)
            )

        recognizer.write(TRAINER_PATH)

        return True, "Face model updated successfully."

    except Exception as e:

        return False, str(e)


# Navigation
with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🎓</div>
            <div class="brand-title">Smart Attendance</div>
            <div class="brand-subtitle">
                Face Recognition System
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Mark Attendance",
            "Attendance Records",
            "Students",
            "Register Student",
            "About"
        ]
    )

    st.markdown("---")

    current = now_ist()

    st.markdown(
        f"""
        <div class="info-box">
            <b>System Status</b><br><br>
            <span class="status-online">● ONLINE</span><br><br>
            India Time<br>
            <b>{current.strftime("%d %b %Y")}</b><br>
            {current.strftime("%I:%M:%S %p")}
        </div>
        """,
        unsafe_allow_html=True
    )


# Load data
students_df = load_students()
attendance_df = load_attendance()


# Dashboard
if page == "Dashboard":

    current = now_ist()

    st.markdown(
        f"""
        <div class="live-clock">
            🇮🇳 India Standard Time &nbsp; | &nbsp;
            {current.strftime("%d %b %Y, %I:%M:%S %p")}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                Smart Attendance System
            </div>
            <div class="hero-text">
                Automated attendance using face recognition,
                secure student records and one-hour duplicate protection.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    today = current.strftime("%Y-%m-%d")

    if not attendance_df.empty:

        today_records = attendance_df[
            attendance_df["date"].astype(str).str.strip()
            == today
        ]

        present_today = today_records[
            today_records["status"].str.lower()
            == "present"
        ]

    else:

        today_records = pd.DataFrame()
        present_today = pd.DataFrame()

    total_students = len(students_df)
    total_records = len(attendance_df)
    today_count = len(
        present_today["rollno"].unique()
    ) if not present_today.empty else 0

    attendance_rate = (
        (today_count / total_students) * 100
        if total_students > 0
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    👥 Total Students
                </div>
                <div class="metric-value">
                    {total_students}
                </div>
                <div class="metric-small">
                    Registered students
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    ✅ Present Today
                </div>
                <div class="metric-value">
                    {today_count}
                </div>
                <div class="metric-small">
                    Unique students
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    📊 Attendance Rate
                </div>
                <div class="metric-value">
                    {attendance_rate:.1f}%
                </div>
                <div class="metric-small">
                    Today's attendance
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    📝 Total Records
                </div>
                <div class="metric-value">
                    {total_records}
                </div>
                <div class="metric-small">
                    All attendance records
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-card">',
        unsafe_allow_html=True
    )

    st.subheader("Today's Attendance")

    if not today_records.empty:

        display_df = today_records[
            [
                "rollno",
                "name",
                "branch",
                "date",
                "time",
                "status"
            ]
        ].copy()

        display_df.columns = [
            "Roll No",
            "Name",
            "Branch",
            "Date",
            "Time",
            "Status"
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No attendance has been marked today."
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# Mark Attendance
elif page == "Mark Attendance":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                📷 Mark Attendance
            </div>
            <div class="hero-text">
                Capture one face. The system will identify the
                registered student and apply the one-hour rule.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    left, right = st.columns(
        [1.25, 0.75]
    )

    with left:

        st.markdown(
            "### Face Scanner"
        )

        camera_image = st.camera_input(
            "Capture your face",
            key="attendance_camera"
        )

        if camera_image is not None:

            image_bytes = camera_image.getvalue()

            image_array = np.frombuffer(
                image_bytes,
                dtype=np.uint8
            )

            frame = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )

            if frame is None:

                st.error(
                    "Unable to read camera image."
                )

            else:

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                cascade, recognizer = load_face_system()

                if cascade.empty():

                    st.error(
                        "Face detection model not found."
                    )

                elif recognizer is None:

                    st.error(
                        "Face recognition model is not available. "
                        "Please register a student first."
                    )

                else:

                    gray, faces = detect_faces(
                        frame,
                        cascade
                    )

                    if len(faces) == 0:

                        st.error(
                            "❌ No face detected. "
                            "Please look directly at the camera."
                        )

                    else:

                        # Select the largest face
                        face = max(
                            faces,
                            key=lambda f: f[2] * f[3]
                        )

                        x, y, w, h = face

                        face_gray = gray[
                            y:y+h,
                            x:x+w
                        ]

                        if face_gray.size == 0:

                            st.error(
                                "Unable to process the detected face."
                            )

                        else:

                            face_gray = cv2.resize(
                                face_gray,
                                (200, 200)
                            )

                            try:

                                label, confidence = recognizer.predict(
                                    face_gray
                                )

                                students_df = load_students()

                                rollno = str(int(label))

                                matched = students_df[
                                    students_df["rollno"].astype(str).str.strip()
                                    == rollno
                                ]

                                if (
                                    not matched.empty
                                    and confidence < 85
                                ):

                                    student = matched.iloc[0]

                                    name = student["name"]
                                    branch = student["branch"]

                                    # Draw face rectangle
                                    cv2.rectangle(
                                        frame,
                                        (x, y),
                                        (x + w, y + h),
                                        (0, 220, 140),
                                        3
                                    )

                                    label_text = (
                                        f"{name} | {rollno}"
                                    )

                                    text_y = max(
                                        y - 12,
                                        25
                                    )

                                    text_width = min(
                                        w + 100,
                                        frame.shape[1] - x
                                    )

                                    cv2.rectangle(
                                        frame,
                                        (
                                            x,
                                            max(0, y - 45)
                                        ),
                                        (
                                            x + text_width,
                                            y
                                        ),
                                        (8, 25, 40),
                                        -1
                                    )

                                    cv2.putText(
                                        frame,
                                        label_text,
                                        (
                                            x + 8,
                                            text_y
                                        ),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.65,
                                        (255, 255, 255),
                                        2,
                                        cv2.LINE_AA
                                    )

                                    st.image(
                                        frame,
                                        caption="Recognized Face",
                                        use_container_width=True
                                    )

                                    result = mark_attendance(
                                        rollno,
                                        name,
                                        branch
                                    )

                                    current_time = result["time"]

                                    if result["marked"]:

                                        st.markdown(
                                            f"""
                                            <div class="result-card recognized-card">
                                                <div class="result-title">
                                                    ✅ Student Recognized
                                                </div>

                                                <div class="result-row">
                                                    <span class="result-key">
                                                        Name
                                                    </span>
                                                    <span class="result-value">
                                                        {name}
                                                    </span>
                                                </div>

                                                <div class="result-row">
                                                    <span class="result-key">
                                                        Roll No
                                                    </span>
                                                    <span class="result-value">
                                                        {rollno}
                                                    </span>
                                                </div>

                                                <div class="result-row">
                                                    <span class="result-key">
                                                        Branch
                                                    </span>
                                                    <span class="result-value">
                                                        {branch}
                                                    </span>
                                                </div>

                                                <div class="result-row">
                                                    <span class="result-key">
                                                        Match Distance
                                                    </span>
                                                    <span class="result-value">
                                                        {confidence:.2f}
                                                    </span>
                                                </div>

                                                <div class="result-row">
                                                    <span class="result-key">
                                                        Attendance
                                                    </span>
                                                    <span class="result-value">
                                                        ✅ Marked
                                                    </span>
                                                </div>

                                                <div class="result-row">
                                                    <span class="result-key">
                                                        Current Time
                                                    </span>
                                                    <span class="result-value">
                                                        {current_time.strftime("%I:%M:%S %p")}
                                                    </span>
                                                </div>
                                            </div>
                                            """,
                                            unsafe_allow_html=True
                                        )

                                        st.success(
                                            "Attendance saved successfully."
                                        )

                                    else:

                                        st.markdown(
                                            f"""
                                            <div class="result-card warning-card">
                                                <div class="result-title">
                                                    ⏱ Attendance Already Marked
                                                </div>

                                                <div class="result-row">
                                                    <span class="result-key">
                                                        Student
                                                    </span>
                                                    <span class="result-value">
                                                        {name}
                                                    </span>
                                                </div>

                                                <div class="result-row">
                                                    <span class="result-key">
                                                        Roll No
                                                    </span>
                                                    <span class="result-value">
                                                        {rollno}
                                                    </span>
                                                </div>

                                                <div class="result-row">
                                                    <span class="result-key">
                                                        Current Time
                                                    </span>
                                                    <span class="result-value">
                                                        {current_time.strftime("%I:%M:%S %p")}
                                                    </span>
                                                </div>
                                            </div>
                                            """,
                                            unsafe_allow_html=True
                                        )

                                        st.warning(
                                            result["message"]
                                        )

                                else:

                                    # Unknown face
                                    cv2.rectangle(
                                        frame,
                                        (x, y),
                                        (x + w, y + h),
                                        (255, 90, 90),
                                        3
                                    )

                                    cv2.putText(
                                        frame,
                                        "Unknown Face",
                                        (
                                            x,
                                            max(y - 12, 25)
                                        ),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.75,
                                        (255, 90, 90),
                                        2,
                                        cv2.LINE_AA
                                    )

                                    st.image(
                                        frame,
                                        caption="Face not registered",
                                        use_container_width=True
                                    )

                                    st.markdown(
                                        """
                                        <div class="result-card error-card">
                                            <div class="result-title">
                                                ❌ Face Not Registered
                                            </div>

                                            <div class="info-box">
                                                This face could not be matched
                                                with a registered student.
                                                Please register the student
                                                before marking attendance.
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

                            except Exception as e:

                                st.error(
                                    f"Face recognition error: {e}"
                                )

    with right:

        st.markdown(
            "### How it works"
        )

        st.markdown(
            """
            <div class="section-card">

            <b>1. Capture</b><br>
            Take one clear image using the camera.

            <br><br>

            <b>2. Detect</b><br>
            Haar Cascade detects the face.

            <br><br>

            <b>3. Recognize</b><br>
            LBPH compares the face with registered students.

            <br><br>

            <b>4. Verify</b><br>
            Recognition threshold is checked.

            <br><br>

            <b>5. Attendance</b><br>
            Current India time is stored with the attendance record.

            <br><br>

            <b>6. Duplicate Protection</b><br>
            The same student cannot be marked again within one hour.

            </div>
            """,
            unsafe_allow_html=True
        )


# Attendance Records
elif page == "Attendance Records":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                📋 Attendance Records
            </div>
            <div class="hero-text">
                View and search all attendance records.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    attendance_df = load_attendance()

    if attendance_df.empty:

        st.info(
            "No attendance records available."
        )

    else:

        search = st.text_input(
            "🔎 Search by name or roll number",
            placeholder="Enter student name or roll number..."
        )

        filtered_df = attendance_df.copy()

        if search.strip():

            query = search.strip().lower()

            filtered_df = filtered_df[
                filtered_df["name"]
                .astype(str)
                .str.lower()
                .str.contains(query, na=False)
                |
                filtered_df["rollno"]
                .astype(str)
                .str.lower()
                .str.contains(query, na=False)
            ]

        display_df = filtered_df[
            [
                "rollno",
                "name",
                "branch",
                "date",
                "time",
                "status"
            ]
        ].copy()

        display_df.columns = [
            "Roll No",
            "Name",
            "Branch",
            "Date",
            "Time",
            "Status"
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"Showing {len(display_df)} record(s)"
        )


# Students
elif page == "Students":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                👥 Registered Students
            </div>
            <div class="hero-text">
                Students currently available for face recognition.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    students_df = load_students()

    if students_df.empty:

        st.info(
            "No students registered yet."
        )

    else:

        display_df = students_df.copy()

        display_df.columns = [
            "Roll No",
            "Name",
            "Branch"
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


# Register Student
elif page == "Register Student":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                ➕ Register Student
            </div>
            <div class="hero-text">
                Register a student and create their face recognition profile.
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
            placeholder="Example: 11232743"
        )

        name = st.text_input(
            "Student Name",
            placeholder="Enter full name"
        )

        branch = st.text_input(
            "Branch",
            placeholder="Example: Computer Science Engineering"
        )

        st.markdown(
            """
            <div class="info-box">
                Use a clear front-facing image.
                Only one person should be visible in the camera.
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            "### Face Capture"
        )

        registration_image = st.camera_input(
            "Capture student face",
            key="registration_camera"
        )

    if st.button(
        "Register Student",
        type="primary",
        use_container_width=True
    ):

        students_df = load_students()

        rollno = rollno.strip()
        name = name.strip()
        branch = branch.strip()

        if not rollno or not name or not branch:

            st.error(
                "Please fill all student details."
            )

        elif not rollno.isdigit():

            st.error(
                "Roll number must contain only numbers."
            )

        elif rollno in students_df["rollno"].astype(str).values:

            st.error(
                "This roll number is already registered."
            )

        elif registration_image is None:

            st.error(
                "Please capture the student's face first."
            )

        else:

            image_bytes = registration_image.getvalue()

            image_array = np.frombuffer(
                image_bytes,
                dtype=np.uint8
            )

            frame = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )

            if frame is None:

                st.error(
                    "Unable to read camera image."
                )

            else:

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                cascade, _ = load_face_system()

                gray, faces = detect_faces(
                    frame,
                    cascade
                )

                if len(faces) == 0:

                    st.error(
                        "❌ No face detected. "
                        "Please capture a clearer image."
                    )

                else:

                    # Select the largest face
                    x, y, w, h = max(
                        faces,
                        key=lambda f: f[2] * f[3]
                    )

                    face_crop = gray[
                        y:y+h,
                        x:x+w
                    ]

                    if face_crop.size == 0:

                        st.error(
                            "Unable to extract the face."
                        )

                    else:

                        face_crop = cv2.equalizeHist(
                            face_crop
                        )

                        face_crop = cv2.resize(
                            face_crop,
                            (200, 200)
                        )

                        student_dir = os.path.join(
                            DATASET_PATH,
                            rollno
                        )

                        os.makedirs(
                            student_dir,
                            exist_ok=True
                        )

                        # Create a few variations from one capture
                        samples = []

                        samples.append(
                            face_crop
                        )

                        samples.append(
                            cv2.flip(
                                face_crop,
                                1
                            )
                        )

                        bright = cv2.convertScaleAbs(
                            face_crop,
                            alpha=1.08,
                            beta=8
                        )

                        dark = cv2.convertScaleAbs(
                            face_crop,
                            alpha=0.92,
                            beta=-8
                        )

                        samples.append(bright)
                        samples.append(dark)

                        for index, sample in enumerate(
                            samples,
                            start=1
                        ):

                            filename = os.path.join(
                                student_dir,
                                f"User.{index}.{rollno}.jpg"
                            )

                            cv2.imwrite(
                                filename,
                                sample
                            )

                        new_student = pd.DataFrame(
                            [
                                {
                                    "rollno": rollno,
                                    "name": name,
                                    "branch": branch
                                }
                            ]
                        )

                        students_df = pd.concat(
                            [
                                students_df,
                                new_student
                            ],
                            ignore_index=True
                        )

                        save_students(
                            students_df
                        )

                        success, message = train_new_student(
                            rollno
                        )

                        if success:

                            st.cache_resource.clear()

                            st.success(
                                "✅ Student registered and face model updated successfully."
                            )

                            st.info(
                                f"Student: {name} | "
                                f"Roll No: {rollno}"
                            )

                            st.rerun()

                        else:

                            st.warning(
                                f"Student saved, but model training failed: {message}"
                            )


# About
elif page == "About":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                ℹ️ About Smart Attendance
            </div>
            <div class="hero-text">
                Face recognition based student attendance management system.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="section-card">

            <h3>Technology Stack</h3>

            <b>Frontend / UI</b><br>
            Streamlit

            <br><br>

            <b>Computer Vision</b><br>
            OpenCV + Haar Cascade

            <br><br>

            <b>Face Recognition</b><br>
            LBPH Face Recognizer

            <br><br>

            <b>Data Processing</b><br>
            Python + Pandas

            <br><br>

            <b>Storage</b><br>
            CSV + Excel

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="section-card">

            <h3>Attendance Rules</h3>

            <b>Face Detection</b><br>
            Detects the primary face from the captured image.

            <br><br>

            <b>Face Recognition</b><br>
            LBPH compares the captured face with registered profiles.

            <br><br>

            <b>Recognition Threshold</b><br>
            Matching distance below 85 is accepted.

            <br><br>

            <b>Duplicate Protection</b><br>
            A student cannot receive another attendance record
            within one hour.

            <br><br>

            <b>Timezone</b><br>
            Attendance timestamps use Asia/Kolkata (IST).

            </div>
            """,
            unsafe_allow_html=True
        )


# Footer
st.markdown(
    """
    <div class="footer">
        Smart Attendance System · Face Recognition · IST Attendance
    </div>
    """,
    unsafe_allow_html=True
)
