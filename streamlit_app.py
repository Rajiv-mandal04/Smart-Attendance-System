import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import pandas as pd
import streamlit as st


# Page configuration
st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STUDENT_PATH = os.path.join(BASE_DIR, "data", "students.csv")
ATTENDANCE_PATH = os.path.join(BASE_DIR, "attendance", "attendance.xlsx")
TRAINER_PATH = os.path.join(BASE_DIR, "trainer", "trainer.yml")
CASCADE_PATH = os.path.join(
    BASE_DIR,
    "haarcascade",
    "haarcascade_frontalface_default.xml",
)
DATASET_PATH = os.path.join(BASE_DIR, "dataset")


# India timezone
IST = ZoneInfo("Asia/Kolkata")


def now_ist():
    return datetime.now(IST).replace(tzinfo=None)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #0b1220;
        color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background: #111827;
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    .hero {
        padding: 25px 30px;
        border-radius: 18px;
        background: linear-gradient(135deg, #172033, #111827);
        border: 1px solid #263247;
        margin-bottom: 22px;
    }

    .hero-title {
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 15px;
    }

    .metric-card {
        background: #111827;
        border: 1px solid #263247;
        border-radius: 15px;
        padding: 20px;
        min-height: 125px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 14px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 800;
        margin-top: 8px;
    }

    .success-box {
        padding: 16px;
        border-radius: 12px;
        background: #0f2d22;
        border: 1px solid #1f6b4d;
        color: #bbf7d0;
    }

    .warning-box {
        padding: 16px;
        border-radius: 12px;
        background: #30260d;
        border: 1px solid #80651b;
        color: #fde68a;
    }

    .danger-box {
        padding: 16px;
        border-radius: 12px;
        background: #32151a;
        border: 1px solid #7f2934;
        color: #fecaca;
    }

    .info-box {
        padding: 16px;
        border-radius: 12px;
        background: #10253b;
        border: 1px solid #28527c;
        color: #bfdbfe;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# STUDENT DATA
# ============================================================

def load_students():

    columns = [
        "rollno",
        "name",
        "branch",
    ]

    if not os.path.exists(STUDENT_PATH):
        return pd.DataFrame(columns=columns)

    try:

        df = pd.read_csv(
            STUDENT_PATH,
            sep=None,
            engine="python",
            dtype=str,
        )

        df.columns = [
            str(col)
            .strip()
            .lower()
            .replace(" ", "_")
            for col in df.columns
        ]

        rename_map = {}

        for col in df.columns:

            if col in [
                "roll",
                "roll_no",
                "student_id",
                "id",
            ]:
                rename_map[col] = "rollno"

            elif col in [
                "student_name",
                "full_name",
                "fullname",
            ]:
                rename_map[col] = "name"

            elif col in [
                "department",
                "dept",
                "course",
                "stream",
            ]:
                rename_map[col] = "branch"

        df = df.rename(
            columns=rename_map
        )

        for col in columns:

            if col not in df.columns:
                df[col] = ""

        return df[columns].fillna("")

    except Exception:

        return pd.DataFrame(columns=columns)


# ============================================================
# ATTENDANCE DATA
# ============================================================

def load_attendance():

    columns = [
        "rollno",
        "name",
        "branch",
        "date",
        "time",
        "timestamp",
        "status",
    ]

    if not os.path.exists(ATTENDANCE_PATH):
        return pd.DataFrame(columns=columns)

    try:

        df = pd.read_excel(
            ATTENDANCE_PATH
        )

        if df.empty:
            return pd.DataFrame(columns=columns)

        rename_map = {}

        for col in df.columns:

            clean = (
                str(col)
                .strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )

            if clean in [
                "rollno",
                "roll_no",
                "roll",
                "student_id",
                "id",
            ]:
                rename_map[col] = "rollno"

            elif clean in [
                "name",
                "student_name",
                "fullname",
                "full_name",
            ]:
                rename_map[col] = "name"

            elif clean in [
                "branch",
                "department",
                "dept",
                "course",
                "stream",
            ]:
                rename_map[col] = "branch"

            elif clean in [
                "date",
                "attendance_date",
            ]:
                rename_map[col] = "date"

            elif clean in [
                "time",
                "attendance_time",
            ]:
                rename_map[col] = "time"

            elif clean in [
                "timestamp",
                "datetime",
                "date_time",
                "attendance_timestamp",
            ]:
                rename_map[col] = "timestamp"

            elif clean in [
                "status",
                "attendance_status",
            ]:
                rename_map[col] = "status"

        df = df.rename(
            columns=rename_map
        )

        for col in columns:

            if col not in df.columns:
                df[col] = ""

        return df[columns].fillna("")

    except Exception:

        return pd.DataFrame(columns=columns)


# ============================================================
# SAVE ATTENDANCE
# ============================================================

def save_attendance(df):

    os.makedirs(
        os.path.dirname(ATTENDANCE_PATH),
        exist_ok=True,
    )

    df.to_excel(
        ATTENDANCE_PATH,
        index=False,
    )


# ============================================================
# SAVE STUDENT
# ============================================================

def save_student(
    rollno,
    name,
    branch,
):

    os.makedirs(
        os.path.dirname(STUDENT_PATH),
        exist_ok=True,
    )

    students = load_students()

    new_student = pd.DataFrame(
        [
            {
                "rollno": str(rollno),
                "name": str(name),
                "branch": str(branch),
            }
        ]
    )

    students = pd.concat(
        [
            students,
            new_student,
        ],
        ignore_index=True,
    )

    students.to_csv(
        STUDENT_PATH,
        sep="\t",
        index=False,
    )


# ============================================================
# FIND STUDENT
# ============================================================

def get_student(rollno):

    students = load_students()

    if students.empty:
        return None

    students["rollno"] = (
        students["rollno"]
        .astype(str)
        .str.strip()
    )

    result = students[
        students["rollno"]
        == str(rollno).strip()
    ]

    if result.empty:
        return None

    return result.iloc[0]


# ============================================================
# ATTENDANCE MARKING
# ============================================================

def mark_attendance(
    rollno,
    name,
    branch,
):

    attendance_df = load_attendance()

    current_time = now_ist()

    if not attendance_df.empty:

        attendance_df["rollno"] = (
            attendance_df["rollno"]
            .astype(str)
            .str.strip()
        )

        student_records = attendance_df[
            attendance_df["rollno"]
            == str(rollno).strip()
        ].copy()

        if not student_records.empty:

            timestamps = pd.to_datetime(
                student_records["timestamp"],
                errors="coerce",
            ).dropna()

            if not timestamps.empty:

                last_time = (
                    timestamps
                    .max()
                    .to_pydatetime()
                )

                difference = (
                    current_time
                    - last_time
                )

                if (
                    difference >= timedelta(0)
                    and difference
                    < timedelta(hours=1)
                ):

                    remaining = (
                        timedelta(hours=1)
                        - difference
                    )

                    minutes = int(
                        remaining.total_seconds()
                        // 60
                    )

                    seconds = int(
                        remaining.total_seconds()
                        % 60
                    )

                    return (
                        False,
                        (
                            "Attendance already marked. "
                            f"Try again after "
                            f"{minutes}m {seconds}s."
                        ),
                    )

    new_record = {
        "rollno": str(rollno),
        "name": str(name),
        "branch": str(branch),
        "date": current_time.strftime(
            "%Y-%m-%d"
        ),
        "time": current_time.strftime(
            "%I:%M:%S %p"
        ),
        "timestamp": current_time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "status": "Present",
    }

    attendance_df = pd.concat(
        [
            attendance_df,
            pd.DataFrame(
                [new_record]
            ),
        ],
        ignore_index=True,
    )

    save_attendance(
        attendance_df
    )

    return (
        True,
        (
            "Attendance marked successfully at "
            f"{current_time.strftime('%I:%M:%S %p')} IST."
        ),
    )


# ============================================================
# FACE CASCADE
# ============================================================

@st.cache_resource
def load_face_cascade():

    if not os.path.exists(
        CASCADE_PATH
    ):
        return None

    cascade = cv2.CascadeClassifier(
        CASCADE_PATH
    )

    if cascade.empty():
        return None

    return cascade


# ============================================================
# LBPH MODEL
# ============================================================

@st.cache_resource
def load_recognizer():

    if not os.path.exists(
        TRAINER_PATH
    ):
        return None

    if not hasattr(
        cv2,
        "face",
    ):
        return None

    try:

        recognizer = (
            cv2.face
            .LBPHFaceRecognizer_create()
        )

        recognizer.read(
            TRAINER_PATH
        )

        return recognizer

    except Exception:

        return None


# ============================================================
# FACE DETECTION
# ============================================================

def detect_single_face(
    frame,
    cascade,
):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2GRAY,
    )

    gray = cv2.equalizeHist(
        gray
    )

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(70, 70),
    )

    valid_faces = []

    for x, y, w, h in faces:

        area = w * h

        if area >= 5000:

            valid_faces.append(
                (x, y, w, h)
            )

    if not valid_faces:

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(40, 40),
        )

        for x, y, w, h in faces:

            area = w * h

            if area >= 3000:

                valid_faces.append(
                    (x, y, w, h)
                )

    if not valid_faces:
        return None, gray

    face = max(
        valid_faces,
        key=lambda item:
        item[2] * item[3],
    )

    return face, gray


# ============================================================
# FACE RECOGNITION
# ============================================================

def recognize_face(
    image_bytes,
):

    cascade = load_face_cascade()

    recognizer = load_recognizer()

    if cascade is None:

        return (
            None,
            None,
            "Face cascade not found.",
        )

    if recognizer is None:

        return (
            None,
            None,
            "Trained face model not found.",
        )

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if frame is None:

        return (
            None,
            None,
            "Unable to read captured image.",
        )

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    face, gray = detect_single_face(
        frame,
        cascade,
    )

    if face is None:

        return (
            frame,
            None,
            "No face detected.",
        )

    x, y, w, h = face

    roi = gray[
        y:y + h,
        x:x + w,
    ]

    if roi.size == 0:

        return (
            frame,
            None,
            "Face region is empty.",
        )

    roi = cv2.resize(
        roi,
        (200, 200),
    )

    try:

        label, confidence = (
            recognizer.predict(roi)
        )

    except Exception as e:

        return (
            frame,
            None,
            f"Recognition error: {e}",
        )

    if confidence >= 85:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (220, 50, 50),
            3,
        )

        cv2.putText(
            frame,
            "Unknown / Not Registered",
            (
                x,
                max(y - 12, 25),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (220, 50, 50),
            2,
        )

        return (
            frame,
            None,
            (
                "Face detected, but student "
                "not recognized."
            ),
        )

    student = get_student(
        label
    )

    if student is None:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (220, 50, 50),
            3,
        )

        cv2.putText(
            frame,
            "Unknown Student",
            (
                x,
                max(y - 12, 25),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (220, 50, 50),
            2,
        )

        return (
            frame,
            None,
            (
                f"Recognized ID {label}, "
                "but no student record exists."
            ),
        )

    rollno = str(
        student["rollno"]
    )

    name = str(
        student["name"]
    )

    branch = str(
        student["branch"]
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        (40, 200, 90),
        3,
    )

    cv2.putText(
        frame,
        name,
        (
            x,
            max(y - 38, 25),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (40, 200, 90),
        2,
    )

    cv2.putText(
        frame,
        f"Roll No: {rollno}",
        (
            x,
            max(y - 10, 50),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (40, 200, 90),
        2,
    )

    return (
        frame,
        {
            "rollno": rollno,
            "name": name,
            "branch": branch,
            "confidence": confidence,
        },
        "Student recognized.",
    )


# ============================================================
# TRAIN MODEL AFTER NEW REGISTRATION
# ============================================================

def train_model():

    if not hasattr(
        cv2,
        "face",
    ):
        return False, "OpenCV face module not available."

    if not os.path.exists(
        DATASET_PATH
    ):
        return False, "Dataset folder not found."

    images = []
    labels = []

    for roll_folder in os.listdir(
        DATASET_PATH
    ):

        folder_path = os.path.join(
            DATASET_PATH,
            roll_folder,
        )

        if not os.path.isdir(
            folder_path
        ):
            continue

        try:
            label = int(
                roll_folder
            )
        except ValueError:
            continue

        for filename in os.listdir(
            folder_path
        ):

            if not filename.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                )
            ):
                continue

            image_path = os.path.join(
                folder_path,
                filename,
            )

            image = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE,
            )

            if image is None:
                continue

            image = cv2.resize(
                image,
                (200, 200),
            )

            images.append(
                image
            )

            labels.append(
                label
            )

    if not images:

        return (
            False,
            "No training images found.",
        )

    recognizer = (
        cv2.face
        .LBPHFaceRecognizer_create()
    )

    recognizer.train(
        images,
        np.array(
            labels,
            dtype=np.int32,
        ),
    )

    os.makedirs(
        os.path.dirname(
            TRAINER_PATH
        ),
        exist_ok=True,
    )

    recognizer.write(
        TRAINER_PATH
    )

    load_recognizer.clear()

    return (
        True,
        f"Model trained using {len(images)} face samples.",
    )


# ============================================================
# DASHBOARD
# ============================================================

def dashboard():

    students = load_students()
    attendance = load_attendance()

    today = now_ist().strftime(
        "%Y-%m-%d"
    )

    if attendance.empty:

        today_attendance = (
            pd.DataFrame()
        )

    else:

        attendance["date"] = (
            attendance["date"]
            .astype(str)
            .str[:10]
        )

        today_attendance = attendance[
            attendance["date"] == today
        ]

    total_students = len(
        students
    )

    present_today = (
        today_attendance["rollno"]
        .nunique()
        if not today_attendance.empty
        else 0
    )

    attendance_rate = (
        (
            present_today
            / total_students
        )
        * 100
        if total_students > 0
        else 0
    )

    total_records = len(
        attendance
    )

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                🎓 Smart Attendance System
            </div>
            <div class="hero-subtitle">
                Face Recognition based attendance management
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current = now_ist()

    st.caption(
        f"🕒 Current Time: "
        f"{current.strftime('%d %B %Y, %I:%M:%S %p')} IST"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Total Students
                </div>
                <div class="metric-value">
                    {total_students}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Present Today
                </div>
                <div class="metric-value">
                    {present_today}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Attendance Rate
                </div>
                <div class="metric-value">
                    {attendance_rate:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Total Records
                </div>
                <div class="metric-value">
                    {total_records}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">'
        "Today's Attendance"
        "</div>",
        unsafe_allow_html=True,
    )

    if today_attendance.empty:

        st.info(
            "No attendance marked today."
        )

    else:

        display_df = (
            today_attendance[
                [
                    "rollno",
                    "name",
                    "branch",
                    "date",
                    "time",
                    "status",
                ]
            ]
            .copy()
        )

        display_df.columns = [
            "Roll No",
            "Name",
            "Branch",
            "Date",
            "Time",
            "Status",
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# MARK ATTENDANCE
# ============================================================

def mark_attendance_page():

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                📷 Mark Attendance
            </div>
            <div class="hero-subtitle">
                Capture your face and let the system recognize you.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-box">
            Keep only one person in front of the camera.
            Make sure your face is clearly visible and well illuminated.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    captured_image = st.camera_input(
        "Capture Face",
        key="attendance_camera",
    )

    if captured_image is None:
        return

    with st.spinner(
        "Recognizing face..."
    ):

        annotated_frame, student, message = (
            recognize_face(
                captured_image.getvalue()
            )
        )

    st.image(
        annotated_frame,
        caption="Recognition Result",
        use_container_width=True,
    )

    if student is None:

        if "No face detected" in message:

            st.markdown(
                """
                <div class="warning-box">
                    ⚠️ No face detected. Please face the camera
                    properly and capture again.
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif (
            "Unknown" in message
            or "not recognized" in message
        ):

            st.markdown(
                f"""
                <div class="danger-box">
                    ❌ {message}
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.error(
                message
            )

        return

    st.success(
        f"Recognized: {student['name']} "
        f"(Roll No: {student['rollno']})"
    )

    st.write(
        f"**Branch:** {student['branch']}"
    )

    success, result_message = (
        mark_attendance(
            student["rollno"],
            student["name"],
            student["branch"],
        )
    )

    if success:

        st.markdown(
            f"""
            <div class="success-box">
                ✅ {result_message}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.balloons()

    else:

        st.markdown(
            f"""
            <div class="warning-box">
                ⏱️ {result_message}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# ATTENDANCE RECORDS
# ============================================================

def attendance_records():

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                📋 Attendance Records
            </div>
            <div class="hero-subtitle">
                View all recorded attendance data.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    attendance = load_attendance()

    if attendance.empty:

        st.info(
            "No attendance records found."
        )

        return

    display_df = attendance.copy()

    display_df.columns = [
        "Roll No",
        "Name",
        "Branch",
        "Date",
        "Time",
        "Timestamp",
        "Status",
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = (
        attendance
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "⬇️ Download CSV",
        csv_data,
        file_name="attendance_records.csv",
        mime="text/csv",
    )


# ============================================================
# STUDENTS
# ============================================================

def students_page():

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                👨‍🎓 Students
            </div>
            <div class="hero-subtitle">
                Registered students in the system.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    students = load_students()

    if students.empty:

        st.info(
            "No students registered."
        )

        return

    display_df = students.copy()

    display_df.columns = [
        "Roll No",
        "Name",
        "Branch",
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# REGISTER STUDENT - FIXED
# ============================================================

def register_student():

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                ➕ Register Student
            </div>
            <div class="hero-subtitle">
                Add a new student to the attendance system.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        rollno = st.text_input(
            "Roll Number",
            placeholder="Enter roll number",
            key="register_rollno",
        )

    with col2:

        name = st.text_input(
            "Student Name",
            placeholder="Enter full name",
            key="register_name",
        )

    branch = st.text_input(
        "Branch",
        placeholder="e.g. CSE",
        key="register_branch",
    )

    st.write("")

    st.markdown(
        "### 📷 Capture Student Face"
    )

    st.info(
        "Keep the face clearly visible in the camera. "
        "Only one person should be in front of the camera."
    )

    captured = st.camera_input(
        "Take Student Photo",
        key="registration_camera",
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Store captured image in session state immediately.
    # This prevents Streamlit rerun from losing the image.
    # --------------------------------------------------------

    if captured is not None:

        st.session_state[
            "registration_photo"
        ] = captured.getvalue()

    photo = st.session_state.get(
        "registration_photo"
    )

    if photo is not None:

        st.success(
            "✅ Student photo captured."
        )

        st.image(
            photo,
            caption="Captured Photo",
            use_container_width=True,
        )

    st.write("")

    register_clicked = st.button(
        "➕ Register Student",
        type="primary",
        use_container_width=True,
        key="register_button",
    )

    if not register_clicked:
        return

    # --------------------------------------------------------
    # VALIDATE FORM
    # --------------------------------------------------------

    rollno = rollno.strip()
    name = name.strip()
    branch = branch.strip()

    if not rollno:

        st.warning(
            "Please enter Roll Number."
        )

        return

    if not name:

        st.warning(
            "Please enter Student Name."
        )

        return

    if not branch:

        st.warning(
            "Please enter Branch."
        )

        return

    if photo is None:

        st.warning(
            "📷 Please capture the student's face first."
        )

        return

    # --------------------------------------------------------
    # CHECK DUPLICATE
    # --------------------------------------------------------

    students = load_students()

    if not students.empty:

        students["rollno"] = (
            students["rollno"]
            .astype(str)
            .str.strip()
        )

        existing = students[
            students["rollno"] == rollno
        ]

        if not existing.empty:

            st.error(
                f"Roll number {rollno} is already registered."
            )

            return

    # --------------------------------------------------------
    # DECODE IMAGE
    # --------------------------------------------------------

    image_array = np.frombuffer(
        photo,
        dtype=np.uint8,
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if frame is None:

        st.error(
            "❌ Unable to process captured image."
        )

        return

    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.equalizeHist(
        gray
    )

    cascade = load_face_cascade()

    if cascade is None:

        st.error(
            "❌ Haar Cascade file not found."
        )

        return

    # First detection
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
    )

    # Fallback detection
    if len(faces) == 0:

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30),
        )

    if len(faces) == 0:

        st.error(
            "❌ No face detected in the captured photo."
        )

        st.warning(
            "Try again with better lighting and "
            "keep your face closer to the camera."
        )

        return

    # --------------------------------------------------------
    # SELECT LARGEST FACE
    # --------------------------------------------------------

    x, y, w, h = max(
        faces,
        key=lambda item:
        item[2] * item[3],
    )

    # --------------------------------------------------------
    # SHOW FACE BOX
    # --------------------------------------------------------

    preview = frame.copy()

    cv2.rectangle(
        preview,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        3,
    )

    cv2.putText(
        preview,
        "Face Detected",
        (
            x,
            max(y - 12, 25),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
    )

    preview = cv2.cvtColor(
        preview,
        cv2.COLOR_BGR2RGB,
    )

    st.image(
        preview,
        caption="✅ Face Detected",
        use_container_width=True,
    )

    # --------------------------------------------------------
    # EXTRACT FACE
    # --------------------------------------------------------

    face = gray[
        y:y + h,
        x:x + w,
    ]

    if face.size == 0:

        st.error(
            "❌ Face region could not be extracted."
        )

        return

    face = cv2.resize(
        face,
        (200, 200),
    )

    # --------------------------------------------------------
    # CREATE DATASET FOLDER
    # --------------------------------------------------------

    student_folder = os.path.join(
        DATASET_PATH,
        rollno,
    )

    os.makedirs(
        student_folder,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # GENERATE MULTIPLE SAMPLES
    # --------------------------------------------------------

    existing_files = [
        f
        for f in os.listdir(
            student_folder
        )
        if f.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
            )
        )
    ]

    next_index = (
        len(existing_files) + 1
    )

    samples = []

    # Original
    samples.append(
        face
    )

    # Horizontal flip
    samples.append(
        cv2.flip(face, 1)
    )

    # Slight brightness changes
    brighter = cv2.convertScaleAbs(
        face,
        alpha=1.0,
        beta=15,
    )

    darker = cv2.convertScaleAbs(
        face,
        alpha=1.0,
        beta=-15,
    )

    samples.append(
        brighter
    )

    samples.append(
        darker
    )

    # Save samples
    for index, sample in enumerate(
        samples
    ):

        filename = (
            f"User.{rollno}."
            f"{next_index + index}.jpg"
        )

        image_path = os.path.join(
            student_folder,
            filename,
        )

        cv2.imwrite(
            image_path,
            sample,
        )

    # --------------------------------------------------------
    # ONLY NOW SAVE STUDENT RECORD
    # --------------------------------------------------------

    save_student(
        rollno,
        name,
        branch,
    )

    # --------------------------------------------------------
    # RETRAIN MODEL
    # --------------------------------------------------------

    with st.spinner(
        "Training face recognition model..."
    ):

        trained, train_message = (
            train_model()
        )

    if not trained:

        st.error(
            f"Student data saved, but model training failed: "
            f"{train_message}"
        )

        return

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    st.success(
        f"🎉 {name} registered successfully!"
    )

    st.success(
        f"✅ {train_message}"
    )

    st.info(
        f"Roll No: {rollno} | Branch: {branch}"
    )

    st.success(
        "The new student can now be recognized "
        "by the attendance system."
    )

    # Clear captured image
    st.session_state[
        "registration_photo"
    ] = None

    # Clear recognizer cache
    load_recognizer.clear()


# ============================================================
# ABOUT
# ============================================================

def about_page():

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                ℹ️ About
            </div>
            <div class="hero-subtitle">
                Smart Attendance System
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Smart Attendance System

        This system uses:

        - **Python**
        - **OpenCV**
        - **LBPH Face Recognition**
        - **Haar Cascade**
        - **Streamlit**
        - **Pandas**
        - **Excel**

        ### Attendance Rule

        A student can mark attendance only once within a
        **1-hour interval**.

        ### Recognition

        The system detects the face, compares it with the
        trained LBPH model and displays the student's:

        **Name + Roll Number**

        Attendance time is recorded using **Indian Standard Time (IST)**.
        """
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:22px;
            font-weight:800;
            margin-bottom:4px;
        ">
            🎓 Smart Attendance
        </div>

        <div style="
            color:#94a3b8;
            font-size:13px;
            margin-bottom:20px;
        ">
            Face Recognition System
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_time = now_ist()

    st.caption(
        f"🕒 {current_time.strftime('%I:%M:%S %p')} IST"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Mark Attendance",
            "Attendance Records",
            "Students",
            "Register Student",
            "About",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.caption(
        "Smart Attendance System"
    )

    st.caption(
        "© 2026"
    )


# ============================================================
# MAIN ROUTING
# ============================================================

if page == "Dashboard":

    dashboard()

elif page == "Mark Attendance":

    mark_attendance_page()

elif page == "Attendance Records":

    attendance_records()

elif page == "Students":

    students_page()

elif page == "Register Student":

    register_student()

elif page == "About":

    about_page()
