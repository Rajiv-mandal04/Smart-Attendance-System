import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

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


# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STUDENT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "students.csv"
)

ATTENDANCE_PATH = os.path.join(
    BASE_DIR,
    "attendance",
    "attendance.csv"
)

TRAINER_PATH = os.path.join(
    BASE_DIR,
    "trainer",
    "trainer.yml"
)

CASCADE_PATH = os.path.join(
    BASE_DIR,
    "haarcascade",
    "haarcascade_frontalface_default.xml"
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset"
)

os.makedirs(
    os.path.dirname(STUDENT_PATH),
    exist_ok=True
)

os.makedirs(
    os.path.dirname(ATTENDANCE_PATH),
    exist_ok=True
)

os.makedirs(
    os.path.dirname(TRAINER_PATH),
    exist_ok=True
)

os.makedirs(
    DATASET_PATH,
    exist_ok=True
)


# Professional dark UI
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(41,121,255,0.12),
                transparent 30%
            ),
            radial-gradient(
                circle at 85% 80%,
                rgba(124,77,255,0.10),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #0b1020 0%,
                #151b31 55%,
                #0b1020 100%
            );

        color: #e8ecf3;
    }

    section[data-testid="stSidebar"] {
        background: #0e1428;
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }

    section[data-testid="stSidebar"] * {
        color: #e8ecf3 !important;
    }

    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    .main-title {
        font-size: 2.65rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(
            90deg,
            #4fc3f7,
            #7c4dff,
            #ec407a
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.15rem;
    }

    .sub-title {
        text-align: center;
        color: #8b93a7;
        font-size: 0.82rem;
        margin-bottom: 2rem;
        letter-spacing: 2px;
        font-weight: 500;
    }

    .brand-box {
        text-align: center;
        padding: 0.5rem 0 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.07);
        margin-bottom: 1rem;
    }

    .brand-title {
        font-size: 1.15rem;
        font-weight: 700;
        background: linear-gradient(
            90deg,
            #4fc3f7,
            #7c4dff
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .brand-sub {
        font-size: 0.67rem;
        color: #687188;
        letter-spacing: 1.5px;
        margin-top: 3px;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"] > label {
        background: transparent;
        border-radius: 12px;
        padding: 0.65rem 0.9rem;
        margin: 0.2rem 0;
        border: 1px solid transparent;
        transition: 0.25s;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"] > label:hover {
        background: rgba(79,195,247,0.08);
        border-color: rgba(79,195,247,0.25);
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"] > label > div:first-child {
        display: none;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"] > label p {
        font-size: 0.9rem;
        font-weight: 500;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"] > label[data-checked="true"] {
        background: linear-gradient(
            135deg,
            rgba(41,121,255,0.18),
            rgba(124,77,255,0.18)
        );
        border-color: rgba(79,195,247,0.35);
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }

    .side-stat {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 0.75rem 0.9rem;
        margin-bottom: 0.6rem;
        display: flex;
        justify-content: space-between;
    }

    .side-stat-label {
        font-size: 0.73rem;
        color: #8b93a7;
    }

    .side-stat-value {
        font-size: 0.95rem;
        font-weight: 700;
        color: #4fc3f7;
    }

    .clock-box {
        background: linear-gradient(
            135deg,
            rgba(41,121,255,0.13),
            rgba(124,77,255,0.13)
        );
        border: 1px solid rgba(79,195,247,0.22);
        border-radius: 12px;
        padding: 0.8rem;
        text-align: center;
        margin-top: 0.7rem;
    }

    .clock-time {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 1px;
    }

    .clock-date {
        font-size: 0.7rem;
        color: #8b93a7;
        margin-top: 2px;
    }

    .metric-card {
        background: linear-gradient(
            160deg,
            rgba(255,255,255,0.06),
            rgba(255,255,255,0.025)
        );
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.15rem;
        text-align: center;
        box-shadow: 0 7px 24px rgba(0,0,0,0.25);
    }

    .metric-label {
        font-size: 0.68rem;
        color: #8b93a7;
        text-transform: uppercase;
        letter-spacing: 1.3px;
        font-weight: 600;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(
            90deg,
            #4fc3f7,
            #7c4dff
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 3px;
    }

    .panel {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1.25rem;
        box-shadow: 0 8px 28px rgba(0,0,0,0.20);
    }

    .recognition-card {
        background: linear-gradient(
            135deg,
            rgba(16,185,129,0.12),
            rgba(79,195,247,0.06)
        );
        border: 1px solid rgba(52,211,153,0.3);
        border-radius: 16px;
        padding: 1.1rem 1.2rem;
        margin-top: 1rem;
    }

    .recognition-title {
        color: #6ee7b7 !important;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }

    .recognition-text {
        color: #f1f5f9 !important;
        font-size: 0.9rem;
        line-height: 1.8;
    }

    .recognition-text b {
        color: #ffffff !important;
    }

    .time-card {
        background: rgba(79,195,247,0.08);
        border: 1px solid rgba(79,195,247,0.2);
        border-radius: 11px;
        padding: 0.65rem 0.8rem;
        color: #8ddcff;
        font-size: 0.8rem;
        margin-top: 0.7rem;
    }

    .success-card {
        background: rgba(16,185,129,0.12);
        border: 1px solid rgba(52,211,153,0.3);
        color: #6ee7b7;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        margin-top: 0.7rem;
        font-weight: 600;
    }

    .warning-card {
        background: rgba(245,158,11,0.10);
        border: 1px solid rgba(245,158,11,0.28);
        color: #fbbf24;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        margin-top: 0.7rem;
        font-weight: 600;
    }

    .danger-card {
        background: rgba(239,68,68,0.10);
        border: 1px solid rgba(248,113,113,0.28);
        color: #fca5a5;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        margin-top: 0.7rem;
        font-weight: 600;
    }

    .info-badge {
        display: inline-block;
        background: rgba(79,195,247,0.09);
        color: #4fc3f7;
        border: 1px solid rgba(79,195,247,0.2);
        border-radius: 20px;
        padding: 0.3rem 0.75rem;
        font-size: 0.72rem;
    }

    .stButton > button {
        background: linear-gradient(
            135deg,
            #2979ff,
            #7c4dff
        );
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(41,121,255,0.25);
    }

    .stButton > button:hover {
        box-shadow: 0 8px 22px rgba(124,77,255,0.35);
        transform: translateY(-1px);
    }

    .stTextInput input,
    .stSelectbox div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.04) !important;
        color: white !important;
        border-color: rgba(255,255,255,0.12) !important;
    }

    .stDataFrame {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# Use Indian time instead of cloud server time
IST = ZoneInfo("Asia/Kolkata")


def now_ist():
    return datetime.now(IST).replace(tzinfo=None)


def current_date():
    return now_ist().strftime("%Y-%m-%d")


def current_time():
    return now_ist().strftime("%H:%M:%S")


def current_display_time():
    return now_ist().strftime(
        "%I:%M:%S %p"
    )


def current_display_date():
    return now_ist().strftime(
        "%A, %d %B %Y"
    )


# Load student data
def load_students():

    if not os.path.exists(
        STUDENT_PATH
    ):
        return pd.DataFrame(
            columns=[
                "rollno",
                "name",
                "branch"
            ]
        )

    try:

        df = pd.read_csv(
            STUDENT_PATH,
            sep=None,
            engine="python"
        )

    except Exception:

        try:

            df = pd.read_csv(
                STUDENT_PATH,
                sep="\t"
            )

        except Exception:

            return pd.DataFrame(
                columns=[
                    "rollno",
                    "name",
                    "branch"
                ]
            )

    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    rename_map = {}

    for col in df.columns:

        if col in [
            "roll",
            "roll_no",
            "roll no",
            "roll number",
            "id"
        ]:
            rename_map[col] = "rollno"

        elif col in [
            "student_name",
            "student name"
        ]:
            rename_map[col] = "name"

        elif col in [
            "department"
        ]:
            rename_map[col] = "branch"

    df = df.rename(
        columns=rename_map
    )

    for col in [
        "rollno",
        "name",
        "branch"
    ]:

        if col not in df.columns:
            df[col] = ""

    return df[
        [
            "rollno",
            "name",
            "branch"
        ]
    ]


# Load attendance data
def load_attendance():

    columns = [
        "Roll No",
        "Name",
        "Branch",
        "Date",
        "Time",
        "Timestamp",
        "Status"
    ]

    if not os.path.exists(
        ATTENDANCE_PATH
    ):
        return pd.DataFrame(
            columns=columns
        )

    try:

        df = pd.read_csv(
            ATTENDANCE_PATH
        )

        if df.empty:
            return pd.DataFrame(
                columns=columns
            )

        df.columns = [
            str(col).strip()
            for col in df.columns
        ]

        rename_map = {}

        for col in df.columns:

            clean = (
                str(col)
                .strip()
                .lower()
            )

            if clean in [
                "rollno",
                "roll_no",
                "roll no",
                "roll number"
            ]:
                rename_map[col] = "Roll No"

            elif clean in [
                "name",
                "student_name",
                "student name"
            ]:
                rename_map[col] = "Name"

            elif clean in [
                "branch",
                "department"
            ]:
                rename_map[col] = "Branch"

            elif clean == "date":
                rename_map[col] = "Date"

            elif clean == "time":
                rename_map[col] = "Time"

            elif clean == "timestamp":
                rename_map[col] = "Timestamp"

            elif clean == "status":
                rename_map[col] = "Status"

        df = df.rename(
            columns=rename_map
        )

        for col in columns:

            if col not in df.columns:
                df[col] = ""

        # Create timestamp for old records
        if "Timestamp" not in df.columns:
            df["Timestamp"] = ""

        for index in df.index:

            if not str(
                df.at[index, "Timestamp"]
            ).strip():

                date_value = str(
                    df.at[index, "Date"]
                )

                time_value = str(
                    df.at[index, "Time"]
                )

                df.at[index, "Timestamp"] = (
                    f"{date_value} {time_value}"
                )

        return df[columns]

    except Exception:

        return pd.DataFrame(
            columns=columns
        )


def save_attendance_file(df):
    df.to_csv(
        ATTENDANCE_PATH,
        index=False
    )


# Load face detector
@st.cache_resource
def get_face_cascade():

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


# Load LBPH model
@st.cache_resource
def get_recognizer():

    if not os.path.exists(
        TRAINER_PATH
    ):
        return None

    try:

        model = (
            cv2.face
            .LBPHFaceRecognizer_create()
        )

        model.read(
            TRAINER_PATH
        )

        return model

    except Exception:

        return None


face_cascade = get_face_cascade()
recognizer = get_recognizer()


# Detect face
def detect_faces(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2GRAY
    )

    gray = cv2.equalizeHist(
        gray
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(40, 40)
    )

    if len(faces) == 0:

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30)
        )

    return gray, faces


# Find student from predicted label
def find_student(label):

    df = load_students()

    if df.empty:
        return None

    result = df[
        df["rollno"]
        .astype(str)
        .str.strip()
        ==
        str(label).strip()
    ]

    if result.empty:
        return None

    return result.iloc[0]


# Get last attendance time
def get_last_attendance(roll_no):

    df = load_attendance()

    if df.empty:
        return None

    records = df[
        df["Roll No"]
        .astype(str)
        .str.strip()
        ==
        str(roll_no).strip()
    ].copy()

    if records.empty:
        return None

    timestamps = pd.to_datetime(
        records["Timestamp"],
        errors="coerce"
    ).dropna()

    if timestamps.empty:
        return None

    last_timestamp = timestamps.max()

    return last_timestamp.to_pydatetime()


# Mark attendance
def mark_attendance(
    roll_no,
    name,
    branch
):

    df = load_attendance()

    now = now_ist()

    last_time = get_last_attendance(
        roll_no
    )

    if last_time is not None:

        difference = (
            now - last_time
        )

        if difference < timedelta(
            hours=1
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

            return {
                "marked": False,
                "status": "Re-Verified",
                "date": now.strftime(
                    "%Y-%m-%d"
                ),
                "time": now.strftime(
                    "%H:%M:%S"
                ),
                "last_time": last_time,
                "remaining": (
                    f"{minutes}m {seconds}s"
                )
            }

    new_row = {
        "Roll No": str(roll_no),
        "Name": str(name),
        "Branch": str(branch),
        "Date": now.strftime(
            "%Y-%m-%d"
        ),
        "Time": now.strftime(
            "%H:%M:%S"
        ),
        "Timestamp": now.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "Status": "Present"
    }

    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [new_row]
            )
        ],
        ignore_index=True
    )

    save_attendance_file(
        df
    )

    return {
        "marked": True,
        "status": "Present",
        "date": new_row["Date"],
        "time": new_row["Time"],
        "timestamp": new_row["Timestamp"],
        "last_time": None,
        "remaining": ""
    }


# Draw scanning box
def draw_recognition_box(
    frame,
    x,
    y,
    w,
    h,
    name,
    roll_no,
    confidence
):

    output = frame.copy()

    green = (
        40,
        220,
        120
    )

    white = (
        255,
        255,
        255
    )

    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        green,
        3
    )

    label = (
        f"{name}  |  Roll: {roll_no}"
    )

    confidence_label = (
        f"Confidence: {confidence:.2f}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    label_size = cv2.getTextSize(
        label,
        font,
        0.55,
        2
    )[0]

    label_width = (
        label_size[0] + 20
    )

    label_height = 35

    label_y = max(
        y - label_height,
        0
    )

    cv2.rectangle(
        output,
        (
            x,
            label_y
        ),
        (
            x + label_width,
            y
        ),
        green,
        -1
    )

    cv2.putText(
        output,
        label,
        (
            x + 8,
            y - 10
        ),
        font,
        0.55,
        white,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        confidence_label,
        (
            x,
            min(
                y + h + 25,
                output.shape[0] - 5
            )
        ),
        font,
        0.55,
        green,
        2,
        cv2.LINE_AA
    )

    return output


# Train LBPH model
def train_model():

    images = []
    labels = []

    if not os.path.exists(
        DATASET_PATH
    ):
        return False, "Dataset folder not found."

    for roll_folder in os.listdir(
        DATASET_PATH
    ):

        folder = os.path.join(
            DATASET_PATH,
            roll_folder
        )

        if not os.path.isdir(
            folder
        ):
            continue

        try:

            label = int(
                roll_folder
            )

        except ValueError:

            continue

        for filename in os.listdir(
            folder
        ):

            path = os.path.join(
                folder,
                filename
            )

            image = cv2.imread(
                path
            )

            if image is None:
                continue

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30)
            )

            if len(faces) == 0:

                images.append(
                    gray
                )

                labels.append(
                    label
                )

            else:

                for x, y, w, h in faces:

                    face = gray[
                        y:y + h,
                        x:x + w
                    ]

                    images.append(
                        face
                    )

                    labels.append(
                        label
                    )

    if not images:

        return False, (
            "No face images available for training."
        )

    try:

        model = (
            cv2.face
            .LBPHFaceRecognizer_create()
        )

        model.train(
            images,
            np.array(labels)
        )

        model.write(
            TRAINER_PATH
        )

        get_recognizer.clear()

        return True, (
            "Face model trained successfully."
        )

    except Exception as e:

        return False, str(e)


# Register student
def register_student(
    roll_no,
    name,
    branch,
    image_file
):

    roll_no = str(
        roll_no
    ).strip()

    name = str(
        name
    ).strip()

    branch = str(
        branch
    ).strip()

    if not roll_no:
        return False, "Roll number is required."

    if not name:
        return False, "Student name is required."

    if not branch:
        return False, "Branch is required."

    if image_file is None:
        return False, "Please capture a face first."

    students_df = load_students()

    existing = students_df[
        students_df["rollno"]
        .astype(str)
        .str.strip()
        ==
        roll_no
    ]

    if not existing.empty:

        return False, (
            "This roll number is already registered."
        )

    try:

        image = Image.open(
            image_file
        )

        frame = np.array(
            image
        )

        gray, faces = detect_faces(
            frame
        )

        if len(faces) == 0:

            return False, (
                "No face detected. Please look directly at the camera."
            )

        if len(faces) > 1:

            return False, (
                "Multiple faces detected. Please keep only one person."
            )

        x, y, w, h = faces[0]

        face = gray[
            y:y + h,
            x:x + w
        ]

        student_folder = os.path.join(
            DATASET_PATH,
            roll_no
        )

        os.makedirs(
            student_folder,
            exist_ok=True
        )

        files = os.listdir(
            student_folder
        )

        image_number = (
            len(files) + 1
        )

        image_path = os.path.join(
            student_folder,
            f"User.{roll_no}.{image_number}.jpg"
        )

        cv2.imwrite(
            image_path,
            face
        )

        new_student = pd.DataFrame(
            [
                {
                    "rollno": roll_no,
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

        students_df.to_csv(
            STUDENT_PATH,
            sep="\t",
            index=False
        )

        success, message = train_model()

        if not success:

            return False, message

        return True, (
            "Student registered and face model updated."
        )

    except Exception as e:

        return False, str(e)


# Sidebar
with st.sidebar:

    st.markdown(
        """
        <div class="brand-box">
            <div class="brand-title">
                🎓 Smart Attendance
            </div>
            <div class="brand-sub">
                AI · REAL-TIME · ATTENDANCE
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    menu = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📸 Mark Attendance",
            "👥 Students",
            "📊 Attendance Records",
            "ℹ️ About"
        ],
        label_visibility="collapsed"
    )

    students_sidebar = load_students()
    attendance_sidebar = load_attendance()

    st.markdown(
        f"""
        <div class="side-stat">
            <span class="side-stat-label">
                Registered Students
            </span>
            <span class="side-stat-value">
                {len(students_sidebar)}
            </span>
        </div>

        <div class="side-stat">
            <span class="side-stat-label">
                Attendance Records
            </span>
            <span class="side-stat-value">
                {len(attendance_sidebar)}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="clock-box">
            <div class="clock-time">
                {current_display_time()}
            </div>
            <div class="clock-date">
                {current_display_date()}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            color:#737d94;
            font-size:0.7rem;
            line-height:1.5;
            margin-top:0.8rem;
        ">
            ⏱️ Same student can receive a new
            <b>Present</b> entry only after 1 hour.
        </div>
        """,
        unsafe_allow_html=True
    )


# Page header
st.markdown(
    '<div class="main-title">Smart Attendance System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">AI POWERED · FACE RECOGNITION · REAL TIME</div>',
    unsafe_allow_html=True
)


# Dashboard
if menu == "🏠 Dashboard":

    students_df = load_students()
    attendance_df = load_attendance()

    today = current_date()

    if attendance_df.empty:

        today_df = pd.DataFrame()

    else:

        today_df = attendance_df[
            attendance_df["Date"]
            .astype(str)
            ==
            today
        ]

    total_students = len(
        students_df
    )

    present_today = (
        today_df["Roll No"]
        .astype(str)
        .nunique()
        if not today_df.empty
        else 0
    )

    total_records = len(
        attendance_df
    )

    attendance_rate = (
        round(
            (
                present_today
                /
                total_students
            )
            * 100,
            1
        )
        if total_students > 0
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Registered Students
                </div>
                <div class="metric-value">
                    {total_students}
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
                    Present Today
                </div>
                <div class="metric-value">
                    {present_today}
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
                    Total Records
                </div>
                <div class="metric-value">
                    {total_records}
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
                    Attendance Rate
                </div>
                <div class="metric-value">
                    {attendance_rate}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        "### 📅 Today's Attendance"
    )

    if today_df.empty:

        st.info(
            "No attendance has been marked today."
        )

    else:

        st.dataframe(
            today_df.sort_values(
                "Timestamp",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )


# Mark attendance
elif menu == "📸 Mark Attendance":

    st.markdown(
        "### 📸 Mark Attendance"
    )

    st.markdown(
        """
        <span class="info-badge">
        🔐 LBPH Face Recognition
        </span>
        &nbsp;
        <span class="info-badge">
        ⏱️ 1 Hour Protection
        </span>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    if face_cascade is None:

        st.error(
            "Haar Cascade file not found."
        )

    elif recognizer is None:

        st.error(
            "trainer.yml could not be loaded."
        )

    else:

        left, right = st.columns(
            [1.4, 0.6],
            gap="large"
        )

        with left:

            st.markdown(
                '<div class="panel">',
                unsafe_allow_html=True
            )

            st.markdown(
                "#### 📷 Face Scanner"
            )

            st.caption(
                "Keep one face in front of the camera and capture a clear image."
            )

            camera_image = st.camera_input(
                "Camera",
                key="attendance_camera"
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

            if camera_image is not None:

                image = Image.open(
                    camera_image
                )

                frame = np.array(
                    image
                )

                scan_now = now_ist()

                gray, faces = detect_faces(
                    frame
                )

                if len(faces) == 0:

                    st.markdown(
                        """
                        <div class="warning-card">
                        ⚠️ Face not detected.
                        Please look directly at the camera.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.image(
                        frame,
                        use_container_width=True
                    )

                elif len(faces) > 1:

                    st.markdown(
                        """
                        <div class="warning-card">
                        ⚠️ Multiple faces detected.
                        Please keep only one person in front of the camera.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.image(
                        frame,
                        use_container_width=True
                    )

                else:

                    x, y, w, h = faces[0]

                    face = gray[
                        y:y + h,
                        x:x + w
                    ]

                    try:

                        label, confidence = (
                            recognizer.predict(
                                face
                            )
                        )

                        student = find_student(
                            label
                        )

                        if (
                            student is None
                            or confidence >= 85
                        ):

                            st.markdown(
                                """
                                <div class="danger-card">
                                ❌ Face not registered.
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            st.image(
                                frame,
                                caption="Unknown person",
                                use_container_width=True
                            )

                            st.info(
                                "Go to the Students section and register this student first."
                            )

                        else:

                            roll_no = student[
                                "rollno"
                            ]

                            name = student[
                                "name"
                            ]

                            branch = student[
                                "branch"
                            ]

                            result_frame = draw_recognition_box(
                                frame,
                                x,
                                y,
                                w,
                                h,
                                name,
                                roll_no,
                                confidence
                            )

                            st.image(
                                result_frame,
                                caption="Face recognized",
                                use_container_width=True
                            )

                            st.markdown(
                                f"""
                                <div class="recognition-card">

                                    <div class="recognition-title">
                                        ✓ Student Recognized
                                    </div>

                                    <div class="recognition-text">
                                        <b>Name:</b> {name}<br>
                                        <b>Roll No:</b> {roll_no}<br>
                                        <b>Branch:</b> {branch}<br>
                                        <b>Confidence:</b> {confidence:.2f}
                                    </div>

                                    <div class="time-card">
                                        🕒 Current Scan Time:
                                        {scan_now.strftime("%d %b %Y, %I:%M:%S %p")}
                                        <br>
                                        🇮🇳 India Standard Time
                                    </div>

                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            result = mark_attendance(
                                roll_no,
                                name,
                                branch
                            )

                            if result["marked"]:

                                st.markdown(
                                    f"""
                                    <div class="success-card">
                                        ✅ Attendance Marked Successfully
                                        <br>
                                        <span style="
                                            font-size:0.78rem;
                                            font-weight:400;
                                        ">
                                        Present · {result["date"]}
                                        · {result["time"]}
                                        </span>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                            else:

                                st.markdown(
                                    f"""
                                    <div class="warning-card">
                                        🔄 Re-Verified
                                        <br>
                                        <span style="
                                            font-size:0.78rem;
                                            font-weight:400;
                                        ">
                                        Current scan:
                                        {result["date"]}
                                        {result["time"]}
                                        <br>
                                        Last attendance:
                                        {result["last_time"].strftime("%d %b %Y, %I:%M:%S")}
                                        <br>
                                        New Present entry available after:
                                        {result["remaining"]}
                                        </span>
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
                """
                <div class="panel">

                <h3>How it works</h3>

                <p style="color:#a5aec0;font-size:0.82rem;">
                <b style="color:#4fc3f7;">01</b>
                Face Detection
                </p>

                <p style="color:#a5aec0;font-size:0.82rem;">
                <b style="color:#4fc3f7;">02</b>
                LBPH Recognition
                </p>

                <p style="color:#a5aec0;font-size:0.82rem;">
                <b style="color:#4fc3f7;">03</b>
                Student Verification
                </p>

                <p style="color:#a5aec0;font-size:0.82rem;">
                <b style="color:#4fc3f7;">04</b>
                1 Hour Attendance Check
                </p>

                <p style="color:#a5aec0;font-size:0.82rem;">
                <b style="color:#4fc3f7;">05</b>
                Current IST Timestamp
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            st.markdown(
                f"""
                <div class="clock-box">
                    <div class="clock-time">
                        {current_display_time()}
                    </div>
                    <div class="clock-date">
                        {current_display_date()}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# Students
elif menu == "👥 Students":

    st.markdown(
        "### 👥 Students"
    )

    students_df = load_students()

    c1, c2 = st.columns(
        [1, 1]
    )

    with c1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Registered Students
                </div>
                <div class="metric-value">
                    {len(students_df)}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                    Recognition Model
                </div>
                <div class="metric-value">
                    LBPH
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    if not students_df.empty:

        st.dataframe(
            students_df,
            use_container_width=True,
            hide_index=True
        )

    st.markdown(
        "---"
    )

    st.markdown(
        "### ➕ Register New Student"
    )

    st.caption(
        "Enter student details and capture one clear face image."
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        new_roll = st.text_input(
            "Roll No",
            placeholder="11232743",
            key="new_roll"
        )

    with col2:

        new_name = st.text_input(
            "Student Name",
            placeholder="Rajiv Kr. Mandal",
            key="new_name"
        )

    with col3:

        new_branch = st.text_input(
            "Branch",
            placeholder="B.Tech CSE",
            key="new_branch"
        )

    register_image = st.camera_input(
        "Capture Face",
        key="register_camera"
    )

    if register_image is not None:

        st.success(
            "Face image captured."
        )

    if st.button(
        "➕ Register Student",
        type="primary",
        use_container_width=True,
        key="register_student_btn"
    ):

        with st.spinner(
            "Registering student and training model..."
        ):

            success, message = register_student(
                new_roll,
                new_name,
                new_branch,
                register_image
            )

        if success:

            st.success(
                message
            )

            st.info(
                "Registration complete. Ab Mark Attendance section mein face scan karo."
            )

            st.rerun()

        else:

            st.error(
                message
            )


# Attendance records
elif menu == "📊 Attendance Records":

    st.markdown(
        "### 📊 Attendance Records"
    )

    df = load_attendance()

    if df.empty:

        st.info(
            "No attendance records available."
        )

    else:

        total = len(df)

        present = len(
            df[
                df["Status"]
                .astype(str)
                .str.lower()
                ==
                "present"
            ]
        )

        unique_students = (
            df["Roll No"]
            .astype(str)
            .nunique()
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        Total Records
                    </div>
                    <div class="metric-value">
                        {total}
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
                        Present Entries
                    </div>
                    <div class="metric-value">
                        {present}
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
                        Unique Students
                    </div>
                    <div class="metric-value">
                        {unique_students}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        col1, col2 = st.columns(2)

        with col1:

            dates = sorted(
                df["Date"]
                .astype(str)
                .unique()
                .tolist(),
                reverse=True
            )

            date_filter = st.selectbox(
                "Filter Date",
                ["All"] + dates
            )

        with col2:

            status_filter = st.selectbox(
                "Filter Status",
                [
                    "All",
                    "Present",
                    "Re-Verified"
                ]
            )

        filtered = df.copy()

        if date_filter != "All":

            filtered = filtered[
                filtered["Date"]
                .astype(str)
                ==
                date_filter
            ]

        if status_filter != "All":

            filtered = filtered[
                filtered["Status"]
                .astype(str)
                .str.lower()
                ==
                status_filter.lower()
            ]

        filtered = filtered.sort_values(
            "Timestamp",
            ascending=False
        )

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True
        )

        csv_data = filtered.to_csv(
            index=False
        ).encode(
            "utf-8"
        )

        st.download_button(
            "⬇️ Download Attendance CSV",
            data=csv_data,
            file_name=(
                f"attendance_{current_date()}.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )


# About
elif menu == "ℹ️ About":

    st.markdown(
        "### ℹ️ About"
    )

    st.markdown(
        """
        <div class="panel">

        <h3>Smart Attendance System</h3>

        <p style="color:#a5aec0;">
        A face-recognition based attendance management
        system built using Python, OpenCV, LBPH and Streamlit.
        </p>

        <hr>

        <p style="color:#a5aec0;">
        <b style="color:#4fc3f7;">Face Detection</b><br>
        Haar Cascade
        </p>

        <p style="color:#a5aec0;">
        <b style="color:#4fc3f7;">Face Recognition</b><br>
        LBPH Face Recognizer
        </p>

        <p style="color:#a5aec0;">
        <b style="color:#4fc3f7;">Attendance Storage</b><br>
        CSV
        </p>

        <p style="color:#a5aec0;">
        <b style="color:#4fc3f7;">Duplicate Protection</b><br>
        1 Hour Rule
        </p>

        <p style="color:#a5aec0;">
        <b style="color:#4fc3f7;">Timezone</b><br>
        India Standard Time
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# Footer
st.markdown(
    """
    <div style="
        text-align:center;
        color:#626d83;
        font-size:0.72rem;
        padding:2rem 0 0.5rem;
    ">
        🎓 Smart Attendance System · OpenCV · LBPH · Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
