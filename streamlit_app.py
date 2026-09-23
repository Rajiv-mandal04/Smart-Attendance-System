import os
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
    "attendance.xlsx"
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


# Professional UI styling
st.markdown(
    """
    <style>

    .stApp {
        background: #f5f7fb;
    }

    [data-testid="stSidebar"] {
        background: #111827;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    .main-title {
        font-size: 36px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 2px;
    }

    .main-subtitle {
        color: #6b7280;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .top-card {
        background: white;
        padding: 18px 22px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.04);
    }

    .recognition-card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #dbe3ef;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.07);
        margin-top: 18px;
    }

    .recognition-title {
        color: #111827 !important;
        font-size: 23px;
        font-weight: 750;
        margin-bottom: 15px;
    }

    .student-info {
        color: #111827 !important;
        font-size: 16px;
        line-height: 1.9;
    }

    .student-info b {
        color: #374151 !important;
    }

    .time-info {
        color: #4b5563 !important;
        font-size: 14px;
        margin-top: 10px;
    }

    .success-card {
        background: #ecfdf5;
        border: 1px solid #86efac;
        color: #166534;
        padding: 17px 20px;
        border-radius: 13px;
        margin-top: 15px;
        font-weight: 600;
    }

    .warning-card {
        background: #fffbeb;
        border: 1px solid #fcd34d;
        color: #92400e;
        padding: 17px 20px;
        border-radius: 13px;
        margin-top: 15px;
        font-weight: 600;
    }

    .danger-card {
        background: #fef2f2;
        border: 1px solid #fca5a5;
        color: #991b1b;
        padding: 17px 20px;
        border-radius: 13px;
        margin-top: 15px;
        font-weight: 600;
    }

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.04);
    }

    .metric-title {
        color: #6b7280;
        font-size: 13px;
        font-weight: 600;
    }

    .metric-value {
        color: #111827;
        font-size: 28px;
        font-weight: 800;
        margin-top: 4px;
    }

    .scan-time {
        background: #eef2ff;
        color: #3730a3;
        border: 1px solid #c7d2fe;
        border-radius: 10px;
        padding: 10px 14px;
        font-size: 14px;
        font-weight: 600;
        margin-top: 12px;
    }

    .footer {
        text-align: center;
        color: #9ca3af;
        padding: 30px 0 15px;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# Load students
def load_students():

    if not os.path.exists(STUDENT_PATH):
        return pd.DataFrame(
            columns=[
                "rollno",
                "name",
                "branch"
            ]
        )

    try:
        students = pd.read_csv(
            STUDENT_PATH,
            sep=None,
            engine="python"
        )

    except Exception:

        try:
            students = pd.read_csv(
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

    students.columns = [
        str(col).strip().lower()
        for col in students.columns
    ]

    rename_map = {}

    for col in students.columns:

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

    students = students.rename(
        columns=rename_map
    )

    for column in [
        "rollno",
        "name",
        "branch"
    ]:

        if column not in students.columns:
            students[column] = ""

    return students[
        [
            "rollno",
            "name",
            "branch"
        ]
    ]


students = load_students()


# Load attendance
def load_attendance():

    columns = [
        "Roll No",
        "Name",
        "Branch",
        "Date",
        "Time",
        "Status"
    ]

    if not os.path.exists(
        ATTENDANCE_PATH
    ):
        return pd.DataFrame(
            columns=columns
        )

    try:

        df = pd.read_excel(
            ATTENDANCE_PATH
        )

        if df.empty:
            return pd.DataFrame(
                columns=columns
            )

        rename_map = {}

        for col in df.columns:

            clean_col = (
                str(col)
                .strip()
                .lower()
            )

            if clean_col in [
                "rollno",
                "roll_no",
                "roll no",
                "roll number",
                "id"
            ]:
                rename_map[col] = "Roll No"

            elif clean_col in [
                "name",
                "student_name",
                "student name"
            ]:
                rename_map[col] = "Name"

            elif clean_col in [
                "branch",
                "department"
            ]:
                rename_map[col] = "Branch"

            elif clean_col == "date":
                rename_map[col] = "Date"

            elif clean_col == "time":
                rename_map[col] = "Time"

            elif clean_col == "status":
                rename_map[col] = "Status"

        df = df.rename(
            columns=rename_map
        )

        for column in columns:

            if column not in df.columns:
                df[column] = ""

        return df[columns]

    except Exception:

        return pd.DataFrame(
            columns=columns
        )


attendance_df = load_attendance()


# Load face detector
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


face_cascade = load_face_cascade()


# Load face recognition model
@st.cache_resource
def load_recognizer():

    if not os.path.exists(
        TRAINER_PATH
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


recognizer = load_recognizer()


# Detect faces
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


# Get student
def get_student(roll_no):

    if students.empty:
        return None

    result = students[
        students["rollno"]
        .astype(str)
        .str.strip()
        ==
        str(roll_no).strip()
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

    records["datetime"] = pd.to_datetime(
        records["Date"].astype(str)
        + " "
        + records["Time"].astype(str),
        errors="coerce"
    )

    records = records.dropna(
        subset=["datetime"]
    )

    if records.empty:
        return None

    return records[
        "datetime"
    ].max()


# Save attendance
def save_attendance(
    roll_no,
    name,
    branch
):

    global attendance_df

    now = datetime.now()

    today = now.strftime(
        "%Y-%m-%d"
    )

    current_time = now.strftime(
        "%H:%M:%S"
    )

    attendance_df = load_attendance()

    last_attendance = (
        get_last_attendance(
            roll_no
        )
    )

    if last_attendance is not None:

        time_difference = (
            now - last_attendance
        )

        if time_difference < timedelta(
            hours=1
        ):

            minutes_left = max(
                0,
                int(
                    60
                    - time_difference.total_seconds()
                    / 60
                )
            )

            return {
                "status": "Re-Verified",
                "date": today,
                "time": current_time,
                "last_time": last_attendance,
                "minutes_left": minutes_left
            }

    new_record = pd.DataFrame(
        [
            {
                "Roll No": roll_no,
                "Name": name,
                "Branch": branch,
                "Date": today,
                "Time": current_time,
                "Status": "Present"
            }
        ]
    )

    attendance_df = pd.concat(
        [
            attendance_df,
            new_record
        ],
        ignore_index=True
    )

    attendance_df.to_excel(
        ATTENDANCE_PATH,
        index=False
    )

    return {
        "status": "Present",
        "date": today,
        "time": current_time,
        "last_time": None,
        "minutes_left": 0
    }


# Draw face box and student details
def draw_face_box(
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

    color = (
        34,
        197,
        94
    )

    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        color,
        3
    )

    label = (
        f"{name} | Roll: {roll_no}"
    )

    confidence_text = (
        f"Confidence: {confidence:.2f}"
    )

    text_y = max(
        y - 15,
        30
    )

    cv2.rectangle(
        output,
        (x, text_y - 35),
        (
            x + max(
                300,
                len(label) * 10
            ),
            text_y
        ),
        color,
        -1
    )

    cv2.putText(
        output,
        label,
        (x + 8, text_y - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        confidence_text,
        (x, y + h + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA
    )

    return output


# Train model
def train_model():

    image_data = []
    labels = []

    if not os.path.exists(
        DATASET_PATH
    ):
        return False, "Dataset folder not found."

    for roll_folder in os.listdir(
        DATASET_PATH
    ):

        folder_path = os.path.join(
            DATASET_PATH,
            roll_folder
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

        for image_name in os.listdir(
            folder_path
        ):

            image_path = os.path.join(
                folder_path,
                image_name
            )

            image = cv2.imread(
                image_path
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

                image_data.append(
                    gray
                )

                labels.append(
                    label
                )

                continue

            for x, y, w, h in faces:

                face = gray[
                    y:y + h,
                    x:x + w
                ]

                image_data.append(
                    face
                )

                labels.append(
                    label
                )

    if not image_data:
        return False, "No face images found."

    try:

        model = (
            cv2.face
            .LBPHFaceRecognizer_create()
        )

        model.train(
            image_data,
            np.array(labels)
        )

        model.write(
            TRAINER_PATH
        )

        load_recognizer.clear()

        return True, "Face model trained successfully."

    except Exception as e:

        return False, str(e)


# Register student
def register_student(
    roll_no,
    name,
    branch,
    image_file
):

    global students
    global recognizer

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
        return False, "Please enter roll number."

    if not name:
        return False, "Please enter student name."

    if not branch:
        return False, "Please enter branch."

    if image_file is None:
        return False, "Please capture the student's face."

    existing = students[
        students["rollno"]
        .astype(str)
        .str.strip()
        ==
        roll_no
    ]

    if not existing.empty:
        return False, "This roll number is already registered."

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
            return False, "No face detected. Please look directly at the camera."

        if len(faces) > 1:
            return False, "Multiple faces detected. Please keep only one person in front of the camera."

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

        existing_images = os.listdir(
            student_folder
        )

        image_number = (
            len(existing_images)
            + 1
        )

        image_path = os.path.join(
            student_folder,
            f"User.{roll_no}.{image_number}.jpg"
        )

        cv2.imwrite(
            image_path,
            face
        )

    except Exception as e:

        return False, (
            f"Could not save face image: {e}"
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

    students = pd.concat(
        [
            students,
            new_student
        ],
        ignore_index=True
    )

    students.to_csv(
        STUDENT_PATH,
        sep="\t",
        index=False
    )

    success, message = train_model()

    if not success:
        return False, message

    recognizer = load_recognizer()

    return True, (
        "Student registered successfully."
    )


# Header
st.markdown(
    '<div class="main-title">🎓 Smart Attendance System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">Fast and secure face recognition based attendance management</div>',
    unsafe_allow_html=True
)


# Sidebar
with st.sidebar:

    st.markdown(
        "## 🎓 Smart Attendance"
    )

    st.markdown(
        "### System Overview"
    )

    st.metric(
        "Registered Students",
        len(students)
    )

    st.metric(
        "Attendance Records",
        len(attendance_df)
    )

    st.divider()

    current_sidebar_time = datetime.now().strftime(
        "%d %b %Y • %I:%M:%S %p"
    )

    st.caption(
        "Current Time"
    )

    st.write(
        current_sidebar_time
    )

    st.divider()

    st.info(
        "A student can receive a new Present entry only after the previous attendance is older than 1 hour."
    )


# Tabs
tab1, tab2, tab3 = st.tabs(
    [
        "📷  Scan Attendance",
        "➕  Register Student",
        "📋  Attendance Records"
    ]
)


# Scan attendance
with tab1:

    left, right = st.columns(
        [1.35, 0.65],
        gap="large"
    )

    with left:

        st.subheader(
            "Face Recognition"
        )

        st.caption(
            "Look directly at the camera and capture your face."
        )

        camera_image = st.camera_input(
            "Camera",
            key="attendance_camera"
        )

        if camera_image is not None:

            image = Image.open(
                camera_image
            )

            frame = np.array(
                image
            )

            scan_time = datetime.now()

            st.markdown(
                f"""
                <div class="scan-time">
                🕒 Scan Time:
                {scan_time.strftime("%d %b %Y, %I:%M:%S %p")}
                </div>
                """,
                unsafe_allow_html=True
            )

            if face_cascade is None:

                st.error(
                    "Face detection model could not be loaded."
                )

            elif recognizer is None:

                st.error(
                    "Face recognition model is not available."
                )

            else:

                gray, faces = detect_faces(
                    frame
                )

                if len(faces) == 0:

                    st.markdown(
                        """
                        <div class="warning-card">
                        ⚠️ No face detected. Please look directly at the camera.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.image(
                        frame,
                        use_container_width=True
                    )

                elif len(faces) > 1:

                    st.warning(
                        "Multiple faces detected. Please keep only one person in front of the camera."
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

                        label, confidence = recognizer.predict(
                            face
                        )

                        student = get_student(
                            label
                        )

                        if (
                            student is None
                            or confidence >= 85
                        ):

                            st.markdown(
                                """
                                <div class="danger-card">
                                ❌ Face not registered. Please register the student first.
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            st.image(
                                frame,
                                use_container_width=True
                            )

                            st.info(
                                "Open Register Student and add this student's face."
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

                            annotated_frame = draw_face_box(
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
                                annotated_frame,
                                caption="Student detected",
                                use_container_width=True
                            )

                            result = save_attendance(
                                roll_no,
                                name,
                                branch
                            )

                            st.markdown(
                                f"""
                                <div class="recognition-card">

                                    <div class="recognition-title">
                                    👤 Student Recognized
                                    </div>

                                    <div class="student-info">
                                    <b>Name:</b> {name}<br>
                                    <b>Roll No:</b> {roll_no}<br>
                                    <b>Branch:</b> {branch}<br>
                                    <b>Confidence:</b> {confidence:.2f}
                                    </div>

                                    <div class="time-info">
                                    🕒 Current Scan:
                                    {result["date"]} •
                                    {result["time"]}
                                    </div>

                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            if result["status"] == "Present":

                                st.markdown(
                                    f"""
                                    <div class="success-card">
                                    ✅ Attendance Marked Successfully
                                    <br>
                                    <small>
                                    Present recorded at
                                    {result["time"]}
                                    </small>
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
                                    <small>
                                    Current scan:
                                    {result["time"]}
                                    <br>
                                    Last attendance:
                                    {result["last_time"].strftime("%d %b %Y, %I:%M:%S %p")}
                                    <br>
                                    New attendance can be marked after approximately
                                    {result["minutes_left"]} minute(s).
                                    </small>
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
            <div class="top-card">

            <h3 style="color:#111827;">
            How it works
            </h3>

            <p style="color:#4b5563;">
            1. Capture your face
            </p>

            <p style="color:#4b5563;">
            2. System detects the face
            </p>

            <p style="color:#4b5563;">
            3. LBPH model identifies the student
            </p>

            <p style="color:#4b5563;">
            4. Attendance is checked against the 1-hour rule
            </p>

            <p style="color:#4b5563;">
            5. Present or Re-Verified status is shown
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


# Register student
with tab2:

    st.subheader(
        "Register New Student"
    )

    st.caption(
        "Add student details and capture one clear face image."
    )

    col1, col2, col3 = st.columns(
        3,
        gap="medium"
    )

    with col1:

        roll_no = st.text_input(
            "Roll No",
            placeholder="Example: 11232743"
        )

    with col2:

        name = st.text_input(
            "Student Name",
            placeholder="Example: Rajiv Kr. Mandal"
        )

    with col3:

        branch = st.text_input(
            "Branch",
            placeholder="Example: B.Tech CSE"
        )

    st.write("")

    register_camera = st.camera_input(
        "Capture Student Face",
        key="register_camera"
    )

    if register_camera is not None:

        st.success(
            "Face captured successfully."
        )

    if st.button(
        "Register Student",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Registering student and training face model..."
        ):

            success, message = register_student(
                roll_no,
                name,
                branch,
                register_camera
            )

        if success:

            st.success(
                message
            )

            st.info(
                "Registration complete. Go to Scan Attendance and scan the student."
            )

            st.rerun()

        else:

            st.error(
                message
            )


# Attendance records
with tab3:

    st.subheader(
        "Attendance Records"
    )

    attendance_df = load_attendance()

    if attendance_df.empty:

        st.info(
            "No attendance records available yet."
        )

    else:

        total_records = len(
            attendance_df
        )

        unique_students = (
            attendance_df["Roll No"]
            .astype(str)
            .replace("", np.nan)
            .dropna()
            .nunique()
        )

        today = datetime.now().strftime(
            "%Y-%m-%d"
        )

        today_count = len(
            attendance_df[
                attendance_df["Date"]
                .astype(str)
                .str.contains(
                    today,
                    na=False
                )
            ]
        )

        col1, col2, col3 = st.columns(
            3,
            gap="medium"
        )

        with col1:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">
                    TOTAL RECORDS
                    </div>
                    <div class="metric-value">
                    {total_records}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">
                    UNIQUE STUDENTS
                    </div>
                    <div class="metric-value">
                    {unique_students}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">
                    TODAY'S ATTENDANCE
                    </div>
                    <div class="metric-value">
                    {today_count}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        display_df = attendance_df.copy()

        display_df = display_df.rename(
            columns={
                "Roll No": "Roll No",
                "Name": "Student",
                "Branch": "Branch",
                "Date": "Date",
                "Time": "Time",
                "Status": "Status"
            }
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Status": st.column_config.TextColumn(
                    "Status"
                ),
                "Date": st.column_config.TextColumn(
                    "Date"
                ),
                "Time": st.column_config.TextColumn(
                    "Time"
                )
            }
        )

        st.write("")

        if os.path.exists(
            ATTENDANCE_PATH
        ):

            with open(
                ATTENDANCE_PATH,
                "rb"
            ) as file:

                st.download_button(
                    "⬇️ Download Attendance Excel",
                    data=file,
                    file_name="attendance.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )


# Footer
st.markdown(
    """
    <div class="footer">
        Smart Attendance System · OpenCV · LBPH · Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
