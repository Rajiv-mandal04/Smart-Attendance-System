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
    layout="wide"
)


# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STUDENT_PATH = os.path.join(BASE_DIR, "data", "students.csv")
ATTENDANCE_PATH = os.path.join(BASE_DIR, "attendance", "attendance.xlsx")
TRAINER_PATH = os.path.join(BASE_DIR, "trainer", "trainer.yml")
CASCADE_PATH = os.path.join(
    BASE_DIR,
    "haarcascade",
    "haarcascade_frontalface_default.xml"
)
DATASET_PATH = os.path.join(BASE_DIR, "dataset")


os.makedirs(os.path.dirname(STUDENT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(ATTENDANCE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(TRAINER_PATH), exist_ok=True)
os.makedirs(DATASET_PATH, exist_ok=True)


# Custom styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }

    .title {
        font-size: 34px;
        font-weight: 700;
        color: #1f2937;
    }

    .subtitle {
        color: #6b7280;
        font-size: 16px;
        margin-bottom: 20px;
    }

    .student-card {
        background: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        margin-top: 15px;
    }

    .status-card {
        padding: 18px;
        border-radius: 12px;
        margin-top: 15px;
        background: #ffffff;
        border: 1px solid #e5e7eb;
    }

    .success-box {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
        padding: 15px;
        border-radius: 10px;
    }

    .warning-box {
        background: #fffbeb;
        border: 1px solid #fde68a;
        color: #92400e;
        padding: 15px;
        border-radius: 10px;
    }

    .danger-box {
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #991b1b;
        padding: 15px;
        border-radius: 10px;
    }

    .footer {
        text-align: center;
        color: #9ca3af;
        padding: 25px 0;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load student data
def load_students():
    if not os.path.exists(STUDENT_PATH):
        return pd.DataFrame(columns=["rollno", "name", "branch"])

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
            return pd.DataFrame(columns=["rollno", "name", "branch"])

    students.columns = [
        str(col).strip().lower()
        for col in students.columns
    ]

    rename_map = {}

    for col in students.columns:
        if col in ["roll", "roll_no", "roll number", "id"]:
            rename_map[col] = "rollno"

        elif col in ["student_name", "student name"]:
            rename_map[col] = "name"

        elif col in ["department"]:
            rename_map[col] = "branch"

    students = students.rename(columns=rename_map)

    for column in ["rollno", "name", "branch"]:
        if column not in students.columns:
            students[column] = ""

    return students[["rollno", "name", "branch"]]


students = load_students()


# Load attendance records
def load_attendance():
    if not os.path.exists(ATTENDANCE_PATH):
        return pd.DataFrame(
            columns=["Roll No", "Name", "Branch", "Date", "Time", "Status"]
        )

    try:
        df = pd.read_excel(ATTENDANCE_PATH)
        return df
    except Exception:
        return pd.DataFrame(
            columns=["Roll No", "Name", "Branch", "Date", "Time", "Status"]
        )


attendance_df = load_attendance()


# Load face detector
@st.cache_resource
def load_face_cascade():
    if not os.path.exists(CASCADE_PATH):
        return None

    cascade = cv2.CascadeClassifier(CASCADE_PATH)

    if cascade.empty():
        return None

    return cascade


face_cascade = load_face_cascade()


# Load LBPH model
@st.cache_resource
def load_recognizer():
    if not os.path.exists(TRAINER_PATH):
        return None

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(TRAINER_PATH)
        return recognizer
    except Exception:
        return None


recognizer = load_recognizer()


# Find faces in the image
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

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


# Get student details from roll number
def get_student(roll_no):
    if students.empty:
        return None

    result = students[
        students["rollno"].astype(str).str.strip()
        == str(roll_no).strip()
    ]

    if result.empty:
        return None

    return result.iloc[0]


# Check whether attendance can be marked
def save_attendance(roll_no, name, branch):
    global attendance_df

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    attendance_df = load_attendance()

    if not attendance_df.empty:
        same_student = attendance_df[
            attendance_df["Roll No"].astype(str).str.strip()
            == str(roll_no).strip()
        ].copy()

        if not same_student.empty:
            for _, row in same_student.iterrows():
                try:
                    record_date = pd.to_datetime(
                        str(row["Date"])
                    ).strftime("%Y-%m-%d")

                    record_time = str(row["Time"])

                    if "." in record_time:
                        record_time = record_time.split(".")[0]

                    record_datetime = datetime.strptime(
                        f"{record_date} {record_time}",
                        "%Y-%m-%d %H:%M:%S"
                    )

                    if now - record_datetime < timedelta(hours=1):
                        return "Re-Verified"

                except Exception:
                    continue

    new_record = pd.DataFrame(
        [{
            "Roll No": roll_no,
            "Name": name,
            "Branch": branch,
            "Date": today,
            "Time": current_time,
            "Status": "Present"
        }]
    )

    attendance_df = pd.concat(
        [attendance_df, new_record],
        ignore_index=True
    )

    os.makedirs(os.path.dirname(ATTENDANCE_PATH), exist_ok=True)

    attendance_df.to_excel(
        ATTENDANCE_PATH,
        index=False
    )

    return "Present"


# Train the face recognition model
def train_model():
    image_data = []
    labels = []

    if not os.path.exists(DATASET_PATH):
        return False, "Dataset folder not found."

    for roll_folder in os.listdir(DATASET_PATH):
        folder_path = os.path.join(
            DATASET_PATH,
            roll_folder
        )

        if not os.path.isdir(folder_path):
            continue

        try:
            label = int(roll_folder)
        except ValueError:
            continue

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(
                folder_path,
                image_name
            )

            image = cv2.imread(image_path)

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
                continue

            for x, y, w, h in faces:
                face = gray[y:y + h, x:x + w]

                image_data.append(face)
                labels.append(label)

    if not image_data:
        return False, "No face images found for training."

    try:
        model = cv2.face.LBPHFaceRecognizer_create()

        model.train(
            image_data,
            np.array(labels)
        )

        model.write(TRAINER_PATH)

        load_recognizer.clear()

        return True, "Face model trained successfully."

    except Exception as e:
        return False, str(e)


# Save a new student and train the model
def register_student(roll_no, name, branch, images):
    global students
    global recognizer

    roll_no = str(roll_no).strip()
    name = str(name).strip()
    branch = str(branch).strip()

    if not roll_no or not name or not branch:
        return False, "Please fill all student details."

    if not images:
        return False, "Please capture at least one face image."

    existing = students[
        students["rollno"].astype(str).str.strip()
        == roll_no
    ]

    if not existing.empty:
        return False, "This roll number is already registered."

    student_folder = os.path.join(
        DATASET_PATH,
        roll_no
    )

    os.makedirs(student_folder, exist_ok=True)

    saved_count = 0

    for index, image_file in enumerate(images, start=1):
        try:
            image = Image.open(image_file)
            frame = np.array(image)

            gray, faces = detect_faces(frame)

            if len(faces) == 0:
                continue

            for x, y, w, h in faces[:1]:
                face = gray[y:y + h, x:x + w]

                file_path = os.path.join(
                    student_folder,
                    f"User.{roll_no}.{index}.jpg"
                )

                cv2.imwrite(
                    file_path,
                    face
                )

                saved_count += 1

        except Exception:
            continue

    if saved_count == 0:
        return False, "No face detected. Please capture a clear face."

    new_student = pd.DataFrame(
        [{
            "rollno": roll_no,
            "name": name,
            "branch": branch
        }]
    )

    students = pd.concat(
        [students, new_student],
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

    return True, f"Student registered successfully with {saved_count} face sample(s)."


# Header
st.markdown(
    '<div class="title">🎓 Smart Attendance System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Face recognition based attendance management system</div>',
    unsafe_allow_html=True
)


# Sidebar
with st.sidebar:
    st.header("System Info")

    st.metric(
        "Registered Students",
        len(students)
    )

    st.metric(
        "Attendance Records",
        len(attendance_df)
    )

    st.divider()

    st.info(
        "Attendance for the same student cannot be marked again within 1 hour."
    )


# Main tabs
tab1, tab2, tab3 = st.tabs(
    [
        "📷 Scan Attendance",
        "➕ Register Student",
        "📋 Attendance Records"
    ]
)


# Attendance scanning
with tab1:
    st.subheader("Scan Attendance")

    st.write(
        "Capture your face using the camera to mark attendance."
    )

    camera_image = st.camera_input(
        "Take a photo"
    )

    if camera_image is not None:
        image = Image.open(camera_image)
        frame = np.array(image)

        if face_cascade is None:
            st.error(
                "Face detection model could not be loaded."
            )

        elif recognizer is None:
            st.error(
                "Face recognition model is not available."
            )

        else:
            gray, faces = detect_faces(frame)

            if len(faces) == 0:
                st.markdown(
                    """
                    <div class="warning-box">
                    No face detected. Please look directly at the camera
                    and make sure your face is clearly visible.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif len(faces) > 1:
                st.warning(
                    "Multiple faces detected. Please keep only one person in front of the camera."
                )

            else:
                x, y, w, h = faces[0]

                face = gray[y:y + h, x:x + w]

                try:
                    label, confidence = recognizer.predict(face)

                    student = get_student(label)

                    if student is None or confidence >= 85:
                        st.markdown(
                            """
                            <div class="danger-box">
                            Face not registered. Please register the student first.
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.info(
                            "Go to the 'Register Student' tab and register this student."
                        )

                    else:
                        roll_no = student["rollno"]
                        name = student["name"]
                        branch = student["branch"]

                        st.markdown(
                            f"""
                            <div class="student-card">
                                <h3>Student Recognized</h3>
                                <p><b>Name:</b> {name}</p>
                                <p><b>Roll No:</b> {roll_no}</p>
                                <p><b>Branch:</b> {branch}</p>
                                <p><b>Confidence:</b> {confidence:.2f}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        status = save_attendance(
                            roll_no,
                            name,
                            branch
                        )

                        if status == "Present":
                            st.markdown(
                                """
                                <div class="success-box">
                                ✅ Attendance marked successfully.
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        else:
                            st.markdown(
                                """
                                <div class="warning-box">
                                🔄 Re-Verified. Attendance was already marked within the last 1 hour.
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                except Exception as e:
                    st.error(
                        f"Face recognition error: {e}"
                    )


# Student registration
with tab2:
    st.subheader("Register New Student")

    st.write(
        "Enter student details and capture a few clear face samples."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        roll_no = st.text_input(
            "Roll No"
        )

    with col2:
        name = st.text_input(
            "Student Name"
        )

    with col3:
        branch = st.text_input(
            "Branch"
        )

    st.write("Capture face samples")

    sample1 = st.camera_input(
        "Face Sample 1",
        key="register_sample_1"
    )

    sample2 = st.camera_input(
        "Face Sample 2",
        key="register_sample_2"
    )

    sample3 = st.camera_input(
        "Face Sample 3",
        key="register_sample_3"
    )

    sample4 = st.camera_input(
        "Face Sample 4",
        key="register_sample_4"
    )

    sample5 = st.camera_input(
        "Face Sample 5",
        key="register_sample_5"
    )

    captured_images = [
        image
        for image in [
            sample1,
            sample2,
            sample3,
            sample4,
            sample5
        ]
        if image is not None
    ]

    if captured_images:
        st.success(
            f"{len(captured_images)} face sample(s) captured."
        )

    if st.button(
        "Register Student",
        type="primary",
        use_container_width=True
    ):
        with st.spinner("Registering student and training model..."):
            success, message = register_student(
                roll_no,
                name,
                branch,
                captured_images
            )

        if success:
            st.success(message)
            st.info(
                "Now go to Scan Attendance and scan the registered student's face."
            )
            st.rerun()
        else:
            st.error(message)


# Attendance records
with tab3:
    st.subheader("Attendance Records")

    attendance_df = load_attendance()

    if attendance_df.empty:
        st.info(
            "No attendance records available yet."
        )

    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Records",
                len(attendance_df)
            )

        with col2:
            st.metric(
                "Students",
                attendance_df["Roll No"].nunique()
            )

        with col3:
            today = datetime.now().strftime("%Y-%m-%d")

            today_count = len(
                attendance_df[
                    attendance_df["Date"].astype(str).str.contains(
                        today,
                        na=False
                    )
                ]
            )

            st.metric(
                "Today's Attendance",
                today_count
            )

        st.dataframe(
            attendance_df,
            use_container_width=True,
            hide_index=True
        )

        try:
            with open(
                ATTENDANCE_PATH,
                "rb"
            ) as file:
                st.download_button(
                    "Download Attendance Excel",
                    data=file,
                    file_name="attendance.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

        except Exception:
            pass


# Footer
st.markdown(
    """
    <div class="footer">
        Smart Attendance System • Python • OpenCV • LBPH • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
