import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime, timedelta
from pathlib import Path

# PAGE CONFIG

st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="📸",
    layout="centered"
)

# PATH

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "trainer" / "trainer.yml"
STUDENT_PATH = BASE_DIR / "data" / "students.csv"
CASCADE_PATH = BASE_DIR / "haarcascade" / "haarcascade_frontalface_default.xml"

ATTENDANCE_DIR = BASE_DIR / "attendance"
ATTENDANCE_FILE = ATTENDANCE_DIR / "attendance.xlsx"

ATTENDANCE_DIR.mkdir(exist_ok=True)

# HEADER

st.title("📸 Smart Attendance System")
st.write("Face Recognition Based Attendance")

# CHECK FILES

if not MODEL_PATH.exists():
    st.error(f"❌ Model not found: {MODEL_PATH}")
    st.stop()

if not STUDENT_PATH.exists():
    st.error(f"❌ students.csv not found: {STUDENT_PATH}")
    st.stop()

if not CASCADE_PATH.exists():
    st.error(f"❌ Haar Cascade not found: {CASCADE_PATH}")
    st.stop()

# LOAD LBPH MODEL

try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))

except Exception as e:
    st.error(f"❌ Error loading LBPH model: {e}")
    st.stop()

# LOAD HAAR CASCADE

try:
    face_cascade = cv2.CascadeClassifier(
        str(CASCADE_PATH)
    )

    if face_cascade.empty():
        st.error("❌ Haar Cascade could not be loaded.")
        st.stop()

except Exception as e:
    st.error(f"❌ Error loading Haar Cascade: {e}")
    st.stop()

# LOAD STUDENTS CSV

try:
    # Automatically detects comma OR tab-separated CSV
    students = pd.read_csv(
        STUDENT_PATH,
        sep=None,
        engine="python"
    )

    # Clean column names
    students.columns = (
        students.columns
        .astype(str)
        .str.strip()
        .str.lower()
    )

except Exception as e:
    st.error(f"❌ Could not read students.csv: {e}")
    st.stop()

# FIND ID AND NAME COLUMNS

id_col = None
name_col = None
branch_col = None

for col in students.columns:

    clean_col = str(col).strip().lower()

    if clean_col in [
        "id",
        "rollno",
        "roll_no",
        "roll number",
        "rollnumber",
        "student_id",
        "studentid"
    ]:
        id_col = col

    if clean_col in [
        "name",
        "student_name",
        "student name",
        "studentname"
    ]:
        name_col = col

    if clean_col == "branch":
        branch_col = col

# VALIDATE COLUMNS

if id_col is None or name_col is None:

    st.error(
        "❌ Could not identify student ID/name columns."
    )

    st.write(
        "Found columns:",
        list(students.columns)
    )

    st.stop()

# GET STUDENT DETAILS

def get_student_details(student_id):

    rows = students[
        students[id_col]
        .astype(str)
        .str.strip()
        == str(student_id).strip()
    ]

    if len(rows) == 0:
        return None, None

    student_name = str(
        rows.iloc[0][name_col]
    ).strip()

    student_branch = ""

    if branch_col is not None:
        student_branch = str(
            rows.iloc[0][branch_col]
        ).strip()

    return student_name, student_branch


# MARK ATTENDANCE

def mark_attendance(
    student_id,
    student_name,
    student_branch
):

    now = datetime.now()
    
    # CHECK EXISTING ATTENDANCE

    if ATTENDANCE_FILE.exists():

        try:

            old_df = pd.read_excel(
                ATTENDANCE_FILE
            )

            if len(old_df) > 0:

                existing_id_col = None
                timestamp_col = None

                # Find RollNo column
                for col in old_df.columns:

                    clean_col = (
                        str(col)
                        .strip()
                        .lower()
                    )

                    if clean_col in [
                        "rollno",
                        "roll_no",
                        "roll number",
                        "rollnumber",
                        "id",
                        "student_id"
                    ]:
                        existing_id_col = col
                        break

                # Find timestamp column
                for col in old_df.columns:

                    clean_col = (
                        str(col)
                        .strip()
                        .lower()
                    )

                    if clean_col in [
                        "timestamp",
                        "datetime",
                        "date_time"
                    ]:
                        timestamp_col = col
                        break

                # ONE HOUR DUPLICATE CHECK

                if (
                    existing_id_col is not None
                    and timestamp_col is not None
                ):

                    previous = old_df[
                        old_df[existing_id_col]
                        .astype(str)
                        .str.strip()
                        == str(student_id).strip()
                    ]

                    if len(previous) > 0:

                        previous_time = pd.to_datetime(
                            previous.iloc[-1][timestamp_col],
                            errors="coerce"
                        )

                        if pd.notna(previous_time):

                            time_difference = (
                                now - previous_time
                            )

                            if time_difference < timedelta(
                                hours=1
                            ):

                                return (
                                    False,
                                    "Re-Verified ⚠️",
                                    previous_time
                                )

        except Exception as e:

            st.warning(
                f"Attendance history warning: {e}"
            )

    # NEW ATTENDANCE RECORD

    new_record = pd.DataFrame(
        [
            {
                "RollNo": student_id,
                "Name": student_name,
                "Branch": student_branch,
                "Date": now.strftime(
                    "%Y-%m-%d"
                ),
                "Time": now.strftime(
                    "%H:%M:%S"
                ),
                "Timestamp": now
            }
        ]
    )

    # SAVE ATTENDANCE

    if ATTENDANCE_FILE.exists():

        try:

            old_df = pd.read_excel(
                ATTENDANCE_FILE
            )

            final_df = pd.concat(
                [
                    old_df,
                    new_record
                ],
                ignore_index=True
            )

        except Exception:

            final_df = new_record

    else:

        final_df = new_record

    final_df.to_excel(
        ATTENDANCE_FILE,
        index=False
    )

    return (
        True,
        "Present ✅",
        now
    )

# CAMERA

st.subheader("📷 Face Scanner")

camera_image = st.camera_input(
    "Take a photo for attendance"
)

# PROCESS CAMERA IMAGE

if camera_image is not None:

    try:

        # Read image
        image = Image.open(
            camera_image
        ).convert("RGB")

        frame = np.array(image)

        # RGB → Grayscale
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2GRAY
        )

        # FACE DETECTION

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        # NO FACE

        if len(faces) == 0:

            st.warning(
                "⚠️ No face detected."
            )

            st.info(
                "Please look directly at the camera "
                "and make sure your face is clearly visible."
            )

        # MULTIPLE FACES

        elif len(faces) > 1:

            st.warning(
                "⚠️ Multiple faces detected."
            )

            st.info(
                "Please keep only one person in the frame."
            )

        # SINGLE FACE

        else:

            x, y, w, h = faces[0]

            face = gray[
                y:y + h,
                x:x + w
            ]

            # LBPH PREDICTION

            try:

                student_id, confidence = (
                    recognizer.predict(face)
                )

                # Lower = better
                RECOGNITION_THRESHOLD = 85

                # FACE RECOGNIZED
                
                if confidence <= RECOGNITION_THRESHOLD:

                    student_name, student_branch = (
                        get_student_details(
                            student_id
                        )
                    )

                    # UNKNOWN STUDENT

                    if student_name is None:

                        st.error(
                            f"❌ Face recognized with ID "
                            f"{student_id}, but student "
                            f"was not found in students.csv."
                        )
                        
                    # STUDENT FOUND
                    
                    else:

                        status, message, timestamp = (
                            mark_attendance(
                                student_id,
                                student_name,
                                student_branch
                            )
                        )

                        # STUDENT INFO

                        st.success(
                            f"👤 Student: {student_name}"
                        )

                        st.info(
                            f"🎓 Roll No: {student_id}"
                        )

                        if student_branch:
                            st.info(
                                f"🏫 Branch: {student_branch}"
                            )

                        st.metric(
                            "LBPH Confidence",
                            f"{confidence:.2f}"
                        )

                        # ATTENDANCE STATUS

                        if status:

                            st.success(
                                "✅ Attendance Status: Present"
                            )

                            st.success(
                                f"🕒 Time: "
                                f"{timestamp.strftime('%H:%M:%S')}"
                            )

                        else:

                            st.warning(
                                "⚠️ Attendance Status: "
                                "Re-Verified"
                            )

                            st.info(
                                "This student was already "
                                "marked present within "
                                "the last 1 hour."
                            )
                            
                # FACE NOT RECOGNIZED

                else:

                    st.error(
                        "❌ Face not recognized."
                    )

                    st.caption(
                        f"LBPH confidence: "
                        f"{confidence:.2f}"
                    )

                    st.caption(
                        "Lower confidence values indicate "
                        "a better LBPH match."
                    )

            except Exception as e:

                st.error(
                    f"❌ Face recognition error: {e}"
                )

    except Exception as e:

        st.error(
            f"❌ Image processing error: {e}"
        )


# ATTENDANCE TABLE

st.divider()

st.subheader("📋 Attendance Records")

if ATTENDANCE_FILE.exists():

    try:

        attendance_df = pd.read_excel(
            ATTENDANCE_FILE
        )

        if len(attendance_df) > 0:

            st.dataframe(
                attendance_df,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                f"Total records: {len(attendance_df)}"
            )

        else:

            st.info(
                "No attendance records yet."
            )

    except Exception as e:

        st.warning(
            f"Could not read attendance file: {e}"
        )

else:

    st.info(
        "No attendance recorded yet."
    )


# FOOTER

st.divider()

st.caption(
    "Smart Attendance System | "
    "OpenCV + LBPH Face Recognition"
)

