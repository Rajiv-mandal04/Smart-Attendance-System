import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime, timedelta
from pathlib import Path
import os

st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="📸",
    layout="centered"
)

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "trainer" / "trainer.yml"
STUDENT_PATH = BASE_DIR / "data" / "students.csv"
CASCADE_PATH = BASE_DIR / "haarcascade" / "haarcascade_frontalface_default.xml"
ATTENDANCE_DIR = BASE_DIR / "attendance"
ATTENDANCE_FILE = ATTENDANCE_DIR / "attendance.xlsx"

ATTENDANCE_DIR.mkdir(exist_ok=True)

st.title("📸 Smart Attendance System")
st.write("Face Recognition Based Attendance")

# -----------------------------
# CHECK REQUIRED FILES
# -----------------------------
if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()

if not STUDENT_PATH.exists():
    st.error(f"Student file not found: {STUDENT_PATH}")
    st.stop()

if not CASCADE_PATH.exists():
    st.error(f"Haar Cascade not found: {CASCADE_PATH}")
    st.stop()

# -----------------------------
# LOAD MODEL
# -----------------------------
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))

    face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))

except Exception as e:
    st.error(f"Error loading face recognition model: {e}")
    st.stop()

# -----------------------------
# LOAD STUDENTS
# -----------------------------
students = pd.read_csv(STUDENT_PATH)

# Normalize column names
students.columns = [str(c).strip().lower() for c in students.columns]

# Try to detect ID/name columns
id_col = None
name_col = None

for col in students.columns:
    if col in ["id", "rollno", "roll_no", "roll number", "rollnumber"]:
        id_col = col
    if col in ["name", "student_name", "student name"]:
        name_col = col

if id_col is None or name_col is None:
    st.error(
        f"Could not identify student ID/name columns. "
        f"Found columns: {list(students.columns)}"
    )
    st.stop()


def get_student_name(student_id):
    rows = students[students[id_col].astype(str) == str(student_id)]

    if len(rows) > 0:
        return str(rows.iloc[0][name_col])

    return "Unknown Student"


# -----------------------------
# ATTENDANCE FUNCTION
# -----------------------------
def mark_attendance(student_id, student_name):

    now = datetime.now()

    if ATTENDANCE_FILE.exists():

        try:
            df = pd.read_excel(ATTENDANCE_FILE)

            if len(df) > 0:

                # Find ID column
                existing_id_col = None

                for col in df.columns:
                    if str(col).lower() in [
                        "id",
                        "rollno",
                        "roll_no",
                        "roll number",
                        "rollnumber"
                    ]:
                        existing_id_col = col
                        break

                if existing_id_col is not None:

                    previous = df[
                        df[existing_id_col].astype(str)
                        == str(student_id)
                    ]

                    if len(previous) > 0:

                        # Find datetime column
                        time_col = None

                        for col in df.columns:
                            if str(col).lower() in [
                                "timestamp",
                                "datetime",
                                "date",
                                "time"
                            ]:
                                time_col = col
                                break

                        if time_col is not None:

                            previous_time = pd.to_datetime(
                                previous.iloc[-1][time_col],
                                errors="coerce"
                            )

                            if pd.notna(previous_time):

                                if now - previous_time < timedelta(hours=1):

                                    return (
                                        False,
                                        "Re-Verified ⚠️",
                                        previous_time
                                    )

        except Exception:
            pass

    # New attendance
    new_record = pd.DataFrame(
        [{
            "RollNo": student_id,
            "Name": student_name,
            "Date": now.strftime("%Y-%m-%d"),
            "Time": now.strftime("%H:%M:%S"),
            "Timestamp": now
        }]
    )

    if ATTENDANCE_FILE.exists():

        try:
            old_df = pd.read_excel(ATTENDANCE_FILE)
            final_df = pd.concat(
                [old_df, new_record],
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

    return True, "Present ✅", now


# -----------------------------
# CAMERA
# -----------------------------
st.subheader("📷 Scan Face")

camera_image = st.camera_input(
    "Take a photo for attendance"
)

if camera_image is not None:

    image = Image.open(camera_image).convert("RGB")

    frame = np.array(image)

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:

        st.warning(
            "No face detected. Please look directly at the camera."
        )

    elif len(faces) > 1:

        st.warning(
            "Multiple faces detected. Please keep only one person in frame."
        )

    else:

        x, y, w, h = faces[0]

        face = gray[y:y+h, x:x+w]

        try:

            student_id, confidence = recognizer.predict(face)

            # LBPH confidence:
            # lower = better
            if confidence <= 85:

                student_name = get_student_name(student_id)

                if student_name == "Unknown Student":

                    st.error(
                        f"Face recognized but student ID {student_id} "
                        "was not found in students.csv"
                    )

                else:

                    status, message, timestamp = mark_attendance(
                        student_id,
                        student_name
                    )

                    st.success(
                        f"Student: {student_name}"
                    )

                    st.info(
                        f"Roll No: {student_id}"
                    )

                    st.metric(
                        "Recognition Confidence",
                        f"{confidence:.2f}"
                    )

                    if status:
                        st.success(
                            f"Attendance Status: {message}"
                        )
                    else:
                        st.warning(
                            f"Attendance Status: {message}"
                        )

            else:

                st.error(
                    f"Face not recognized ❌ "
                    f"(confidence: {confidence:.2f})"
                )

        except Exception as e:

            st.error(
                f"Face recognition error: {e}"
            )


# -----------------------------
# ATTENDANCE VIEW
# -----------------------------
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
                use_container_width=True
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
