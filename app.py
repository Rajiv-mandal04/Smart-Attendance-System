from flask import Flask, render_template, jsonify, request, redirect
from flask_cors import CORS
import cv2
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


app = Flask(__name__)

# ============================================================
# CORS
# ============================================================

CORS(
    app,
    resources={
        r"/*": {
            "origins": "*"
        }
    }
)


# ============================================================
# PATHS
# ============================================================

STUDENTS = "data/students.csv"
ATTENDANCE = "attendance/attendance.xlsx"
TRAINER = "trainer/trainer.yml"
CASCADE = "haarcascade/haarcascade_frontalface_default.xml"
DATASET = "dataset"

CONF_THRESHOLD = 85


# ============================================================
# LOAD STUDENT DATA
# ============================================================

if os.path.exists(STUDENTS):

    try:
        students = pd.read_csv(
            STUDENTS,
            sep="\t"
        )

    except Exception as e:

        print("Student data loading error:", e)

        students = pd.DataFrame(
            columns=[
                "rollno",
                "name",
                "branch"
            ]
        )

else:

    students = pd.DataFrame(
        columns=[
            "rollno",
            "name",
            "branch"
        ]
    )


# ============================================================
# LOAD ATTENDANCE DATA
# ============================================================

if os.path.exists(ATTENDANCE):

    try:

        attendance_df = pd.read_excel(
            ATTENDANCE
        )

    except Exception as e:

        print(
            "Attendance loading error:",
            e
        )

        attendance_df = pd.DataFrame(
            columns=[
                "RollNo",
                "Name",
                "Date",
                "Time",
                "Status"
            ]
        )

else:

    attendance_df = pd.DataFrame(
        columns=[
            "RollNo",
            "Name",
            "Date",
            "Time",
            "Status"
        ]
    )


# ============================================================
# FACE DETECTOR
# ============================================================

face_cascade = cv2.CascadeClassifier(
    CASCADE
)

if face_cascade.empty():

    print(
        "WARNING: Haar Cascade could not be loaded."
    )


# ============================================================
# LBPH RECOGNIZER
# ============================================================

recognizer = cv2.face.LBPHFaceRecognizer_create()

if os.path.exists(TRAINER):

    try:

        recognizer.read(
            TRAINER
        )

        print(
            "LBPH model loaded successfully."
        )

    except Exception as e:

        print(
            "Could not load LBPH model:",
            e
        )

else:

    print(
        "LBPH trainer file not found."
    )


# ============================================================
# ATTENDANCE CACHE
# ============================================================

attendance_cache = {}


def initialize_attendance_cache():

    global attendance_cache

    if attendance_df.empty:
        return

    for _, row in attendance_df.iterrows():

        try:

            rollno = int(
                row["RollNo"]
            )

            date_str = str(
                row["Date"]
            )

            time_str = str(
                row["Time"]
            )

            dt = datetime.strptime(
                f"{date_str} {time_str}",
                "%Y-%m-%d %H:%M:%S"
            )

            attendance_cache[rollno] = {
                "last_time": dt,
                "date": date_str
            }

        except Exception:
            pass


initialize_attendance_cache()


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# ATTENDANCE PAGE
# ============================================================

@app.route("/attendance")
def attendance_page():

    return render_template(
        "attendance.html"
    )


# ============================================================
# REGISTER PAGE
# ============================================================

@app.route("/register")
def register():

    return render_template(
        "register.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "message": "Smart Attendance API is running"
    })


# ============================================================
# RECOGNIZE FACE
# ============================================================

@app.route(
    "/recognize",
    methods=["POST"]
)
def recognize():

    if "image" not in request.files:

        return jsonify({
            "status": "fail",
            "msg": "No image received"
        }), 400

    file = request.files["image"]

    try:

        image_bytes = file.read()

        np_arr = np.frombuffer(
            image_bytes,
            np.uint8
        )

        frame = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            return jsonify({
                "status": "fail",
                "msg": "Invalid image"
            }), 400

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5
        )

        if len(faces) == 0:

            return jsonify({
                "status": "fail",
                "msg": "No face detected"
            })

        # Largest detected face
        x, y, w, h = max(
            faces,
            key=lambda face: face[2] * face[3]
        )

        face = gray[
            y:y + h,
            x:x + w
        ]

        # Check whether model is available
        if not os.path.exists(TRAINER):

            return jsonify({
                "status": "fail",
                "msg": "Face recognition model not available",
                "face": {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h)
                },
                "image_width": int(frame.shape[1]),
                "image_height": int(frame.shape[0])
            })

        rollno, confidence = recognizer.predict(
            face
        )

        print(
            f"Prediction: Roll={rollno}, "
            f"Confidence={confidence:.2f}"
        )

        face_data = {
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h)
        }

        image_data = {
            "image_width": int(
                frame.shape[1]
            ),
            "image_height": int(
                frame.shape[0]
            )
        }

        # ====================================================
        # UNKNOWN FACE
        # ====================================================

        if confidence >= CONF_THRESHOLD:

            return jsonify({
                "status": "unknown",
                "msg": "Unknown face",
                "confidence": round(
                    float(confidence),
                    2
                ),
                "face": face_data,
                **image_data
            })

        # ====================================================
        # FIND STUDENT
        # ====================================================

        student = students[
            students["rollno"] == rollno
        ]

        if student.empty:

            return jsonify({
                "status": "unknown",
                "msg": "Student not registered",
                "confidence": round(
                    float(confidence),
                    2
                ),
                "face": face_data,
                **image_data
            })

        name = str(
            student["name"].values[0]
        )

        # ====================================================
        # RECOGNIZED
        # ====================================================

        return jsonify({
            "status": "recognized",
            "rollno": int(rollno),
            "name": name,
            "confidence": round(
                float(confidence),
                2
            ),
            "face": face_data,
            **image_data
        })

    except Exception as e:

        print(
            "Recognition error:",
            e
        )

        return jsonify({
            "status": "fail",
            "msg": "Recognition failed"
        }), 500


# ============================================================
# MARK ATTENDANCE
# ============================================================

@app.route(
    "/mark-attendance",
    methods=["POST"]
)
def mark_attendance():

    global attendance_df
    global attendance_cache

    data = request.get_json(
        silent=True
    )

    if not data:

        return jsonify({
            "status": "fail",
            "msg": "No data received"
        }), 400

    if "rollno" not in data:

        return jsonify({
            "status": "fail",
            "msg": "Roll number missing"
        }), 400

    try:

        rollno = int(
            data["rollno"]
        )

    except Exception:

        return jsonify({
            "status": "fail",
            "msg": "Invalid roll number"
        }), 400

    student = students[
        students["rollno"] == rollno
    ]

    if student.empty:

        return jsonify({
            "status": "fail",
            "msg": "Student not found"
        }), 404

    name = str(
        student["name"].values[0]
    )

    now = datetime.now()

    current_date = now.strftime(
        "%Y-%m-%d"
    )

    current_time = now.strftime(
        "%H:%M:%S"
    )

    # ========================================================
    # 1 HOUR RE-VERIFICATION CHECK
    # ========================================================

    if rollno in attendance_cache:

        last_record = attendance_cache[
            rollno
        ]

        last_date = last_record[
            "date"
        ]

        last_time = last_record[
            "last_time"
        ]

        if last_date == current_date:

            time_difference = (
                now - last_time
            )

            if time_difference < timedelta(
                hours=1
            ):

                return jsonify({
                    "status": "reverified",
                    "rollno": rollno,
                    "name": name,
                    "time": current_time,
                    "msg": "Already verified within 1 hour"
                })

    # ========================================================
    # NEW ATTENDANCE
    # ========================================================

    new_entry = {
        "RollNo": rollno,
        "Name": name,
        "Date": current_date,
        "Time": current_time,
        "Status": "Present"
    }

    attendance_df = pd.concat(
        [
            attendance_df,
            pd.DataFrame(
                [new_entry]
            )
        ],
        ignore_index=True
    )

    os.makedirs(
        os.path.dirname(
            ATTENDANCE
        ),
        exist_ok=True
    )

    attendance_df.to_excel(
        ATTENDANCE,
        index=False
    )

    attendance_cache[rollno] = {
        "last_time": now,
        "date": current_date
    }

    print(
        f"Attendance marked: "
        f"{rollno} - {name}"
    )

    return jsonify({
        "status": "success",
        "rollno": rollno,
        "name": name,
        "time": current_time,
        "msg": "Attendance marked successfully"
    })


# ============================================================
# CAPTURE FACE
# ============================================================

@app.route(
    "/capture-face",
    methods=["POST"]
)
def capture_face():

    if "image" not in request.files:

        return jsonify({
            "status": "fail",
            "msg": "No image received"
        }), 400

    rollno = request.form.get(
        "rollno",
        ""
    ).strip()

    if not rollno:

        return jsonify({
            "status": "fail",
            "msg": "Roll No required"
        }), 400

    try:

        int(rollno)

    except ValueError:

        return jsonify({
            "status": "fail",
            "msg": "Invalid Roll No"
        }), 400

    file = request.files["image"]

    try:

        image_bytes = file.read()

        np_arr = np.frombuffer(
            image_bytes,
            np.uint8
        )

        frame = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            return jsonify({
                "status": "fail",
                "msg": "Invalid image"
            }), 400

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        if len(faces) == 0:

            return jsonify({
                "status": "fail",
                "msg": "No face detected"
            })

        x, y, w, h = max(
            faces,
            key=lambda face: face[2] * face[3]
        )

        face_img = gray[
            y:y + h,
            x:x + w
        ]

        student_path = os.path.join(
            DATASET,
            str(rollno)
        )

        os.makedirs(
            student_path,
            exist_ok=True
        )

        existing_images = [
            f
            for f in os.listdir(
                student_path
            )
            if f.lower().endswith(".jpg")
        ]

        image_number = len(
            existing_images
        )

        if image_number >= 100:

            return jsonify({
                "status": "complete",
                "count": 100,
                "total": 100,
                "msg": "100 face images already captured"
            })

        face_path = os.path.join(
            student_path,
            f"{image_number}.jpg"
        )

        cv2.imwrite(
            face_path,
            face_img
        )

        print(
            f"Captured face: "
            f"{rollno} "
            f"{image_number + 1}/100"
        )

        return jsonify({
            "status": "success",
            "count": image_number + 1,
            "total": 100,
            "msg": (
                f"Face captured "
                f"{image_number + 1}/100"
            )
        })

    except Exception as e:

        print(
            "Capture error:",
            e
        )

        return jsonify({
            "status": "fail",
            "msg": "Face capture failed"
        }), 500


# ============================================================
# SAVE STUDENT
# ============================================================

@app.route(
    "/save-student",
    methods=["POST"]
)
def save_student():

    global students

    rollno = request.form.get(
        "rollno",
        ""
    ).strip()

    name = request.form.get(
        "name",
        ""
    ).strip()

    branch = request.form.get(
        "branch",
        ""
    ).strip()

    if not rollno or not name or not branch:

        return "All fields are required", 400

    try:

        rollno_int = int(
            rollno
        )

    except ValueError:

        return "Roll No must be a number", 400

    # ========================================================
    # DUPLICATE ROLL NUMBER
    # ========================================================

    if not students.empty:

        duplicate = students[
            students["rollno"] == rollno_int
        ]

        if not duplicate.empty:

            return (
                "Student with this "
                "Roll No already exists",
                409
            )

    # ========================================================
    # MAKE DATA FOLDER
    # ========================================================

    os.makedirs(
        os.path.dirname(
            STUDENTS
        ),
        exist_ok=True
    )

    # ========================================================
    # SAVE STUDENT
    # ========================================================

    with open(
        STUDENTS,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            f"{rollno}\t"
            f"{name}\t"
            f"{branch}\n"
        )

    # ========================================================
    # RELOAD STUDENTS
    # ========================================================

    students = pd.read_csv(
        STUDENTS,
        sep="\t"
    )

    print(
        f"Student saved: "
        f"{rollno} - {name}"
    )

    # ========================================================
    # TRAIN MODEL
    # ========================================================

    result = train_model()

    print(
        "Training result:",
        result
    )

    return redirect("/")


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    faces = []
    ids = []

    if not os.path.exists(
        DATASET
    ):

        return "Dataset folder not found"

    for foldername in os.listdir(
        DATASET
    ):

        if not foldername.isdigit():
            continue

        folder_path = os.path.join(
            DATASET,
            foldername
        )

        if not os.path.isdir(
            folder_path
        ):
            continue

        for filename in os.listdir(
            folder_path
        ):

            if not filename.lower().endswith(
                ".jpg"
            ):
                continue

            img_path = os.path.join(
                folder_path,
                filename
            )

            img = cv2.imread(
                img_path,
                cv2.IMREAD_GRAYSCALE
            )

            if img is not None:

                faces.append(
                    img
                )

                ids.append(
                    int(foldername)
                )

    if not faces:

        return "No training faces found"

    try:

        recognizer.train(
            faces,
            np.array(ids)
        )

        os.makedirs(
            os.path.dirname(
                TRAINER
            ),
            exist_ok=True
        )

        recognizer.write(
            TRAINER
        )

        print(
            f"Model trained with "
            f"{len(faces)} images."
        )

        return "Model trained successfully"

    except Exception as e:

        print(
            "Training error:",
            e
        )

        return "Training failed"


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )