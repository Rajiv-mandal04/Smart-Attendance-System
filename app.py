from flask import Flask, render_template, jsonify, request, redirect
from flask_cors import CORS
import cv2
import os
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/*": {
            "origins": "*"
        }
    }
)

# ============================================================
# CONFIG
# ============================================================

DATABASE_URL = os.environ.get("DATABASE_URL")

CASCADE = "haarcascade/haarcascade_frontalface_default.xml"

CONF_THRESHOLD = 85

# Windows: %TEMP%
# Render/Linux: /tmp
MODEL_PATH = os.path.join(
    os.environ.get("TEMP", "/tmp"),
    "trainer.yml"
)


# ============================================================
# DATABASE
# ============================================================

def get_db():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured")

    return psycopg2.connect(DATABASE_URL)


def init_database():

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id SERIAL PRIMARY KEY,
            rollno BIGINT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            branch TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id SERIAL PRIMARY KEY,
            rollno BIGINT NOT NULL,
            name TEXT NOT NULL,
            date DATE NOT NULL,
            time TIME NOT NULL,
            status TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_samples (
            id SERIAL PRIMARY KEY,
            rollno BIGINT NOT NULL,
            image BYTEA NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_models (
            id INTEGER PRIMARY KEY,
            model BYTEA NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()

    cur.close()
    conn.close()

    print("Database initialized successfully.")


# ============================================================
# FACE DETECTOR
# ============================================================

face_cascade = cv2.CascadeClassifier(CASCADE)

if face_cascade.empty():
    print("WARNING: Haar Cascade could not be loaded.")
else:
    print("Haar Cascade loaded successfully.")


# ============================================================
# LBPH
# ============================================================

recognizer = cv2.face.LBPHFaceRecognizer_create()


# ============================================================
# LOAD MODEL FROM DATABASE
# ============================================================

def load_model_from_database():

    global recognizer

    try:

        conn = get_db()
        cur = conn.cursor()

        cur.execute("""
            SELECT model
            FROM face_models
            WHERE id = 1
        """)

        row = cur.fetchone()

        cur.close()
        conn.close()

        if not row:
            print("No trained model found in database.")
            return False

        model_bytes = bytes(row[0])

        # Make sure temp directory exists
        model_directory = os.path.dirname(MODEL_PATH)

        if model_directory:
            os.makedirs(model_directory, exist_ok=True)

        with open(MODEL_PATH, "wb") as f:
            f.write(model_bytes)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_PATH)

        print("LBPH model loaded from database.")

        return True

    except Exception as e:

        print("Model loading error:", e)

        return False


# ============================================================
# SAVE MODEL TO DATABASE
# ============================================================

def save_model_to_database():

    try:

        if not os.path.exists(MODEL_PATH):
            return False

        with open(MODEL_PATH, "rb") as f:
            model_bytes = f.read()

        conn = get_db()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO face_models
            (id, model, updated_at)
            VALUES (1, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (id)
            DO UPDATE SET
                model = EXCLUDED.model,
                updated_at = CURRENT_TIMESTAMP
        """, (psycopg2.Binary(model_bytes),))

        conn.commit()

        cur.close()
        conn.close()

        print("LBPH model saved to database.")

        return True

    except Exception as e:

        print("Model database save error:", e)

        return False


# ============================================================
# LOAD DATABASE
# ============================================================

try:

    init_database()
    load_model_from_database()

except Exception as e:

    print("Database startup error:", e)


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
# HEALTH
# ============================================================

@app.route("/health")
def health():

    database_status = "unknown"

    try:

        conn = get_db()
        cur = conn.cursor()

        cur.execute("SELECT 1")
        cur.fetchone()

        cur.close()
        conn.close()

        database_status = "connected"

    except Exception:

        database_status = "error"

    return jsonify({
        "status": "ok",
        "message": "Smart Attendance API is running",
        "database": database_status
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

    try:

        file = request.files["image"]

        image_bytes = file.read()

        if not image_bytes:

            return jsonify({
                "status": "fail",
                "msg": "Empty image received"
            }), 400

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

        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        print(
            f"Recognition frame: "
            f"{frame.shape[1]}x{frame.shape[0]}, "
            f"faces={len(faces)}"
        )

        if len(faces) == 0:

            return jsonify({
                "status": "fail",
                "msg": "No face detected",
                "faces_found": 0,
                "image_width": int(frame.shape[1]),
                "image_height": int(frame.shape[0])
            })

        x, y, w, h = max(
            faces,
            key=lambda face: face[2] * face[3]
        )

        face = gray[
            y:y + h,
            x:x + w
        ]

        face_data = {
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h)
        }

        image_data = {
            "image_width": int(frame.shape[1]),
            "image_height": int(frame.shape[0])
        }

        # Check model
        if not os.path.exists(MODEL_PATH):

            return jsonify({
                "status": "fail",
                "msg": "Face recognition model not available",
                "face": face_data,
                **image_data
            })

        try:

            rollno, confidence = recognizer.predict(
                face
            )

        except Exception as e:

            print(
                "Prediction error:",
                e
            )

            return jsonify({
                "status": "fail",
                "msg": "Face recognition model error",
                "face": face_data,
                **image_data
            })

        print(
            f"Prediction: "
            f"Roll={rollno}, "
            f"Confidence={confidence:.2f}"
        )

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

        # Find student
        conn = get_db()

        cur = conn.cursor(
            cursor_factory=RealDictCursor
        )

        cur.execute("""
            SELECT rollno, name, branch
            FROM students
            WHERE rollno = %s
        """, (int(rollno),))

        student = cur.fetchone()

        cur.close()
        conn.close()

        if not student:

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

        return jsonify({
            "status": "recognized",
            "rollno": int(student["rollno"]),
            "name": student["name"],
            "branch": student["branch"],
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

    conn = get_db()

    cur = conn.cursor(
        cursor_factory=RealDictCursor
    )

    cur.execute("""
        SELECT rollno, name, branch
        FROM students
        WHERE rollno = %s
    """, (rollno,))

    student = cur.fetchone()

    if not student:

        cur.close()
        conn.close()

        return jsonify({
            "status": "fail",
            "msg": "Student not found"
        }), 404

    name = student["name"]

    now = datetime.now()

    current_date = now.date()
    current_time = now.time()

    # ========================================================
    # LAST ATTENDANCE
    # ========================================================

    cur.execute("""
        SELECT date, time
        FROM attendance
        WHERE rollno = %s
        ORDER BY created_at DESC
        LIMIT 1
    """, (rollno,))

    last_record = cur.fetchone()

    if last_record:

        last_datetime = datetime.combine(
            last_record["date"],
            last_record["time"]
        )

        if (
            now - last_datetime
        ) < timedelta(hours=1):

            cur.close()
            conn.close()

            return jsonify({
                "status": "reverified",
                "rollno": rollno,
                "name": name,
                "time": now.strftime("%H:%M:%S"),
                "msg": "Already verified within 1 hour"
            })

    # ========================================================
    # INSERT ATTENDANCE
    # ========================================================

    cur.execute("""
        INSERT INTO attendance
        (rollno, name, date, time, status)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        rollno,
        name,
        current_date,
        current_time,
        "Present"
    ))

    conn.commit()

    cur.close()
    conn.close()

    print(
        f"Attendance marked: "
        f"{rollno} - {name}"
    )

    return jsonify({
        "status": "success",
        "rollno": rollno,
        "name": name,
        "time": now.strftime("%H:%M:%S"),
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

        rollno = int(rollno)

    except ValueError:

        return jsonify({
            "status": "fail",
            "msg": "Invalid Roll No"
        }), 400

    try:

        file = request.files["image"]

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

        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
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

        # ====================================================
        # CHECK CURRENT COUNT
        # ====================================================

        conn = get_db()

        cur = conn.cursor()

        cur.execute("""
            SELECT COUNT(*)
            FROM face_samples
            WHERE rollno = %s
        """, (rollno,))

        count = cur.fetchone()[0]

        if count >= 100:

            cur.close()
            conn.close()

            return jsonify({
                "status": "complete",
                "count": 100,
                "total": 100,
                "msg": "100 face images already captured"
            })

        # ====================================================
        # ENCODE FACE
        # ====================================================

        success, encoded = cv2.imencode(
            ".jpg",
            face_img
        )

        if not success:

            cur.close()
            conn.close()

            return jsonify({
                "status": "fail",
                "msg": "Face encoding failed"
            }), 500

        image_bytes = encoded.tobytes()

        # ====================================================
        # SAVE FACE TO DATABASE
        # ====================================================

        cur.execute("""
            INSERT INTO face_samples
            (rollno, image)
            VALUES (%s, %s)
        """, (
            rollno,
            psycopg2.Binary(image_bytes)
        ))

        conn.commit()

        cur.close()
        conn.close()

        new_count = count + 1

        print(
            f"Captured face: "
            f"{rollno} "
            f"{new_count}/100"
        )

        return jsonify({
            "status": "success",
            "count": new_count,
            "total": 100,
            "msg": (
                f"Face captured "
                f"{new_count}/100"
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

        rollno = int(rollno)

    except ValueError:

        return "Roll No must be a number", 400

    conn = get_db()

    cur = conn.cursor()

    # ========================================================
    # DUPLICATE STUDENT
    # ========================================================

    cur.execute("""
        SELECT id
        FROM students
        WHERE rollno = %s
    """, (rollno,))

    if cur.fetchone():

        cur.close()
        conn.close()

        return (
            "Student with this Roll No already exists",
            409
        )

    # ========================================================
    # CHECK FACE SAMPLES
    # ========================================================

    cur.execute("""
        SELECT COUNT(*)
        FROM face_samples
        WHERE rollno = %s
    """, (rollno,))

    sample_count = cur.fetchone()[0]

    if sample_count == 0:

        cur.close()
        conn.close()

        return (
            "Please capture face images first",
            400
        )

    # ========================================================
    # SAVE STUDENT
    # ========================================================

    cur.execute("""
        INSERT INTO students
        (rollno, name, branch)
        VALUES (%s, %s, %s)
    """, (
        rollno,
        name,
        branch
    ))

    conn.commit()

    cur.close()
    conn.close()

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

    if result != "Model trained successfully":

        return (
            f"Student saved but training failed: {result}",
            500
        )

    return redirect("/")


# ============================================================
# TEMPORARY RETRAIN ENDPOINT
# ============================================================

@app.route(
    "/retrain",
    methods=["POST"]
)
def retrain():

    result = train_model()

    if result == "Model trained successfully":

        return jsonify({
            "status": "success",
            "message": result
        })

    return jsonify({
        "status": "fail",
        "message": result
    }), 500


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    global recognizer

    faces = []
    ids = []

    try:

        conn = get_db()

        cur = conn.cursor()

        cur.execute("""
            SELECT rollno, image
            FROM face_samples
            ORDER BY rollno, id
        """)

        rows = cur.fetchall()

        cur.close()
        conn.close()

        for rollno, image_bytes in rows:

            np_arr = np.frombuffer(
                bytes(image_bytes),
                np.uint8
            )

            img = cv2.imdecode(
                np_arr,
                cv2.IMREAD_GRAYSCALE
            )

            if img is not None:

                faces.append(img)
                ids.append(int(rollno))

        if not faces:

            return "No training faces found"

        print(
            f"Training with "
            f"{len(faces)} face images."
        )

        new_recognizer = (
            cv2.face.LBPHFaceRecognizer_create()
        )

        new_recognizer.train(
            faces,
            np.array(ids)
        )

        # ====================================================
        # MAKE SURE MODEL DIRECTORY EXISTS
        # ====================================================

        model_directory = os.path.dirname(MODEL_PATH)

        if model_directory:
            os.makedirs(
                model_directory,
                exist_ok=True
            )

        # ====================================================
        # SAVE TEMPORARY MODEL
        # ====================================================

        new_recognizer.write(
            MODEL_PATH
        )

        # ====================================================
        # SAVE MODEL PERMANENTLY IN POSTGRESQL
        # ====================================================

        if not save_model_to_database():

            return "Model database save failed"

        recognizer = new_recognizer

        print(
            "Model trained and permanently saved."
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