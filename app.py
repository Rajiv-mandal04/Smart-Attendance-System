from flask import Flask, render_template, request, jsonify, Response, redirect, url_for
import cv2
import pandas as pd
import os
from datetime import datetime, timedelta

app = Flask(__name__)

# ---------------- PATHS ----------------
DATASET = "dataset"
TRAINER = "trainer/trainer.yml"
STUDENTS = "data/students.csv"
ATTENDANCE = "attendance/attendance.xlsx"
CASCADE = "haarcascade/haarcascade_frontalface_default.xml"

# ---------------- FOLDERS ----------------
os.makedirs("dataset", exist_ok=True)
os.makedirs("trainer", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("attendance", exist_ok=True)

# ---------------- INITIALIZE FILES ----------------
if not os.path.exists(STUDENTS):
    pd.DataFrame(columns=["rollno","name","branch"]).to_csv(STUDENTS, sep="\t", index=False)
if not os.path.exists(ATTENDANCE):
    pd.DataFrame(columns=["RollNo","Name","Date","Time","Status"]).to_excel(ATTENDANCE, index=False)

# ---------------- LOAD DATA ----------------
students = pd.read_csv(STUDENTS, sep="\t")
attendance_df = pd.read_excel(ATTENDANCE)

# ---------------- LOAD MODELS ----------------
face_cascade = cv2.CascadeClassifier(CASCADE)
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Trainer file may not exist yet → ignore error

if os.path.exists(TRAINER):
    recognizer.read(TRAINER)

# ---------------- GLOBAL FRAME ----------------
latest_frame = None

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")  # register + attendance buttons

# ---------------- REGISTER ----------------
@app.route("/register")
def register_page():
    return render_template("register.html")

@app.route("/save-student", methods=["POST"])
def save_student():
    global students
    rollno = request.form.get("rollno")
    name = request.form.get("name")
    branch = request.form.get("branch")

    if not rollno or not name or not branch:
        return "All fields are required!"

    df = pd.read_csv(STUDENTS, sep="\t")
    if int(rollno) in df["rollno"].values:
        return "Roll No already exists ❌"

    df = pd.concat([df, pd.DataFrame({
        "rollno":[int(rollno)],
        "name":[name],
        "branch":[branch]
    })], ignore_index=True)
    df.to_csv(STUDENTS, sep="\t", index=False)
    students = df

    return redirect(url_for("attendance_page"))

# ---------------- ATTENDANCE ----------------
@app.route("/attendance")
def attendance_page():
    return render_template("attendance.html")

# ---------------- LIVE VIDEO STREAM ----------------
def gen_frames():
    global latest_frame
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        success, frame = cap.read()
        if not success:
            break

        latest_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            roi = gray[y:y+h, x:x+w]
            try:
                rollno, conf = recognizer.predict(roi)
                if conf < 100:
                    student = students[students["rollno"]==int(rollno)]
                    if student.empty:
                        label = "Unknown"
                    else:
                        name = student["name"].values[0]
                        today = datetime.now().strftime("%Y-%m-%d")
                        rec = attendance_df[(attendance_df["RollNo"]==int(rollno)) & (attendance_df["Date"]==today)]
                        label = f"{name} | {rollno}"
                        if not rec.empty:
                            last_time = datetime.strptime(rec.iloc[-1]["Time"], "%H:%M:%S")
                            if datetime.now() - last_time < timedelta(hours=1):
                                label = "VERIFIED ✅"
                    cv2.putText(frame,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                else:
                    raise Exception
            except:
                cv2.putText(frame,"Unknown",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ---------------- CAPTURE FACE ----------------
@app.route("/capture-face")
def capture_face():
    rollno = request.args.get("rollno")
    if not rollno:
        return "Roll No missing!"

    path = os.path.join(DATASET, rollno)
    os.makedirs(path, exist_ok=True)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    count = 0
    MAX_IMAGES = 50

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(path,f"{count}.jpg"), face_img)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,f"Image: {count}",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

        cv2.imshow("Capture Face", frame)
        if cv2.waitKey(1) & 0xFF==27 or count>=MAX_IMAGES:
            break

    cap.release()
    cv2.destroyAllWindows()
    return f"Face Captured ✅ for Roll No {rollno}. Now run train_model.py"

# ---------------- MARK ATTENDANCE ----------------
@app.route("/mark-attendance")
def mark_attendance():
    global latest_frame

    if latest_frame is None:
        return jsonify({
            "status": "fail",
            "msg": "Camera frame not available"
        })

    frame = latest_frame.copy()

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        1.3,
        5
    )

    # Always read latest attendance data from Excel
    attendance_df = pd.read_excel(ATTENDANCE)

    for (x, y, w, h) in faces:

        roi = gray[y:y+h, x:x+w]

        try:

            rollno, conf = recognizer.predict(roi)

            if conf < 85:

                student = students[
                    students["rollno"] == int(rollno)
                ]

                if student.empty:
                    return jsonify({
                        "status": "fail",
                        "msg": "Unknown student"
                    })

                name = student["name"].values[0]

                now = datetime.now()

                current_date = now.strftime("%Y-%m-%d")
                current_time = now.strftime("%H:%M:%S")

                # Get all previous attendance of this student
                prev = attendance_df[
                    attendance_df["RollNo"] == int(rollno)
                ]

                # =====================================
                # CHECK LAST ATTENDANCE TIME
                # =====================================

                if not prev.empty:

                    last_record = prev.iloc[-1]

                    last_datetime_string = (
                        str(last_record["Date"]) +
                        " " +
                        str(last_record["Time"])
                    )

                    # Convert Excel Date + Time into datetime
                    last_datetime = pd.to_datetime(
                        last_datetime_string,
                        errors="coerce"
                    )

                    if pd.notna(last_datetime):

                        time_difference = (
                            now - last_datetime
                        )

                        # Within 1 hour → VERIFIED
                        if time_difference < timedelta(hours=1):

                            return jsonify({
                                "status": "verified",
                                "name": name,
                                "msg": "Already marked within 1 hour"
                            })

                # =====================================
                # AFTER 1 HOUR → MARK AGAIN
                # =====================================

                new_record = pd.DataFrame([{
                    "RollNo": int(rollno),
                    "Name": name,
                    "Date": current_date,
                    "Time": current_time,
                    "Status": "Present"
                }])

                attendance_df = pd.concat(
                    [attendance_df, new_record],
                    ignore_index=True
                )

                attendance_df.to_excel(
                    ATTENDANCE,
                    index=False
                )

                return jsonify({
                    "status": "success",
                    "name": name,
                    "msg": "Attendance marked successfully"
                })

            else:

                return jsonify({
                    "status": "fail",
                    "msg": "Face not recognized"
                })

        except Exception as e:

            print("Recognition Error:", e)

            return jsonify({
                "status": "fail",
                "msg": str(e)
            })

    return jsonify({
        "status": "fail",
        "msg": "No face detected"
    })

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)
