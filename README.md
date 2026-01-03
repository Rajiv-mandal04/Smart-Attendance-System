# 🎯 Smart Attendance System.

A **Smart Attendance System** built using **Python, OpenCV, Flask, and Machine Learning** that marks attendance automatically using **face recognition**.

This system prevents **duplicate attendance within 1 hour** and shows **Re-Verified** status if the same person tries again within the restricted time.

---

## 🚀 Features

✅ Face Detection & Recognition (LBPH Algorithm)  
✅ Automatic Attendance Marking  
✅ Duplicate Attendance Prevention (1-hour rule)  
✅ Re-Verified Status for repeated attempts  
✅ Live Camera Feed (Web Interface)  
✅ Excel-based Attendance Storage  
✅ Clean & Professional UI  
✅ Flask Web Application  

---

## 🛠️ Tech Stack

- **Python**
- **OpenCV**
- **Flask**
- **NumPy**
- **Pandas**
- **Machine Learning (LBPH Face Recognizer)**
- **HTML, CSS, Bootstrap**

---

## 📂 Project Structure 

├── app.py    #Main Flask application <br>
├── train_model.py   # Train face recognition model <br>
├── attendance.py   # Attendance logic <br>
├── templates/   # HTML files <br>
├── dataset/   # Face images (ignored in GitHub) <br>
├── trainer/   # Trained model (ignored) <br>
├── data/   # Student data <br>
├── attendance/   # Attendance Excel file <br>
├── haarcascade/   # Haar Cascade files <br>
└── README.md

