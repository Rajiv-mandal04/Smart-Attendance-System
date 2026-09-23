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


# Custom professional CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #eaeaea;
    }

    section[data-testid="stSidebar"] {
        background: rgba(20, 20, 40, 0.95);
        border-right: 1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(12px);
    }

    section[data-testid="stSidebar"] * {
        color: #eaeaea !important;
    }

    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
    }

    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00c6ff, #0072ff, #8e2de2, #ff2e93);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        animation: glow 3s ease-in-out infinite alternate;
    }

    @keyframes glow {
        from { filter: drop-shadow(0 0 4px #0072ff); }
        to   { filter: drop-shadow(0 0 12px #ff2e93); }
    }

    .sub-title {
        text-align: center;
        color: #b8b8d1;
        font-size: 1rem;
        margin-bottom: 2rem;
        letter-spacing: 1px;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }

    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(0,198,255,0.6);
        box-shadow: 0 12px 30px rgba(0,198,255,0.25);
    }

    .metric-label {
        font-size: 0.85rem;
        color: #9aa0b4;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 500;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00c6ff, #8e2de2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 0.3rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #0072ff, #8e2de2);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1.6rem;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 6px 18px rgba(0,114,255,0.35);
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 26px rgba(142,45,226,0.55);
        background: linear-gradient(135deg, #8e2de2, #ff2e93);
    }

    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stNumberInput > div > div > input {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
    }

    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .stAlert {
        border-radius: 12px;
    }

    hr {
        border-color: rgba(255,255,255,0.1);
    }

    .info-badge {
        display: inline-block;
        background: rgba(0,198,255,0.15);
        color: #00c6ff;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        border: 1px solid rgba(0,198,255,0.3);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STUDENT_PATH = os.path.join(BASE_DIR, "data", "students.csv")
ATTENDANCE_PATH = os.path.join(BASE_DIR, "attendance", "attendance.csv")
TRAINER_PATH = os.path.join(BASE_DIR, "trainer", "trainer.yml")


# Always return fresh current timestamp
def get_current_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")


def get_current_time():
    return datetime.now().strftime("%H:%M:%S")


# Load students data safely
def load_students():
    if not os.path.exists(STUDENT_PATH):
        return pd.DataFrame(columns=["ID", "Name", "Roll", "Department"])
    try:
        return pd.read_csv(STUDENT_PATH)
    except Exception:
        return pd.DataFrame(columns=["ID", "Name", "Roll", "Department"])


# Load attendance data safely
def load_attendance():
    if not os.path.exists(ATTENDANCE_PATH):
        return pd.DataFrame(
            columns=["ID", "Name", "Roll", "Department", "Date", "Time", "Timestamp", "Status"]
        )
    try:
        df = pd.read_csv(ATTENDANCE_PATH)
        # Backward compatibility: agar Timestamp column nahi hai
        if "Timestamp" not in df.columns:
            if "Date" in df.columns and "Time" in df.columns:
                df["Timestamp"] = df["Date"].astype(str) + " " + df["Time"].astype(str)
            else:
                df["Timestamp"] = ""
        return df
    except Exception:
        return pd.DataFrame(
            columns=["ID", "Name", "Roll", "Department", "Date", "Time", "Timestamp", "Status"]
        )


def save_attendance(df):
    df.to_csv(ATTENDANCE_PATH, index=False)


# Get real total unique students (merge of students.csv + attendance.csv)
def get_total_unique_students():
    students_df = load_students()
    attendance_df = load_attendance()

    ids = set()

    if not students_df.empty and "ID" in students_df.columns:
        ids.update(students_df["ID"].astype(str).dropna().unique())

    if not attendance_df.empty and "ID" in attendance_df.columns:
        ids.update(attendance_df["ID"].astype(str).dropna().unique())

    return len(ids)


# Mark attendance with fresh timestamp and 1 hour gap rule
def mark_attendance(student_id, name, roll, department, status="Present"):
    df = load_attendance()

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")

    # Check last entry for this student
    if not df.empty:
        student_rows = df[df["ID"].astype(str) == str(student_id)]

        if not student_rows.empty:
            # Get last timestamp
            last_row = student_rows.sort_values("Timestamp").iloc[-1]
            last_ts_str = str(last_row.get("Timestamp", ""))

            try:
                last_dt = datetime.strptime(last_ts_str, "%Y-%m-%d %H:%M:%S")
                diff = now - last_dt

                # Rule: 1 hour gap mandatory
                if diff < timedelta(hours=1):
                    remaining = timedelta(hours=1) - diff
                    mins = int(remaining.total_seconds() // 60)
                    secs = int(remaining.total_seconds() % 60)
                    return False, f"Wait {mins}m {secs}s (1 hour rule)"
            except Exception:
                pass

    # Fresh timestamp entry
    new_row = {
        "ID": student_id,
        "Name": name,
        "Roll": roll,
        "Department": department,
        "Date": now.strftime("%Y-%m-%d"),
        "Time": now.strftime("%H:%M:%S"),
        "Timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "Status": status,
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_attendance(df)
    return True, new_row["Timestamp"]


# Sidebar navigation
with st.sidebar:
    st.markdown("## 🎓 Smart Attendance")

    menu = st.radio(
        "Navigation",
        ["🏠 Dashboard", "📸 Mark Attendance", "📊 Attendance Records", "👥 Students", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Show real counts (fixed bug)
    total_students_sidebar = get_total_unique_students()
    total_records_sidebar = len(load_attendance())

    st.markdown(
        f"""
        <div class="metric-card" style="padding:1rem; margin-bottom:0.8rem;">
            <div class="metric-label">Registered Students</div>
            <div class="metric-value" style="font-size:1.5rem;">{total_students_sidebar}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="metric-card" style="padding:1rem; margin-bottom:0.8rem;">
            <div class="metric-label">Attendance Records</div>
            <div class="metric-value" style="font-size:1.5rem;">{total_records_sidebar}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    now_sidebar = datetime.now()
    st.markdown(
        f"""
        <div class="metric-card" style="padding:1rem;">
            <div class="metric-label">Current Time</div>
            <div class="metric-value" style="font-size:1.15rem;">{now_sidebar.strftime('%H:%M:%S')}</div>
            <div style="color:#9aa0b4; font-size:0.75rem; margin-top:0.3rem;">
                {now_sidebar.strftime('%d %b %Y')}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="color:#8a8fa3; font-size:0.75rem; line-height:1.5;">
            ⏱️ A student can receive a new Present entry only after the previous attendance is older than 1 hour.
        </div>
        """,
        unsafe_allow_html=True
    )


# Header
st.markdown('<div class="main-title">Smart Attendance System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-POWERED • REAL-TIME • PROFESSIONAL</div>', unsafe_allow_html=True)


# Dashboard page
if menu == "🏠 Dashboard":
    students_df = load_students()
    attendance_df = load_attendance()

    today = get_current_date()
    today_df = (
        attendance_df[attendance_df["Date"].astype(str) == today]
        if not attendance_df.empty else pd.DataFrame()
    )

    total_students = get_total_unique_students()
    total_records = len(attendance_df)
    present_today = today_df["ID"].astype(str).nunique() if not today_df.empty else 0
    absent_today = max(total_students - present_today, 0)

    attendance_rate = (
        round((present_today / total_students) * 100, 2) if total_students > 0 else 0.0
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Students</div>
            <div class="metric-value">{total_students}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Present Today</div>
            <div class="metric-value">{present_today}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Absent Today</div>
            <div class="metric-value">{absent_today}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Attendance Rate</div>
            <div class="metric-value">{attendance_rate}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Overview row
    st.markdown("### 📋 System Overview")

    o1, o2, o3 = st.columns(3)
    with o1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Records</div>
            <div class="metric-value">{total_records}</div>
        </div>
        """, unsafe_allow_html=True)
    with o2:
        unique_in_attendance = (
            attendance_df["ID"].astype(str).nunique() if not attendance_df.empty else 0
        )
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Unique Students</div>
            <div class="metric-value">{unique_in_attendance}</div>
        </div>
        """, unsafe_allow_html=True)
    with o3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Today's Attendance</div>
            <div class="metric-value">{len(today_df)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 📅 Today's Attendance Log")
    if today_df.empty:
        st.info("Aaj tak koi attendance mark nahi hui. 'Mark Attendance' tab pe jao aur start karo.")
    else:
        st.dataframe(
            today_df.sort_values("Timestamp", ascending=False),
            use_container_width=True,
            hide_index=True
        )


# Mark attendance page
elif menu == "📸 Mark Attendance":
    st.markdown("### 📸 Mark Attendance")
    st.markdown(
        '<span class="info-badge">⏱️ 1 hour gap required between two Present entries</span>',
        unsafe_allow_html=True
    )

    students_df = load_students()

    if students_df.empty:
        st.warning("⚠️ Koi student nahi mila. Pehle 'Students' tab se add karo.")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Manual Entry")

            student_options = {
                f"{row['Name']} ({row.get('Roll', 'N/A')})": idx
                for idx, row in students_df.iterrows()
            }

            selected = st.selectbox("Select Student", list(student_options.keys()))
            status = st.selectbox("Status", ["Present", "Absent", "Late"])

            if st.button("✅ Mark Attendance"):
                row = students_df.iloc[student_options[selected]]
                ok, msg = mark_attendance(
                    row.get("ID", student_options[selected]),
                    row.get("Name", ""),
                    row.get("Roll", ""),
                    row.get("Department", ""),
                    status
                )
                if ok:
                    st.success(f"✅ {row['Name']} ki attendance mark ho gayi — {msg}")
                    st.balloons()
                else:
                    st.warning(f"⏳ {row['Name']} — {msg}")

        with col2:
            st.markdown("#### Camera Capture")
            img_file = st.camera_input("Capture your photo")

            if img_file is not None:
                image = Image.open(img_file)
                st.image(image, caption="Captured Image", use_container_width=True)
                st.info("Face recognition logic yahan integrate kar sakte ho.")


# Attendance records page
elif menu == "📊 Attendance Records":
    st.markdown("### 📊 Attendance Records")

    df = load_attendance()

    if df.empty:
        st.info("Abhi tak koi attendance record nahi hai.")
    else:
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            dates = ["All"] + sorted(df["Date"].astype(str).unique().tolist(), reverse=True)
            date_filter = st.selectbox("Filter by Date", dates)

        with col2:
            depts = ["All"] + sorted(df["Department"].astype(str).dropna().unique().tolist())
            dept_filter = st.selectbox("Filter by Department", depts)

        with col3:
            statuses = ["All"] + sorted(df["Status"].astype(str).dropna().unique().tolist())
            status_filter = st.selectbox("Filter by Status", statuses)

        filtered = df.copy()
        if date_filter != "All":
            filtered = filtered[filtered["Date"].astype(str) == date_filter]
        if dept_filter != "All":
            filtered = filtered[filtered["Department"].astype(str) == dept_filter]
        if status_filter != "All":
            filtered = filtered[filtered["Status"].astype(str) == status_filter]

        st.markdown(f"**{len(filtered)}** records found")
        st.dataframe(
            filtered.sort_values("Timestamp", ascending=False),
            use_container_width=True,
            hide_index=True
        )

        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            data=csv,
            file_name=f"attendance_{get_current_date()}.csv",
            mime="text/csv"
        )


# Students page
elif menu == "👥 Students":
    st.markdown("### 👥 Registered Students")

    df = load_students()

    if df.empty:
        st.info("Koi student registered nahi hai.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### ➕ Add New Student")

    with st.form("add_student"):
        c1, c2 = st.columns(2)
        with c1:
            sid = st.text_input("Student ID")
            name = st.text_input("Full Name")
        with c2:
            roll = st.text_input("Roll Number")
            dept = st.text_input("Department")

        submitted = st.form_submit_button("Add Student")

        if submitted:
            if not all([sid, name, roll, dept]):
                st.error("Saare fields bharo bhai.")
            else:
                new_row = pd.DataFrame([{
                    "ID": sid, "Name": name, "Roll": roll, "Department": dept
                }])
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(STUDENT_PATH, index=False)
                st.success(f"✅ {name} add ho gaya!")
                st.rerun()


# About page
elif menu == "ℹ️ About":
    st.markdown("### ℹ️ About This System")
    st.markdown(f"""
    **Smart Attendance System** ek AI-powered attendance solution hai jo:

    - 📸 Face recognition se attendance mark karta hai
    - 🕒 Har entry pe fresh current timestamp use karta hai
    - ⏱️ Ek student 1 hour ke andar dobara Present mark nahi kar sakta
    - 📊 Real-time dashboard aur analytics deta hai
    - 💾 CSV export support karta hai

    ---

    **Current Server Time:** `{get_current_timestamp()}`
    """)


# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align:center; color:#8a8fa3; font-size:0.85rem; padding:1rem 0;">
        🎓 Smart Attendance System • Built with ❤️ using Streamlit
        <br>
        <span style="font-size:0.75rem;">Last refreshed: {get_current_timestamp()}</span>
    </div>
    """,
    unsafe_allow_html=True
)
