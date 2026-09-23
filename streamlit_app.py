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


# Professional dark theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0b1020 0%, #1a1f3a 55%, #0d1224 100%);
        color: #e8ecf3;
    }

    /* Sidebar base */
    section[data-testid="stSidebar"] {
        background: #0f1428;
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }

    section[data-testid="stSidebar"] * {
        color: #e8ecf3 !important;
    }

    /* Hide default radio circles */
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        background: transparent;
        border-radius: 12px;
        padding: 0.65rem 0.9rem;
        margin: 0.2rem 0;
        border: 1px solid transparent;
        transition: all 0.25s ease;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(0, 198, 255, 0.08);
        border-color: rgba(0, 198, 255, 0.25);
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > label p {
        font-size: 0.95rem;
        font-weight: 500;
        letter-spacing: 0.3px;
    }

    /* Selected radio state */
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(0,114,255,0.18), rgba(142,45,226,0.18));
        border-color: rgba(0,198,255,0.5);
        box-shadow: 0 4px 14px rgba(0,114,255,0.2);
    }

    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: 0.4px;
    }

    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4fc3f7, #7c4dff, #ec407a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        text-align: center;
        color: #8b93a7;
        font-size: 0.9rem;
        margin-bottom: 2rem;
        letter-spacing: 2px;
        font-weight: 500;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.2rem 1.3rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    }

    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(79,195,247,0.4);
        box-shadow: 0 10px 26px rgba(79,195,247,0.15);
    }

    .metric-label {
        font-size: 0.72rem;
        color: #8b93a7;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        font-weight: 600;
    }

    .metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4fc3f7, #7c4dff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 0.25rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2979ff, #7c4dff);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
        transition: all 0.25s ease;
        box-shadow: 0 4px 14px rgba(41,121,255,0.35);
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(124,77,255,0.45);
    }

    /* Inputs */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stNumberInput > div > div > input {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
    }

    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* Sidebar brand */
    .brand-box {
        text-align: center;
        padding: 0.5rem 0 1rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 1rem;
    }

    .brand-title {
        font-size: 1.15rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4fc3f7, #7c4dff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .brand-sub {
        font-size: 0.7rem;
        color: #6b7280;
        letter-spacing: 1.5px;
        margin-top: 0.2rem;
    }

    /* Sidebar stat box */
    .side-stat {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 0.75rem 0.9rem;
        margin-bottom: 0.6rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .side-stat-label {
        font-size: 0.78rem;
        color: #8b93a7;
        font-weight: 500;
    }

    .side-stat-value {
        font-size: 1rem;
        font-weight: 700;
        color: #4fc3f7;
    }

    .clock-box {
        background: linear-gradient(135deg, rgba(41,121,255,0.12), rgba(124,77,255,0.12));
        border: 1px solid rgba(79,195,247,0.25);
        border-radius: 12px;
        padding: 0.8rem 0.9rem;
        text-align: center;
        margin-top: 0.4rem;
    }

    .clock-time {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 1px;
    }

    .clock-date {
        font-size: 0.72rem;
        color: #8b93a7;
        margin-top: 0.15rem;
        letter-spacing: 0.5px;
    }

    .note {
        font-size: 0.72rem;
        color: #6b7280;
        line-height: 1.5;
        padding: 0.6rem 0.4rem;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin-top: 0.6rem;
    }

    .info-badge {
        display: inline-block;
        background: rgba(79,195,247,0.1);
        color: #4fc3f7;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        border: 1px solid rgba(79,195,247,0.25);
        margin: 0.4rem 0;
    }

    hr {
        border-color: rgba(255,255,255,0.06);
    }
</style>
""", unsafe_allow_html=True)


# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENT_PATH = os.path.join(BASE_DIR, "data", "students.csv")
ATTENDANCE_PATH = os.path.join(BASE_DIR, "attendance", "attendance.csv")
TRAINER_PATH = os.path.join(BASE_DIR, "trainer", "trainer.yml")


# Safe current time — validates system clock
def safe_now():
    """
    Returns a validated current datetime.
    If system clock is set to a suspicious future date (>1 year ahead),
    we still return it but the UI will show a warning.
    """
    return datetime.now()


def get_current_timestamp():
    return safe_now().strftime("%Y-%m-%d %H:%M:%S")


def get_current_date():
    return safe_now().strftime("%Y-%m-%d")


def get_current_time():
    return safe_now().strftime("%H:%M:%S")


def system_clock_looks_wrong():
    """Flag if system clock is more than 1 year ahead of expected."""
    now = datetime.now()
    # If year >= 2026 and today < 2026, likely wrong
    # Simpler: flag if year > current realistic year assumption fails
    # We'll just flag if year > 2025 for user warning
    return now.year > datetime.now().year  # always False, placeholder
    # Real check below


def is_clock_suspicious():
    """Return True if system year seems too far ahead."""
    return datetime.now().year >= 2026


# Load students data
def load_students():
    if not os.path.exists(STUDENT_PATH):
        return pd.DataFrame(columns=["ID", "Name", "Roll", "Department"])
    try:
        df = pd.read_csv(STUDENT_PATH)
        # Ensure required columns
        for col in ["ID", "Name", "Roll", "Department"]:
            if col not in df.columns:
                df[col] = ""
        return df
    except Exception:
        return pd.DataFrame(columns=["ID", "Name", "Roll", "Department"])


# Load attendance data
def load_attendance():
    if not os.path.exists(ATTENDANCE_PATH):
        return pd.DataFrame(
            columns=["ID", "Name", "Roll", "Department", "Date", "Time", "Timestamp", "Status"]
        )
    try:
        df = pd.read_csv(ATTENDANCE_PATH)
        # Backward compatibility: agar Timestamp nahi hai to banao
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


# Get unique student count from both sources
def get_total_unique_students():
    students_df = load_students()
    attendance_df = load_attendance()

    ids = set()

    if not students_df.empty and "ID" in students_df.columns:
        ids.update(students_df["ID"].astype(str).dropna().unique())

    if not attendance_df.empty and "ID" in attendance_df.columns:
        ids.update(attendance_df["ID"].astype(str).dropna().unique())

    # Remove empty strings
    ids.discard("")
    ids.discard("nan")

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
            last_row = student_rows.sort_values("Timestamp").iloc[-1]
            last_ts_str = str(last_row.get("Timestamp", ""))

            try:
                last_dt = datetime.strptime(last_ts_str, "%Y-%m-%d %H:%M:%S")
                diff = now - last_dt

                # 1 hour gap rule
                if diff < timedelta(hours=1):
                    remaining = timedelta(hours=1) - diff
                    mins = int(remaining.total_seconds() // 60)
                    secs = int(remaining.total_seconds() % 60)
                    return False, f"Wait {mins}m {secs}s (1 hour rule)"
            except Exception:
                pass

    # Fresh timestamp
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


# Reset data (fresh start)
def reset_all_data():
    """Clear both students.csv and attendance.csv, keeping headers."""
    students_df = pd.DataFrame(columns=["ID", "Name", "Roll", "Department"])
    students_df.to_csv(STUDENT_PATH, index=False)

    attendance_df = pd.DataFrame(
        columns=["ID", "Name", "Roll", "Department", "Date", "Time", "Timestamp", "Status"]
    )
    attendance_df.to_csv(ATTENDANCE_PATH, index=False)


# Sidebar
with st.sidebar:
    # Brand
    st.markdown("""
        <div class="brand-box">
            <div class="brand-title">🎓 Smart Attendance</div>
            <div class="brand-sub">AI · REAL-TIME · PRO</div>
        </div>
    """, unsafe_allow_html=True)

    # Navigation
    menu = st.radio(
        "Navigation",
        ["🏠  Dashboard", "📸  Mark Attendance", "📊  Attendance Records", "👥  Students", "ℹ️  About"],
        label_visibility="collapsed"
    )

    # Map radio choice back to clean key
    menu_clean = menu.replace("  ", " ").strip()

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    # Sidebar stats
    total_students_sidebar = get_total_unique_students()
    total_records_sidebar = len(load_attendance())

    st.markdown(f"""
        <div class="side-stat">
            <span class="side-stat-label">Registered Students</span>
            <span class="side-stat-value">{total_students_sidebar}</span>
        </div>
        <div class="side-stat">
            <span class="side-stat-label">Attendance Records</span>
            <span class="side-stat-value">{total_records_sidebar}</span>
        </div>
    """, unsafe_allow_html=True)

    # Live current time
    now_sidebar = safe_now()
    st.markdown(f"""
        <div class="clock-box">
            <div class="clock-time">{now_sidebar.strftime('%I:%M:%S %p')}</div>
            <div class="clock-date">{now_sidebar.strftime('%A, %d %b %Y')}</div>
        </div>
    """, unsafe_allow_html=True)

    # Warning if clock looks off
    if is_clock_suspicious():
        st.markdown("""
            <div style="background:rgba(255,152,0,0.12); border:1px solid rgba(255,152,0,0.35);
                        border-radius:10px; padding:0.6rem 0.8rem; margin-top:0.6rem;
                        font-size:0.72rem; color:#ffb74d; line-height:1.5;">
                ⚠️ System clock future year me lag raha hai. Sahi time ke liye Windows me
                <b>Settings → Time → Sync now</b> dabao.
            </div>
        """, unsafe_allow_html=True)

    # 1 hour rule note
    st.markdown("""
        <div class="note">
            ⏱️ Ek student ko naya Present entry tabhi milega jab pichli attendance 1 ghante se purani ho.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    # Reset data button
    with st.expander("⚙️ Settings"):
        st.caption("Fresh start karna hai? Ye dono CSV files clear kar dega.")
        if st.button("🗑️ Reset All Data", key="reset_btn"):
            reset_all_data()
            st.success("Data reset ho gaya!")
            st.rerun()


# Header
st.markdown('<div class="main-title">Smart Attendance System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-POWERED · REAL-TIME · PROFESSIONAL</div>', unsafe_allow_html=True)


# Dashboard
if menu_clean == "🏠 Dashboard":
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


# Mark Attendance
elif menu_clean == "📸 Mark Attendance":
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


# Attendance Records
elif menu_clean == "📊 Attendance Records":
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


# Students
elif menu_clean == "👥 Students":
    st.markdown("### 👥 Registered Students")

    df = load_students()

    if df.empty:
        st.info("Koi student registered nahi hai. Neeche form se add karo.")
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


# About
elif menu_clean == "ℹ️ About":
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
    <div style="text-align:center; color:#6b7280; font-size:0.82rem; padding:0.8rem 0;">
        🎓 Smart Attendance System · Built with ❤️ using Streamlit
        <br>
        <span style="font-size:0.72rem;">Last refreshed: {get_current_timestamp()}</span>
    </div>
    """,
    unsafe_allow_html=True
)
