import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from fpdf import FPDF
from PIL import Image
import tempfile
import os
import random
import pandas as pd
from datetime import datetime
import gc
import pyodbc
import traceback
import hashlib

# --- Page Config ---
st.set_page_config(
    page_title="FitNurture : Posture Detection",
    page_icon="🧘‍♀️",
    layout="wide"
)

# --- Application Constants ---
DB_TABLE_NAME = "PostureRecords"
USERS_TABLE_NAME = "FitNurtureUsers"
LANDMARK_VISIBILITY_THRESHOLD = 0.5
TEXT_OFFSET_X = 70
ARROW_TIP_LENGTH = 0.3
TEXT_FONT_SCALE = 0.5
TEXT_THICKNESS = 1
HIGHLIGHT_COLOR_BGR = (0, 0, 255)
TEXT_BG_COLOR_BGR = (255, 255, 255)

DEFAULT_THRESHOLDS = {
    "kyphosis": 0.12, "lordosis": 0.08, "tech_neck_angle": 80.0,
    "tech_neck_dist": 0.06, "scoliosis": 0.04, "flat_feet": 0.06,
    "gait": 0.22, "knock_knees": 0.15, "bow_legs": 1.3,
}

CLOTHING_ADJUSTMENT_FACTOR = 1.15
VIEWS_SEQUENCE = ['Front View', 'Left Side View', 'Right Side View', 'Back View']
SIDE_VIEWS = ['Left Side View', 'Right Side View']
FRONT_BACK_VIEWS = ['Front View', 'Back View']
AGE_GROUPS = ["6 - 8 years", "9 - 11 years", "12 - 14 years", "15 - 17 years", "Adults (18+)"]
GENDERS = ["Male", "Female", "Prefer not to say"]

POSTURE_RECOMMENDATIONS = {
    "Kyphosis": ["- Practice shoulder blade squeezes", "- Strengthen upper back muscles"],
    "Lordosis": ["- Core strengthening exercises", "- Hip flexor stretches"],
    "Tech Neck": ["- Adjust device height to eye level", "- Take regular breaks from screens"],
    "Scoliosis": ["- Consult with a spine specialist", "- Core strengthening exercises"],
    "Flat Feet": ["- Use arch support insoles", "- Foot strengthening exercises"],
    "Gait Abnormalities": ["- Gait analysis with a specialist", "- Balance exercises"],
    "Knock Knees": ["- Strengthening exercises for legs", "- Balance training"],
    "Bow Legs": ["- Consult with an orthopedic specialist", "- Strengthening exercises"]
}

# --- Session State Initialization ---
def initialize_session_state():
    default_states = {
        'logged_in': False, 'user_info': None, 'show_password_change': False,
        'school_name': '', 'confirm_next_student': False, 'current_entry': {},
        'landmark_image': None, 'all_landmark_images': {}, 'abnormalities': {},
        'records': [], 'current_student_id': None, 'analysis_mode': "Single View Analysis",
        'input_mode': "Upload Image", 'capture_stage': 0, 'all_multi_images_captured': False,
        'processing_done': False, 'cloud_upload_status': None,
        'selected_age_group': AGE_GROUPS[0], 'selected_gender': GENDERS[0], 'loose_clothing': False,
        'selected_abnormalities': {k: True for k in POSTURE_RECOMMENDATIONS.keys()},
        'captured_images_multi': {view: None for view in VIEWS_SEQUENCE},
        'camera_input_key_multi': "camera_multi_0", 'all_multi_images_uploaded': False,
        'thresholds': DEFAULT_THRESHOLDS.copy(), 'scroll_to_top': False,
    }
    for view_key in [f"uploaded_image_{view.lower().replace(' ', '_')}" for view in VIEWS_SEQUENCE]:
        default_states[view_key] = None

    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()


# --- Password Hashing ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- Database Functions ---
def get_db_connection():
    try:
        secrets_dict = {k: st.secrets.get(k) for k in ["DB_DRIVER", "DB_SERVER", "DB_NAME", "DB_UID", "DB_PWD"]}
        if any(not v for k, v in secrets_dict.items() if k != "DB_DRIVER"):
            return {"type": "error", "message": "Database secrets are missing."}
        db_driver = secrets_dict.get("DB_DRIVER") or "{ODBC Driver 17 for SQL Server}"
        conn_str = f"DRIVER={db_driver};SERVER={secrets_dict['DB_SERVER']};DATABASE={secrets_dict['DB_NAME']};UID={secrets_dict['DB_UID']};PWD={secrets_dict['DB_PWD']};Encrypt=yes;TrustServerCertificate=no;ConnectionTimeout=30;"
        return {"type": "success", "connection": pyodbc.connect(conn_str)}
    except Exception as e:
        return {"type": "error", "message": f"DB connection error: {e}"}

def create_users_table_if_not_exists(conn):
    if conn is None: return False
    try:
        cursor = conn.cursor()
        query = f"""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{USERS_TABLE_NAME}' AND xtype='U')
        CREATE TABLE {USERS_TABLE_NAME} (
            UserID INT IDENTITY(1,1) PRIMARY KEY,
            Email NVARCHAR(255) NOT NULL UNIQUE,
            PasswordHash NVARCHAR(255) NOT NULL,
            Role NVARCHAR(50) NOT NULL,
            FullName NVARCHAR(255),
            IsFirstLogin BIT DEFAULT 1,
            DateCreated DATETIME2 DEFAULT GETDATE(),
            LastLogin DATETIME2 NULL,
            AnalyzePostureClicks INT DEFAULT 0,
            DownloadPdfClicks INT DEFAULT 0
        );
        """
        cursor.execute(query)
        for col, col_type in [("LastLogin", "DATETIME2 NULL"), ("AnalyzePostureClicks", "INT DEFAULT 0"), ("DownloadPdfClicks", "INT DEFAULT 0")]:
            cursor.execute(f"IF COL_LENGTH('{USERS_TABLE_NAME}', '{col}') IS NULL ALTER TABLE {USERS_TABLE_NAME} ADD {col} {col_type};")
        conn.commit()
        cursor.execute(f"SELECT COUNT(*) FROM {USERS_TABLE_NAME} WHERE Role = 'admin'")
        if cursor.fetchone()[0] == 0:
            admin_email = "admin@futurenurture.in"
            admin_pass_hash = hash_password("futurenurture123")
            cursor.execute(
                f"INSERT INTO {USERS_TABLE_NAME} (Email, PasswordHash, Role, FullName, IsFirstLogin) VALUES (?, ?, 'admin', 'Default Admin', 0)",
                (admin_email, admin_pass_hash)
            )
            conn.commit()
        return True
    except pyodbc.Error as e:
        st.error(f"Error with users table: {e}")
        return False
    finally:
        if 'cursor' in locals() and cursor: cursor.close()

def update_user_stats_on_login(conn, user_id):
    if conn is None: return
    try:
        cursor = conn.cursor()
        cursor.execute(f"UPDATE {USERS_TABLE_NAME} SET LastLogin = GETDATE() WHERE UserID = ?", (user_id,))
        conn.commit()
    except pyodbc.Error as e:
        st.warning(f"Could not update last login time: {e}")
    finally:
        if 'cursor' in locals() and cursor: cursor.close()

def increment_user_click_stat(user_id, stat_column):
    conn_details = get_db_connection()
    if conn_details['type'] == 'success':
        conn = conn_details['connection']
        try:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE {USERS_TABLE_NAME} SET {stat_column} = {stat_column} + 1 WHERE UserID = ?", (user_id,))
            conn.commit()
        except pyodbc.Error as e:
            st.warning(f"Could not update click stats: {e}")
        finally:
            if conn: conn.close()

def get_user_by_email(conn, email):
    if conn is None: return None
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT UserID, Email, PasswordHash, Role, FullName, IsFirstLogin FROM {USERS_TABLE_NAME} WHERE Email = ?", (email,))
        row = cursor.fetchone()
        if row:
            return {"UserID": row.UserID, "Email": row.Email, "PasswordHash": row.PasswordHash, "Role": row.Role, "FullName": row.FullName, "IsFirstLogin": row.IsFirstLogin}
        return None
    finally:
        if 'cursor' in locals() and cursor: cursor.close()

def add_user(conn, email, full_name, role='user'):
    if conn is None: return {"success": False, "message": "No DB connection."}
    try:
        cursor = conn.cursor()
        temp_password_hash = hash_password(email)
        cursor.execute(f"INSERT INTO {USERS_TABLE_NAME} (Email, FullName, PasswordHash, Role, IsFirstLogin) VALUES (?, ?, ?, ?, 1)", (email, full_name, temp_password_hash, role))
        conn.commit()
        return {"success": True, "message": f"User {email} created. Temporary password is their email."}
    except pyodbc.IntegrityError:
        return {"success": False, "message": "User with this email already exists."}
    except pyodbc.Error as e:
        return {"success": False, "message": f"Database error: {e}"}
    finally:
        if 'cursor' in locals() and cursor: cursor.close()

def update_user_password(conn, user_id, new_password):
    if conn is None: return False
    try:
        cursor = conn.cursor()
        new_password_hash = hash_password(new_password)
        cursor.execute(f"UPDATE {USERS_TABLE_NAME} SET PasswordHash = ?, IsFirstLogin = 0 WHERE UserID = ?", (new_password_hash, user_id))
        conn.commit()
        return True
    except pyodbc.Error as e:
        st.error(f"Failed to update password: {e}")
        return False
    finally:
        if 'cursor' in locals() and cursor: cursor.close()

def get_all_users(conn):
    if conn is None: return []
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT Email, FullName, Role, DateCreated, LastLogin, AnalyzePostureClicks, DownloadPdfClicks FROM {USERS_TABLE_NAME}")
        return [tuple(row) for row in cursor.fetchall()]
    except pyodbc.Error as e:
        st.error(f"Failed to fetch users: {e}")
        return []
    finally:
        if 'cursor' in locals() and cursor: cursor.close()

# --- Login and User Management UI ---
def login_screen():
    st.markdown("<h2 style='text-align: center;'>FitNurture Login</h2>", unsafe_allow_html=True)
    col1_logo, col2_logo, col3_logo = st.columns([1.2, 1, 1.2])
    with col2_logo:
        logo_paths = [os.path.join("assets", name) for name in ["logo.jpg", "logo.JPG", "logo.png", "logo.PNG"]]
        logo_found = any(os.path.exists(p) for p in logo_paths)
        if logo_found:
            st.image(next(p for p in logo_paths if os.path.exists(p)), width=225, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            email = st.text_input("Email (Username)")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

            if submitted:
                if not email or not password:
                    st.error("Please enter both email and password."); return
                conn_details = get_db_connection()
                if conn_details["type"] == "error":
                    st.error(conn_details["message"]); return
                conn = conn_details["connection"]
                if not create_users_table_if_not_exists(conn):
                     st.error("Failed to initialize user database."); conn.close(); return
                user = get_user_by_email(conn, email)
                if user and user["PasswordHash"] == hash_password(password):
                    update_user_stats_on_login(conn, user["UserID"])
                    st.session_state.logged_in = True
                    st.session_state.user_info = user
                    if user["IsFirstLogin"]:
                        st.session_state.show_password_change = True
                    conn.close()
                    st.rerun()
                else:
                    st.error("Invalid email or password.")
                    conn.close()

def password_change_screen(first_time=False):
    if first_time:
        st.warning("This is your first login. Please set a new password to continue.")
    st.subheader("Change Your Password")
    with st.form("password_change_form"):
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        submitted = st.form_submit_button("Change Password")
        if submitted:
            if not new_password or new_password != confirm_password:
                st.error("Passwords do not match or are empty."); return
            if len(new_password) < 8:
                st.error("Password must be at least 8 characters long."); return
            conn_details = get_db_connection()
            if conn_details["type"] == "success":
                conn = conn_details["connection"]
                if update_user_password(conn, st.session_state.user_info["UserID"], new_password):
                    st.success("Password updated successfully!")
                    st.session_state.show_password_change = False
                    st.session_state.user_info['IsFirstLogin'] = False
                    st.session_state.user_info['PasswordHash'] = hash_password(new_password)
                    conn.close()
                    st.rerun()
                else:
                    st.error("Failed to update password in the database.")
                    conn.close()
            else:
                st.error(conn_details["message"])

def admin_panel():
    st.subheader("Admin Panel")
    st.write("---")
    st.write("**Create New User**")
    with st.form("add_user_form"):
        new_email = st.text_input("New User's Email")
        new_full_name = st.text_input("New User's Full Name")
        submitted = st.form_submit_button("Create User")
        if submitted:
            if not new_email or not new_full_name:
                st.error("Email and Full Name are required.")
            else:
                conn_details = get_db_connection()
                if conn_details["type"] == "success":
                    conn = conn_details["connection"]
                    result = add_user(conn, new_email, new_full_name)
                    if result["success"]: st.success(result["message"])
                    else: st.error(result["message"])
                    conn.close()
                else: st.error(conn_details["message"])

    st.write("---")
    st.write("**Existing Users**")
    conn_details = get_db_connection()
    if conn_details["type"] == "success":
        conn = conn_details["connection"]
        users = get_all_users(conn)
        conn.close()
        if users:
            df = pd.DataFrame(users, columns=["Email", "Full Name", "Role", "Date Created", "Last Login", "Analyze Clicks", "Download Clicks"])
            st.dataframe(df)
        else:
            st.info("No users found.")
    else:
        st.error(conn_details["message"])

# --- Main Application Logic ---
def main_app():
    if st.session_state.scroll_to_top:
        st.components.v1.html("<script>window.parent.document.body.scrollTop = 0; window.parent.document.documentElement.scrollTop = 0;</script>", height=0)
        st.session_state.scroll_to_top = False

    with st.sidebar:
        st.subheader(f"Welcome, {st.session_state.user_info['FullName']}")
        st.write(f"Role: {st.session_state.user_info['Role']}")
        if st.button("Change My Password"):
            st.session_state.show_password_change = not st.session_state.show_password_change
        if st.session_state.user_info['Role'] == 'admin':
            st.write("---"); admin_panel()
        st.write("---")
        if st.button("Logout"):
            for key in list(st.session_state.keys()): del st.session_state[key]
            initialize_session_state(); st.rerun()

    if st.session_state.show_password_change:
        password_change_screen(); return

    def clear_image_memory():
        st.session_state.landmark_image = None; st.session_state.all_landmark_images = {}
        st.session_state.abnormalities = {}; st.session_state.processing_done = False
        st.session_state.capture_stage = 0
        st.session_state.captured_images_multi = {view: None for view in VIEWS_SEQUENCE}
        st.session_state.all_multi_images_captured = False
        st.session_state.camera_input_key_multi = f"camera_multi_{st.session_state.capture_stage}"
        for view_key in [f"uploaded_image_{view.lower().replace(' ', '_')}" for view in VIEWS_SEQUENCE]:
            if view_key in st.session_state: st.session_state[view_key] = None
        st.session_state.all_multi_images_uploaded = False
        gc.collect()

    def reset_for_next_student():
        clear_image_memory()
        st.session_state.current_entry = {}; st.session_state.current_student_id = None
        st.session_state.selected_age_group = AGE_GROUPS[0]
        st.session_state.selected_gender = GENDERS[0]
        st.session_state.loose_clothing = False
        st.session_state.scroll_to_top = True

    def optimize_image(image, max_size=800):
        if isinstance(image, np.ndarray): img = Image.fromarray(image)
        else: img = image
        if not hasattr(img, 'size') or img.size[0] == 0 or img.size[1] == 0: return Image.new('RGB', (100, 100), color='lightgray')
        current_max_dim = max(img.size)
        if current_max_dim > max_size:
            ratio = max_size / current_max_dim
            new_width = max(1, int(img.size[0] * ratio)); new_height = max(1, int(img.size[1] * ratio))
            if new_width > 0 and new_height > 0: img = img.resize((new_width, new_height), Image.LANCZOS)
        return img

    st.markdown("""<style>.stButton>button { border: 2px solid #4CAF50; background-color: #4CAF50; color: white; padding: 10px 24px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; transition-duration: 0.4s; cursor: pointer; border-radius: 8px; } .stButton>button:hover { background-color: white; color: black; border: 2px solid #4CAF50; } .copyright-footer { text-align: center; margin-top: 30px; font-size: 0.9em; color: #555; } .copyright-footer a { color: #1e88e5; text-decoration: none; }</style>""", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font-size: 24px; margin-bottom: 20px;'>FitNurture : Posture Detection</h2>", unsafe_allow_html=True)
    col1_logo, col2_logo, col3_logo = st.columns([1.2, 1, 1.2])
    with col2_logo:
        logo_paths = [os.path.join("assets", name) for name in ["logo.jpg", "logo.JPG", "logo.png", "logo.PNG"]]
        logo_found = any(os.path.exists(p) for p in logo_paths)
        if logo_found:
            st.image(next(p for p in logo_paths if os.path.exists(p)), width=225, use_container_width=True)
        else:
            st.warning("Logo not found in assets directory.")
    st.markdown("<br>", unsafe_allow_html=True)

    def calculate_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        if not (a.shape == (2,) and b.shape == (2,) and c.shape == (2,)): return 0.0
        ba, bc = a - b, c - b
        norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
        if norm_ba == 0 or norm_bc == 0: return 0.0
        cosine_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
        return np.degrees(np.arccos(cosine_angle))

    @st.cache_resource
    def load_pose_model():
        return mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
    mp_pose, mp_drawing, pose_static = mp.solutions.pose, mp.solutions.drawing_utils, load_pose_model()

    def create_posture_records_table_if_not_exists(conn):
        if conn is None: return False
        cursor = conn.cursor()
        cols = {
            "Student_ID": "NVARCHAR(50) NOT NULL PRIMARY KEY", "Student_Name": "NVARCHAR(255) NULL",
            "School_Name": "NVARCHAR(255) NULL", "Age_Group": "NVARCHAR(50) NULL", "Gender": "NVARCHAR(20) NULL",
            "Loose_Clothing": "BIT NULL", "Observation_Timestamp": "DATETIME2 NULL", "AnalyzedByUserID": "NVARCHAR(255) NULL",
            "UploadTimestamp": "DATETIME2 DEFAULT GETDATE() NULL"
        }
        for key in POSTURE_RECOMMENDATIONS.keys(): cols[key.replace(' ', '_')] = "BIT NULL"
        metrics_keys = ["shoulder_z", "hip_z", "knee_z", "ear_shoulder_hip_angle", "ear_shoulder_horizontal_distance", "shoulder_y_diff", "foot_z_diff", "ankle_x_diff", "knee_x_diff"]
        for key in metrics_keys: cols[key] = "FLOAT NULL"
        defs_list = [f"[{name}] {typedef}" for name, typedef in cols.items()]
        query = f"IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{DB_TABLE_NAME}' AND xtype='U') CREATE TABLE {DB_TABLE_NAME} ({', '.join(defs_list)});"
        try:
            cursor.execute(query)
            cursor.execute(f"IF COL_LENGTH('{DB_TABLE_NAME}', 'AnalyzedByUserID') IS NULL ALTER TABLE {DB_TABLE_NAME} ADD AnalyzedByUserID NVARCHAR(255) NULL;")
            conn.commit()
            return True
        except pyodbc.Error as e: st.error(f"Error creating table '{DB_TABLE_NAME}': {e}"); return False
        finally: cursor.close()

    def upload_records_to_sql(conn, records_to_upload):
        if not records_to_upload: return {"type": "info", "message": "No new records to upload."}
        if conn is None: return {"type": "error", "message": "Database connection not established."}
        if not create_posture_records_table_if_not_exists(conn): return {"type": "error", "message": "Failed to verify database table."}

        cursor = conn.cursor()
        base_cols = ["Student_ID", "Student_Name", "School_Name", "Age_Group", "Gender", "Loose_Clothing", "Observation_Timestamp", "AnalyzedByUserID"]
        abnormality_cols = [k.replace(' ', '_') for k in POSTURE_RECOMMENDATIONS.keys()]
        metrics_cols = ["shoulder_z", "hip_z", "knee_z", "ear_shoulder_hip_angle", "ear_shoulder_horizontal_distance", "shoulder_y_diff", "foot_z_diff", "ankle_x_diff", "knee_x_diff"]
        all_sql_cols = base_cols + abnormality_cols + metrics_cols
        insert_sql = f"INSERT INTO {DB_TABLE_NAME} ({', '.join(f'[{c}]' for c in all_sql_cols)}) VALUES ({', '.join(['?'] * len(all_sql_cols))})"
        update_sql = f"UPDATE {DB_TABLE_NAME} SET {', '.join(f'[{c}] = ?' for c in all_sql_cols[1:])}, [UploadTimestamp] = GETDATE() WHERE [Student_ID] = ?"
        
        insert_count, update_count, error_count = 0, 0, 0; error_messages = []
        for record in records_to_upload:
            try:
                values_for_sql = [
                    record.get("Student ID"), record.get("Student Name"), record.get("School Name"),
                    record.get("Age_Group"), record.get("Gender"), bool(record.get("Loose_Clothing")),
                    record.get("Timestamp"), record.get("AnalyzedByUserID")
                ]
                values_for_sql.extend([bool(record.get(k, False)) for k in POSTURE_RECOMMENDATIONS.keys()])
                values_for_sql.extend([record.get(k) for k in metrics_cols])
                student_id = record.get("Student ID")
                if not student_id:
                    error_messages.append(f"Skipping record with no ID: {record.get('Student Name', 'N/A')}"); error_count += 1; continue
                cursor.execute(insert_sql, tuple(values_for_sql))
                insert_count += 1
            except pyodbc.IntegrityError:
                try:
                    update_values = values_for_sql[1:] + [student_id]
                    cursor.execute(update_sql, tuple(update_values)); update_count += 1
                except pyodbc.Error as ue:
                    error_messages.append(f"Update Error for '{student_id}': {ue}"); error_count += 1
            except Exception as e:
                error_messages.append(f"Insert Error for '{student_id}': {e}"); error_count += 1
        
        if error_count > 0:
            conn.rollback(); return {"type": "error", "message": f"{error_count} errors. Batch rolled back. Details: {'; '.join(error_messages)}"}
        else:
            conn.commit(); return {"type": "success", "message": f"Uploaded {insert_count} new and updated {update_count} records."}

    def process_image_for_view(image_pil, view_name="Unknown View"):
        if image_pil is None: return None, None, {}
        img_np = np.array(image_pil.convert('RGB'))
        results = pose_static.process(img_np)
        landmarked_pil_image = image_pil
        if not results or not results.pose_landmarks:
            st.warning(f"No person/landmarks detected in {view_name}."); return None, landmarked_pil_image, {}
        
        lm = results.pose_landmarks.landmark
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(img_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarked_pil_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        
        metrics = {}
        def is_visible(le): return lm[le.value].visibility > LANDMARK_VISIBILITY_THRESHOLD if le.value < len(lm) else False

        ear, shoulder, hip, knee, ankle, heel, foot_idx = (mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.LEFT_FOOT_INDEX) if view_name == 'Left Side View' else (mp_pose.PoseLandmark.RIGHT_EAR, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)

        if view_name in SIDE_VIEWS or view_name == "Side View (Single)":
            if all(is_visible(p) for p in [ear, shoulder, hip]):
                metrics["ear_shoulder_hip_angle"] = calculate_angle([lm[ear.value].x, lm[ear.value].y], [lm[shoulder.value].x, lm[shoulder.value].y], [lm[hip.value].x, lm[hip.value].y])
            if all(is_visible(p) for p in [ear, shoulder]):
                metrics["ear_shoulder_horizontal_distance"] = abs(lm[ear.value].x - lm[shoulder.value].x)
            if is_visible(shoulder): metrics["shoulder_z"] = lm[shoulder.value].z
            if is_visible(hip): metrics["hip_z"] = lm[hip.value].z
            if is_visible(knee): metrics["knee_z"] = lm[knee.value].z
            if all(is_visible(p) for p in [heel, foot_idx]): metrics["foot_z_diff"] = abs(lm[heel.value].z - lm[foot_idx.value].z)

        if view_name in FRONT_BACK_VIEWS:
            if all(is_visible(p) for p in [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]): metrics["shoulder_y_diff"] = abs(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y - lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
            if all(is_visible(p) for p in [mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE]): metrics["ankle_x_diff"] = abs(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x - lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x)
            if all(is_visible(p) for p in [mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE]): metrics["knee_x_diff"] = abs(lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x - lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x)

        return results.pose_landmarks, landmarked_pil_image, metrics

    def apply_clothing_adjustment(base_threshold):
        return base_threshold * (CLOTHING_ADJUSTMENT_FACTOR if st.session_state.loose_clothing else 1.0)

    def analyze_multi_view_data(multi_images_pil_dict, selected_abnormalities_config, thresholds):
        all_metrics_by_view, all_landmarked_images_pil = {}, {}
        for view_name, image_pil in multi_images_pil_dict.items():
            if image_pil:
                _, landmarked_img, view_metrics = process_image_for_view(image_pil, view_name)
                all_metrics_by_view[view_name] = view_metrics; all_landmarked_images_pil[view_name] = landmarked_img

        def get_metric(name, views, default=None, use_avg=False):
            vals = [all_metrics_by_view[v][name] for v in views if v in all_metrics_by_view and all_metrics_by_view[v].get(name) is not None]
            return np.nanmean(vals) if use_avg and vals else (vals[0] if vals else default)

        consolidated_metrics = {
            "shoulder_z": get_metric("shoulder_z", SIDE_VIEWS, use_avg=True), "hip_z": get_metric("hip_z", SIDE_VIEWS, use_avg=True),
            "knee_z": get_metric("knee_z", SIDE_VIEWS, use_avg=True), "ear_shoulder_hip_angle": get_metric("ear_shoulder_hip_angle", SIDE_VIEWS, use_avg=True, default=90.0),
            "ear_shoulder_horizontal_distance": get_metric("ear_shoulder_horizontal_distance", SIDE_VIEWS, use_avg=True, default=0.0),
            "foot_z_diff": get_metric("foot_z_diff", SIDE_VIEWS, use_avg=True), "shoulder_y_diff": get_metric("shoulder_y_diff", FRONT_BACK_VIEWS),
            "ankle_x_diff": get_metric("ankle_x_diff", FRONT_BACK_VIEWS), "knee_x_diff": get_metric("knee_x_diff", FRONT_BACK_VIEWS)
        }
        
        final_abnormalities = {k: False for k,v in selected_abnormalities_config.items() if v}
        if "Kyphosis" in final_abnormalities and all(consolidated_metrics.get(k) is not None for k in ["shoulder_z", "hip_z"]): final_abnormalities["Kyphosis"] = (consolidated_metrics["shoulder_z"] - consolidated_metrics["hip_z"]) > apply_clothing_adjustment(thresholds['kyphosis'])
        if "Lordosis" in final_abnormalities and all(consolidated_metrics.get(k) is not None for k in ["hip_z", "knee_z"]): final_abnormalities["Lordosis"] = (consolidated_metrics["hip_z"] - consolidated_metrics["knee_z"]) > apply_clothing_adjustment(thresholds['lordosis'])
        if "Tech Neck" in final_abnormalities and all(consolidated_metrics.get(k) is not None for k in ["ear_shoulder_hip_angle", "ear_shoulder_horizontal_distance"]): final_abnormalities["Tech Neck"] = (consolidated_metrics["ear_shoulder_hip_angle"] < thresholds['tech_neck_angle'] and consolidated_metrics["ear_shoulder_horizontal_distance"] > thresholds['tech_neck_dist'])
        if "Scoliosis" in final_abnormalities and consolidated_metrics.get("shoulder_y_diff") is not None: final_abnormalities["Scoliosis"] = consolidated_metrics["shoulder_y_diff"] > apply_clothing_adjustment(thresholds['scoliosis'])
        if "Flat Feet" in final_abnormalities and consolidated_metrics.get("foot_z_diff") is not None: final_abnormalities["Flat Feet"] = consolidated_metrics["foot_z_diff"] < thresholds['flat_feet']
        if "Gait Abnormalities" in final_abnormalities and consolidated_metrics.get("ankle_x_diff") is not None: final_abnormalities["Gait Abnormalities"] = consolidated_metrics["ankle_x_diff"] > thresholds['gait']
        if "Knock Knees" in final_abnormalities and all(consolidated_metrics.get(k) is not None for k in ["knee_x_diff", "ankle_x_diff"]) and consolidated_metrics.get("ankle_x_diff",0) > 0: final_abnormalities["Knock Knees"] = consolidated_metrics["knee_x_diff"] < (consolidated_metrics.get("ankle_x_diff",0) * thresholds['knock_knees'])
        if "Bow Legs" in final_abnormalities and all(consolidated_metrics.get(k) is not None for k in ["knee_x_diff", "ankle_x_diff"]) and consolidated_metrics.get("knee_x_diff",0) > 0:
            if consolidated_metrics.get("ankle_x_diff", 0) == 0 and consolidated_metrics.get("knee_x_diff",0) > 0.05: final_abnormalities["Bow Legs"] = True
            elif consolidated_metrics.get("ankle_x_diff",0) > 0 and consolidated_metrics.get("ankle_x_diff",0) / consolidated_metrics.get("knee_x_diff",1) < (1/thresholds['bow_legs']): final_abnormalities["Bow Legs"] = True

        primary_img = all_landmarked_images_pil.get('Left Side View') or next(iter(all_landmarked_images_pil.values()), None)
        return final_abnormalities, consolidated_metrics, primary_img, all_landmarked_images_pil

    st.session_state.school_name = st.text_input("Enter School Name (for this session):", value=st.session_state.get('school_name', ''), key="school_name_input")
    st.markdown("---")
    child_name = st.text_input("Enter Child's Name (Mandatory):", key="child_name_input", value=st.session_state.get('current_entry',{}).get('Student Name',''))
    if child_name:
        st.session_state.selected_age_group = st.selectbox("Select Age Group:", options=AGE_GROUPS, key="age_group_select", index=AGE_GROUPS.index(st.session_state.selected_age_group))
        st.session_state.selected_gender = st.radio("Select Gender:", options=GENDERS, key="gender_radio", index=GENDERS.index(st.session_state.selected_gender), horizontal=True)
        st.session_state.loose_clothing = st.checkbox("Subject is NOT wearing body-fitting clothes", key="loose_clothing_checkbox", value=st.session_state.loose_clothing)
        st.markdown("---")

    st.session_state.analysis_mode = st.radio("Select Analysis Mode:", ("Single View Analysis", "Multi-View Analysis (4 Views)"), key="analysis_mode_radio", on_change=clear_image_memory)
    
    with st.expander("⚙️ Advanced: Adjust Detection Thresholds"):
        st.info("Adjust these values to change the sensitivity of the detection. Lower values are generally less strict.")
        
        def create_threshold_input(label, key, min_val, max_val, step, help_text, format_str="%.2f"):
            st.session_state.thresholds[key] = st.number_input(
                label=label, min_value=float(min_val), max_value=float(max_val),
                value=float(st.session_state.thresholds.get(key, float(min_val))),
                step=float(step), key=f"num_input_{key}", help=help_text, format=format_str
            )

        create_threshold_input("Kyphosis (Slouching)", 'kyphosis', 0.05, 0.4, 0.01, "Increasing this value makes detection stricter, requiring more of a slouch to be flagged.", "%.2f")
        create_threshold_input("Lordosis (Lower Back Curve)", 'lordosis', 0.05, 0.4, 0.01, "Increasing this value makes detection stricter, requiring a more pronounced curve.", "%.2f")
        col_tn1, col_tn2 = st.columns(2)
        with col_tn1:
            create_threshold_input("Tech Neck (Head Tilt Angle)", 'tech_neck_angle', 45.0, 90.0, 0.5, "Decreasing this value makes detection stricter, flagging smaller head tilts.", "%.1f")
        with col_tn2:
            create_threshold_input("Tech Neck (Head Forward Distance)", 'tech_neck_dist', 0.0, 0.2, 0.01, "Increasing this value makes detection stricter, flagging smaller forward head positions.", "%.2f")
        create_threshold_input("Scoliosis (Shoulder Height)", 'scoliosis', 0.01, 0.2, 0.01, "Increasing this value makes detection stricter, flagging smaller differences in shoulder height.", "%.2f")
        create_threshold_input("Flat Feet (Foot Arch)", 'flat_feet', 0.01, 0.15, 0.01, "Decreasing this value makes detection stricter, requiring an even flatter foot arch to be flagged.", "%.2f")
        create_threshold_input("Gait (Foot Stance Width)", 'gait', 0.1, 0.5, 0.01, "Increasing this value makes detection stricter, flagging wider foot stances as abnormal.", "%.2f")
        col_kk, col_bl = st.columns(2)
        with col_kk:
            create_threshold_input("Knock Knees (Knee Proximity)", 'knock_knees', 0.05, 0.95, 0.01, "Decreasing this value makes detection stricter for knees that are close together.", "%.2f")
        with col_bl:
            create_threshold_input("Bow Legs (Knee Separation)", 'bow_legs', 1.1, 3.0, 0.05, "Increasing this value makes detection stricter for knees that are far apart.", "%.2f")
        if st.button("Reset Thresholds to Default", key="reset_thresh_btn"):
            st.session_state.thresholds = DEFAULT_THRESHOLDS.copy(); st.rerun()

    st.markdown("---")
    st.session_state.input_mode = st.radio("Choose Input Method:", ("Upload Image", "Use Camera"), key="input_mode_radio", on_change=clear_image_memory)
    
    single_image_data_pil, multi_images_data_pil = None, {view: None for view in VIEWS_SEQUENCE}
    if child_name:
        if st.session_state.analysis_mode == "Single View Analysis":
            if st.session_state.input_mode == "Upload Image":
                uf = st.file_uploader("Upload (Side View Recommended)", type=["jpg","png","jpeg"], key="_file_uploader_key_single")
                if uf: single_image_data_pil = optimize_image(Image.open(uf))
            else:
                cd = st.camera_input("Take picture (Side View Recommended)", key="camera_data_single")
                if cd: single_image_data_pil = optimize_image(Image.fromarray(cv2.cvtColor(cv2.imdecode(np.frombuffer(cd.getvalue(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)))
        else: # Multi-View
            st.info("Provide images for all four views.")
            if st.session_state.input_mode == "Upload Image":
                cols_up = st.columns(min(len(VIEWS_SEQUENCE), 2))
                all_up_local = True
                for i, vn in enumerate(VIEWS_SEQUENCE):
                    with cols_up[i % 2]:
                        uf_multi = st.file_uploader(f"Upload {vn}", type=["jpg","png","jpeg"], key=f"upload_{vn.lower().replace(' ','_')}")
                        if uf_multi: st.session_state[f"uploaded_image_{vn.lower().replace(' ','_')}"] = optimize_image(Image.open(uf_multi))
                        multi_images_data_pil[vn] = st.session_state.get(f"uploaded_image_{vn.lower().replace(' ','_')}")
                        if not multi_images_data_pil[vn]: all_up_local = False
                st.session_state.all_multi_images_uploaded = all_up_local
                if all_up_local: st.success("All 4 images uploaded and ready for analysis.")
            else: # Camera for Multi-View
                cs = st.session_state.capture_stage
                if cs < len(VIEWS_SEQUENCE):
                    vtc = VIEWS_SEQUENCE[cs]
                    st.markdown(f"**Taking photo for: {vtc}**")
                    if cs > 0: st.info(f"Photo for {VIEWS_SEQUENCE[cs-1]} captured. Turn for {vtc}.")
                    cp = st.camera_input(f"Capture {vtc}", key=st.session_state.camera_input_key_multi)
                    if cp:
                        st.session_state.captured_images_multi[vtc] = optimize_image(Image.fromarray(cv2.cvtColor(cv2.imdecode(np.frombuffer(cp.getvalue(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)))
                        st.session_state.capture_stage += 1
                        st.session_state.camera_input_key_multi = f"camera_multi_{st.session_state.capture_stage}"
                        st.rerun()
                else: st.session_state.all_multi_images_captured = True; st.success("All 4 views captured!"); multi_images_data_pil = st.session_state.captured_images_multi
                if any(st.session_state.captured_images_multi.values()):
                    st.markdown("---"); st.write("Captured Images Preview:"); cols_rev = st.columns(len(VIEWS_SEQUENCE))
                    for i, vn_rev in enumerate(VIEWS_SEQUENCE):
                        if st.session_state.captured_images_multi[vn_rev]:
                            with cols_rev[i]: st.image(st.session_state.captured_images_multi[vn_rev],caption=vn_rev,width=150)
                    if st.button("Retake All Camera Images",key="retake_multi_cam"): clear_image_memory(); st.rerun()
                st.markdown("---")
    else: st.warning("Please enter the child's name and select their age group and gender to proceed.")

    enable_btn = bool(child_name and st.session_state.school_name and ( (st.session_state.analysis_mode == "Single View Analysis" and single_image_data_pil) or \
                                (st.session_state.analysis_mode == "Multi-View Analysis (4 Views)" and \
                                 (st.session_state.all_multi_images_uploaded or st.session_state.all_multi_images_captured) and \
                                 all(multi_images_data_pil.get(v) or st.session_state.captured_images_multi.get(v) for v in VIEWS_SEQUENCE) ) ) )

    if st.button("Analyze Posture", key="analyze_button", disabled=not enable_btn):
        increment_user_click_stat(st.session_state.user_info['UserID'], 'AnalyzePostureClicks')
        st.session_state.processing_done = False
        sid = st.session_state.get("current_student_id") or f"FN-{random.randint(1000,9999)}"
        st.session_state.current_student_id = sid
        base_entry_info = {
            "Student ID": sid, "Student Name": child_name, "School Name": st.session_state.school_name,
            "Age_Group": st.session_state.selected_age_group, "Gender": st.session_state.selected_gender,
            "Loose_Clothing": st.session_state.loose_clothing, "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "AnalyzedByUserID": st.session_state.user_info['Email']
        }
        
        current_thresholds = st.session_state.thresholds

        if st.session_state.analysis_mode == "Single View Analysis":
            if single_image_data_pil:
                _, landmarked_img, metrics = process_image_for_view(single_image_data_pil, "Side View (Single)")
                st.session_state.landmark_image = landmarked_img
                st.session_state.all_landmark_images["Single View"] = landmarked_img
                current_abnormalities = {k: False for k,v in st.session_state.selected_abnormalities.items() if v}
                
                eff_kyphosis = apply_clothing_adjustment(current_thresholds['kyphosis'])
                eff_lordosis = apply_clothing_adjustment(current_thresholds['lordosis'])
                eff_scoliosis = apply_clothing_adjustment(current_thresholds['scoliosis'])

                if "Tech Neck" in current_abnormalities and all(metrics.get(k) is not None for k in ["ear_shoulder_hip_angle", "ear_shoulder_horizontal_distance"]):
                    current_abnormalities["Tech Neck"] = (metrics["ear_shoulder_hip_angle"] < current_thresholds['tech_neck_angle'] and metrics["ear_shoulder_horizontal_distance"] > current_thresholds['tech_neck_dist'])
                if "Kyphosis" in current_abnormalities and all(metrics.get(k) is not None for k in ["shoulder_z", "hip_z"]): current_abnormalities["Kyphosis"] = (metrics["shoulder_z"] - metrics["hip_z"]) > eff_kyphosis
                if "Lordosis" in current_abnormalities and all(metrics.get(k) is not None for k in ["hip_z", "knee_z"]): current_abnormalities["Lordosis"] = (metrics["hip_z"] - metrics["knee_z"]) > eff_lordosis
                if "Flat Feet" in current_abnormalities and metrics.get("foot_z_diff") is not None: current_abnormalities["Flat Feet"] = metrics["foot_z_diff"] < current_thresholds['flat_feet']
                if "Scoliosis" in current_abnormalities and metrics.get("shoulder_y_diff") is not None: current_abnormalities["Scoliosis"] = metrics["shoulder_y_diff"] > eff_scoliosis
                if "Gait Abnormalities" in current_abnormalities and metrics.get("ankle_x_diff") is not None: current_abnormalities["Gait Abnormalities"] = metrics["ankle_x_diff"] > current_thresholds['gait']
                if "Knock Knees" in current_abnormalities and all(metrics.get(k) is not None for k in ["knee_x_diff", "ankle_x_diff"]) and metrics.get("ankle_x_diff",0) > 0:
                    current_abnormalities["Knock Knees"] = metrics["knee_x_diff"] < (metrics.get("ankle_x_diff",0) * current_thresholds['knock_knees'])
                if "Bow Legs" in current_abnormalities and all(metrics.get(k) is not None for k in ["knee_x_diff", "ankle_x_diff"]) and metrics.get("knee_x_diff",0) > 0:
                    if metrics.get("ankle_x_diff", 0) == 0 and metrics.get("knee_x_diff",0) > 0.05: current_abnormalities["Bow Legs"] = True
                    elif metrics.get("ankle_x_diff",0) > 0 and metrics.get("ankle_x_diff",0) / metrics.get("knee_x_diff",1) < (1/current_thresholds['bow_legs']): current_abnormalities["Bow Legs"] = True

                st.session_state.abnormalities = current_abnormalities
                st.session_state.current_entry = {**base_entry_info, **current_abnormalities, **metrics}
                st.session_state.processing_done = True
        else: # Multi-view
            images_for_analysis = multi_images_data_pil if st.session_state.all_multi_images_uploaded else st.session_state.captured_images_multi
            if all(images_for_analysis.get(v) for v in VIEWS_SEQUENCE):
                final_abns, cons_metrics, p_img, all_l_imgs = analyze_multi_view_data(images_for_analysis, st.session_state.selected_abnormalities, current_thresholds)
                st.session_state.abnormalities = final_abns
                st.session_state.landmark_image = p_img
                st.session_state.all_landmark_images = all_l_imgs
                st.session_state.current_entry = {**base_entry_info, **final_abns, **cons_metrics}
                st.session_state.processing_done = True
            else: st.error("Not all images for multi-view analysis are available.")

    def get_abnormality_reason_string(condition_name, metrics_dict, thresholds):
        reason = ""
        try:
            if condition_name == "Kyphosis" and all(k in metrics_dict for k in ["shoulder_z", "hip_z"]):
                reason = f"(Sh-Hip Z: {metrics_dict['shoulder_z'] - metrics_dict['hip_z']:.2f} > {thresholds['kyphosis']})"
            elif condition_name == "Lordosis" and all(k in metrics_dict for k in ["hip_z", "knee_z"]):
                reason = f"(Hip-Knee Z: {metrics_dict['hip_z'] - metrics_dict['knee_z']:.2f} > {thresholds['lordosis']})"
            elif condition_name == "Tech Neck" and all(k in metrics_dict for k in ["ear_shoulder_hip_angle", "ear_shoulder_horizontal_distance"]):
                reason = f"(Angle: {metrics_dict['ear_shoulder_hip_angle']:.1f}° < {thresholds['tech_neck_angle']}°, Dist: {metrics_dict['ear_shoulder_horizontal_distance']:.2f} > {thresholds['tech_neck_dist']})"
            elif condition_name == "Scoliosis" and "shoulder_y_diff" in metrics_dict:
                reason = f"(Shoulder Y-diff: {metrics_dict['shoulder_y_diff']:.2f} > {thresholds['scoliosis']})"
            elif condition_name == "Flat Feet" and "foot_z_diff" in metrics_dict:
                reason = f"(Foot Arch: {metrics_dict['foot_z_diff']:.2f} < {thresholds['flat_feet']})"
            elif condition_name == "Gait Abnormalities" and "ankle_x_diff" in metrics_dict:
                reason = f"(Ankle X-diff: {metrics_dict['ankle_x_diff']:.2f} > {thresholds['gait']})"
            elif condition_name in ["Knock Knees", "Bow Legs"] and all(k in metrics_dict for k in ["knee_x_diff", "ankle_x_diff"]):
                reason = f"(Knee X: {metrics_dict['knee_x_diff']:.2f}, Ankle X: {metrics_dict['ankle_x_diff']:.2f})"
        except (KeyError, TypeError): reason = "(metric data missing)"
        return reason

    if st.session_state.processing_done and st.session_state.get("current_entry"):
        st.success("Analysis Complete!")
        if st.session_state.analysis_mode == "Multi-View Analysis (4 Views)" and st.session_state.all_landmark_images:
            st.write("### Processed Images from All Views:")
            cols_proc_imgs = st.columns(min(len(VIEWS_SEQUENCE), 4))
            for i, vn in enumerate(VIEWS_SEQUENCE):
                if vn in st.session_state.all_landmark_images and st.session_state.all_landmark_images[vn]:
                    with cols_proc_imgs[i % 4]: st.image(st.session_state.all_landmark_images[vn], caption=f"Processed: {vn}", width=150)
            st.markdown("---")
        elif st.session_state.landmark_image: st.image(st.session_state.landmark_image, caption="Landmarked Image", use_container_width=True)

        current_abns_display = st.session_state.get("abnormalities", {})
        if current_abns_display:
            st.write(f"### Abnormality Detection for {st.session_state.current_entry['Student Name']}:")
            for cond, pres in current_abns_display.items():
                reason = get_abnormality_reason_string(cond, st.session_state.current_entry, st.session_state.thresholds) if pres else ""
                st.markdown(f"- {cond}: {'**Present**' if pres else 'Not Present'} {reason}")
        else: st.info("No abnormalities detected.")
        st.markdown("---")
        
        pdf_bytes_out = b""
        try:
            data_pdf = st.session_state.current_entry
            abn_pdf = st.session_state.abnormalities
            pdf = FPDF(); pdf.add_page(); pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", "B", 16); pdf.cell(0, 10, "FitNurture Posture Analysis Report", ln=1, align="C")
            pdf.set_font("Arial", "", 9); pdf.cell(0, 7, "www.futurenurture.in", ln=1, align="C", link="http://www.futurenurture.in"); pdf.ln(5)
            
            details_pdf = { "School Name": data_pdf.get('School Name'), "Student Name": data_pdf.get('Student Name'), "Student ID": data_pdf.get('Student ID'), "Age Group": data_pdf.get('Age_Group'), "Gender": data_pdf.get('Gender'), "Timestamp": data_pdf.get('Timestamp')}
            for k_pdf, v_pdf in details_pdf.items(): pdf.cell(0, 7, f"{k_pdf}: {v_pdf or 'N/A'}", ln=1)
            
            pdf.set_font("Arial", "B", 11); pdf.cell(0, 7, "Detected Postural Issues:", ln=1)
            pdf.set_font("Arial", "", 9); detected_cond_pdf = []
            if abn_pdf:
                for cond, pres in abn_pdf.items():
                    reason_str_pdf = get_abnormality_reason_string(cond, data_pdf, st.session_state.thresholds) if pres else ""
                    pdf.cell(0, 5, f"- {cond}: {'Present' if pres else 'Not Present'} {reason_str_pdf}", ln=1)
                    if pres: detected_cond_pdf.append(cond)
            else: pdf.cell(0,5, "- No abnormalities selected for detection or none found.", ln=1)
            
            pdf_output_data = pdf.output(dest='S')
            if isinstance(pdf_output_data, str): pdf_bytes_out = pdf_output_data.encode('latin-1')
            elif isinstance(pdf_output_data, (bytearray, bytes)): pdf_bytes_out = bytes(pdf_output_data)

        except Exception as e:
            st.error(f"Error during PDF generation: {e}")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("💾 Save Result Locally", key="save_result_button"):
                st.session_state.records.append(st.session_state.current_entry.copy())
                st.success(f"Result for {st.session_state.current_entry['Student ID']} saved locally!")
        with col2:
            if st.button("➡️ Next Student", key="next_student_button"):
                reset_for_next_student(); st.rerun()
        with col3:
            if pdf_bytes_out:
                st.download_button(
                    label="📥 Download Report PDF",
                    data=pdf_bytes_out,
                    file_name=f"posture_report_{st.session_state.current_entry.get('Student ID', 'report')}.pdf",
                    mime="application/pdf",
                    key="download_pdf_button",
                    on_click=increment_user_click_stat,
                    args=(st.session_state.user_info['UserID'], 'DownloadPdfClicks')
                )

    st.markdown("---"); st.subheader("📊 View Locally Saved Records")
    if st.session_state.records:
        display_records = []
        for record in st.session_state.records:
            display_record = record.copy()
            display_records.append(display_record)
        
        cols_to_display = ["School Name", "Student ID", "Student Name", "Age_Group", "Gender", "Timestamp"]
        abnormality_cols = []
        if display_records:
            abnormality_cols = [k for k in POSTURE_RECOMMENDATIONS.keys() if k in display_records[0]]
        final_cols = cols_to_display + abnormality_cols
        
        df = pd.DataFrame(display_records)
        final_cols_exist = [col for col in final_cols if col in df.columns]
        
        st.dataframe(df[final_cols_exist])

    st.markdown("---"); st.subheader("☁️ Cloud Data Storage")
    if st.session_state.get('records'):
        if st.button("⬆️ Upload All Saved Records to Cloud", key="upload_to_azure_button"):
            with st.spinner("Connecting to the cloud..."):
                conn_details = get_db_connection()
            if conn_details["type"] == "success":
                st.success("Successfully connected to the database.")
                conn = conn_details["connection"]
                with st.spinner("Uploading records..."):
                    upload_status = upload_records_to_sql(conn, st.session_state.records)
                if upload_status["type"] == "success":
                    st.success(upload_status["message"])
                    st.session_state.records = [] # Clear local records after successful upload
                else:
                    st.error(upload_status["message"])
                try: conn.close()
                except pyodbc.Error: pass
            else:
                st.error(conn_details["message"])

    st.markdown(f"""<div class="copyright-footer">© Copyright {datetime.now().year} FutureNurture | <a href="http://www.futurenurture.in" target="_blank">www.futurenurture.in</a></div>""", unsafe_allow_html=True)

# --- Main Execution Logic ---
if not st.session_state.logged_in:
    login_screen()
elif st.session_state.get('show_password_change', False):
    password_change_screen(first_time=True)
else:
    main_app()
