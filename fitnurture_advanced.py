import streamlit as st

# --- Page Config ---
# This MUST be the first Streamlit command in your script.
st.set_page_config(
    page_title="FitNurture : Posture Detection",
    page_icon="🧘‍♀️",
    layout="wide"
)

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
import gc  # Import garbage collector
import pyodbc
import traceback
import json # For Gemini API
import requests # For synchronous HTTP calls to Gemini API

# --- Application Constants ---
LANDMARK_VISIBILITY_THRESHOLD = 0.5
DB_TABLE_NAME = "PostureRecords"
TEXT_OFFSET_X = 70
ARROW_TIP_LENGTH = 0.3
TEXT_FONT_SCALE = 0.5
TEXT_THICKNESS = 1
HIGHLIGHT_COLOR_BGR = (0, 0, 255)
TEXT_BG_COLOR_BGR = (255, 255, 255)

# Original Thresholds
KYPHOSIS_THRESHOLD_SHOULDER_HIP_Z_DIFF = 0.15
LORDOSIS_THRESHOLD_HIP_KNEE_Z_DIFF = 0.1
TECH_NECK_MAX_ESH_ANGLE = 75.0 # Angle should be LESS than this for tech neck (forward head)
TECH_NECK_MIN_ESH_HORIZ_DIST = 0.08 # Ear should be MORE forward than this distance
SCOLIOSIS_THRESHOLD_SHOULDER_Y_DIFF = 0.05
FLAT_FEET_THRESHOLD_FOOT_Z_DIFF = 0.05 # Arch height (z-diff) LESS than this
GAIT_ABNORMALITIES_THRESHOLD_ANKLE_X_DIFF = 0.25 # Ankles WIDER than this
KNOCK_KNEES_KNEE_ANKLE_RATIO_THRESHOLD = 0.1 # Knees significantly closer than ankles (knee_x_diff < ankle_x_diff * ratio)
BOW_LEGS_ANKLE_KNEE_RATIO_THRESHOLD = 1.5 # Ankles significantly closer than knees (ankle_x_diff < knee_x_diff / ratio, or knee_x_diff > ankle_x_diff * ratio)

CLOTHING_ADJUSTMENT_FACTOR = 1.15 # 15% more lenient threshold for loose clothing for Z and Y diffs

VIEWS_SEQUENCE = ['Front View', 'Left Side View', 'Right Side View', 'Back View']
SIDE_VIEWS = ['Left Side View', 'Right Side View']
FRONT_BACK_VIEWS = ['Front View', 'Back View']

AGE_GROUPS = ["6 - 8 years", "9 - 11 years", "12 - 14 years", "15 - 17 years", "Adults (18+)"]
GENDERS = ["Male", "Female", "Prefer not to say"]

POSTURE_RECOMMENDATIONS = {
    "Kyphosis": ["- Practice shoulder blade squeezes", "- Strengthen upper back muscles", "- Maintain proper sitting posture", "- Consider physical therapy exercises"],
    "Lordosis": ["- Core strengthening exercises", "- Hip flexor stretches", "- Pelvic tilt exercises", "- Regular posture checks"],
    "Tech Neck": ["- Adjust device height to eye level", "- Take regular breaks from screens", "- Neck strengthening exercises", "- Practice chin tucks"],
    "Scoliosis": ["- Consult with a spine specialist", "- Core strengthening exercises", "- Swimming or water therapy", "- Regular monitoring"],
    "Flat Feet": ["- Use arch support insoles", "- Foot strengthening exercises", "- Proper footwear selection", "- Consider physical therapy"],
    "Gait Abnormalities": ["- Gait analysis with a specialist", "- Balance exercises", "- Proper footwear", "- Regular walking practice"],
    "Knock Knees": ["- Strengthening exercises for legs", "- Balance training", "- Proper footwear", "- Regular monitoring"],
    "Bow Legs": ["- Consult with an orthopedic specialist", "- Strengthening exercises", "- Balance training", "- Regular monitoring"]
}

# --- Session State Initialization ---
default_session_states = {
    'current_entry': {}, 'landmark_image': None, 'all_landmark_images': {},
    'abnormalities': {}, 'records': [], 'current_student_id': None,
    'analysis_mode': "Single View Analysis", 'input_mode': "Upload Image",
    'capture_stage': 0, 'all_multi_images_captured': False,
    'processing_done': False, 'cloud_upload_status': None,
    'gemini_suggestions': None, 'gemini_suggestions_error': None,
    'selected_age_group': AGE_GROUPS[0], 'selected_gender': GENDERS[0],
    'loose_clothing': False,
    'selected_abnormalities': {k: True for k in POSTURE_RECOMMENDATIONS.keys()},
    'captured_images_multi': {view: None for view in VIEWS_SEQUENCE},
    'camera_input_key_multi': "camera_multi_0",
    'all_multi_images_uploaded': False,
}
for view_name_key_init in [f"uploaded_image_{view.lower().replace(' ', '_')}" for view in VIEWS_SEQUENCE]:
    default_session_states[view_name_key_init] = None

for key, value in default_session_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Memory Management Functions ---
def clear_image_memory():
    """Clears image and processing related states, keeps student info and settings."""
    st.session_state.landmark_image = None
    st.session_state.all_landmark_images = {}
    # st.session_state.current_entry = {} # Keep demographic part of current_entry
    st.session_state.abnormalities = {}
    st.session_state.processing_done = False
    st.session_state.capture_stage = 0
    st.session_state.captured_images_multi = {view: None for view in VIEWS_SEQUENCE}
    st.session_state.all_multi_images_captured = False
    st.session_state.camera_input_key_multi = "camera_multi_0" # Reset key for unique camera inputs
    st.session_state.gemini_suggestions = None
    st.session_state.gemini_suggestions_error = None
    for view_name_key in [f"uploaded_image_{view.lower().replace(' ', '_')}" for view in VIEWS_SEQUENCE]:
        if view_name_key in st.session_state: st.session_state[view_name_key] = None
    st.session_state.all_multi_images_uploaded = False
    if 'current_image_to_process' in st.session_state: del st.session_state.current_image_to_process
    if 'current_images_to_process_multi' in st.session_state: del st.session_state.current_images_to_process_multi
    gc.collect()

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

# --- UI Elements ---
st.markdown("""
<style>
    .stButton>button {
        border: 2px solid #4CAF50; background-color: #4CAF50; color: white;
        padding: 10px 24px; text-align: center; text-decoration: none;
        display: inline-block; font-size: 16px; margin: 4px 2px;
        transition-duration: 0.4s; cursor: pointer; border-radius: 8px;
    }
    .stButton>button:hover { background-color: white; color: black; border: 2px solid #4CAF50; }
    .gemini-button>button { border: 2px solid #2196F3; background-color: #2196F3; }
    .gemini-button>button:hover { background-color: white; color: black; border: 2px solid #2196F3; }
    .copyright-footer { text-align: center; margin-top: 30px; font-size: 0.9em; color: #555; }
    .copyright-footer a { color: #1e88e5; text-decoration: none; }
</style>
""", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; font-size: 24px; margin-bottom: 20px;'>FitNurture : Posture Detection</h2>", unsafe_allow_html=True)
col1_logo, col2_logo, col3_logo = st.columns([1.2, 1, 1.2])
with col2_logo:
    logo_paths = [os.path.join("assets", name) for name in ["logo.jpg", "logo.JPG", "logo.png", "logo.PNG"]]
    logo_found = False
    for logo_path in logo_paths:
        if os.path.exists(logo_path):
            try: st.image(logo_path, width=225, use_container_width=True); logo_found = True; break
            except Exception as e: st.warning(f"Could not load logo {logo_path}: {e}")
    if not logo_found: st.warning("Logo not found in assets directory.")
st.markdown("<br>", unsafe_allow_html=True)

# --- Core Functions ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    if not (a.shape == (2,) and b.shape == (2,) and c.shape == (2,)): return 0.0
    ba, bc = a - b, c - b
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0: return 0.0
    cosine_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def add_landmark_labels(image, landmarks_results):
    img_cv = image.copy(); h, w = img_cv.shape[:2]
    if landmarks_results.pose_landmarks:
        landmarks = landmarks_results.pose_landmarks.landmark
        landmark_labels_map = {mp_pose.PoseLandmark.NOSE: "Head", mp_pose.PoseLandmark.LEFT_SHOULDER: "L Sh", mp_pose.PoseLandmark.RIGHT_SHOULDER: "R Sh", mp_pose.PoseLandmark.LEFT_ELBOW: "L Elb", mp_pose.PoseLandmark.RIGHT_ELBOW: "R Elb", mp_pose.PoseLandmark.LEFT_HIP: "L Hip", mp_pose.PoseLandmark.RIGHT_HIP: "R Hip", mp_pose.PoseLandmark.LEFT_KNEE: "L Knee", mp_pose.PoseLandmark.RIGHT_KNEE: "R Knee", mp_pose.PoseLandmark.LEFT_ANKLE: "L Ankle", mp_pose.PoseLandmark.RIGHT_ANKLE: "R Ankle"}
        for landmark_id, label in landmark_labels_map.items():
            if landmark_id.value < len(landmarks):
                landmark = landmarks[landmark_id.value]
                if landmark.visibility > LANDMARK_VISIBILITY_THRESHOLD:
                    px, py = int(landmark.x * w), int(landmark.y * h)
                    offset_x = -TEXT_OFFSET_X if px < w/2 else TEXT_OFFSET_X
                    cv2.arrowedLine(img_cv, (px + (offset_x//2), py), (px, py), HIGHLIGHT_COLOR_BGR, TEXT_THICKNESS, tipLength=ARROW_TIP_LENGTH)
                    text_anchor = (px + offset_x - (cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, TEXT_FONT_SCALE, TEXT_THICKNESS)[0][0] if offset_x < 0 else 0) , py + 5)
                    cv2.putText(img_cv, label, text_anchor, cv2.FONT_HERSHEY_SIMPLEX, TEXT_FONT_SCALE, HIGHLIGHT_COLOR_BGR, TEXT_THICKNESS, lineType=cv2.LINE_AA)
    return img_cv

@st.cache_resource
def load_pose_model():
    return mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
mp_pose, mp_drawing, pose_static = mp.solutions.pose, mp.solutions.drawing_utils, load_pose_model()

# --- Database Functions ---
def get_db_connection():
    try:
        secrets_dict = {k: st.secrets.get(k) for k in ["DB_DRIVER", "DB_SERVER", "DB_NAME", "DB_UID", "DB_PWD"]}
        missing = [k for k, v in secrets_dict.items() if not v and k != "DB_DRIVER"] # DB_DRIVER can be optional
        if missing: return {"type": "error", "message": f"DB secrets missing: {', '.join(missing)}."}
        db_driver = secrets_dict.get("DB_DRIVER") or "{ODBC Driver 17 for SQL Server}"
        conn_str = f"DRIVER={db_driver};SERVER={secrets_dict['DB_SERVER']};DATABASE={secrets_dict['DB_NAME']};UID={secrets_dict['DB_UID']};PWD={secrets_dict['DB_PWD']};Encrypt=yes;TrustServerCertificate=no;ConnectionTimeout=30;"
        return {"type": "success", "connection": pyodbc.connect(conn_str)}
    except pyodbc.Error as ex: return {"type": "error", "message": f"DB Connection Error: {ex.args[0]}."}
    except Exception as e: return {"type": "error", "message": f"Unexpected DB connection error: {e}"}

def create_table_if_not_exists(conn):
    if conn is None: return False
    cursor = conn.cursor(); table_name = DB_TABLE_NAME
    cols = {
        "Student_ID": "NVARCHAR(50) NOT NULL PRIMARY KEY",
        "Student_Name": "NVARCHAR(255) NULL",
        "Age_Group": "NVARCHAR(50) NULL",      # NEW
        "Gender": "NVARCHAR(20) NULL",         # NEW
        "Loose_Clothing": "BIT NULL",          # NEW
        "Observation_Timestamp": "DATETIME2 NULL",
        "UploadTimestamp": "DATETIME2 DEFAULT GETDATE() NULL"
    }
    for key in POSTURE_RECOMMENDATIONS.keys(): cols[key.replace(' ', '_').replace('-', '_')] = "BIT NULL"
    metrics_keys = ["shoulder_z", "hip_z", "knee_z", "ear_shoulder_hip_angle", "ear_shoulder_horizontal_distance", "shoulder_y_diff", "foot_z_diff", "ankle_x_diff", "knee_x_diff"]
    for key in metrics_keys: cols[key] = "FLOAT NULL"
    defs_list = [f"[{name}] {typedef}" for name, typedef in cols.items()]
    column_definitions_sql_string = ",\n        ".join(defs_list)
    query = f"IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{table_name}' AND xtype='U') CREATE TABLE {table_name} ({column_definitions_sql_string});"
    try: cursor.execute(query); conn.commit(); return True
    except pyodbc.Error as e: st.error(f"Error creating table '{table_name}': {e}"); conn.rollback(); return False
    finally: cursor.close()

def upload_records_to_sql(conn, records_to_upload):
    if not records_to_upload: return {"type": "info", "message": "No new records to upload."}
    if conn is None: return {"type": "error", "message": "Database connection not established for upload."}
    if not create_table_if_not_exists(conn): return {"type": "error", "message": "Failed to create or verify database table."}

    cursor = conn.cursor(); table_name = DB_TABLE_NAME
    base_sql_cols = ["Student_ID", "Student_Name", "Age_Group", "Gender", "Loose_Clothing", "Observation_Timestamp"] # UPDATED
    abnormality_sql_cols = [k.replace(' ', '_').replace('-', '_') for k in POSTURE_RECOMMENDATIONS.keys()]
    metrics_sql_cols = ["shoulder_z", "hip_z", "knee_z", "ear_shoulder_hip_angle", "ear_shoulder_horizontal_distance", "shoulder_y_diff", "foot_z_diff", "ankle_x_diff", "knee_x_diff"]
    all_sql_cols = base_sql_cols + abnormality_sql_cols + metrics_sql_cols
    insert_cols_str = ", ".join([f"[{col}]" for col in all_sql_cols])
    placeholders = ", ".join(["?" for _ in all_sql_cols])
    insert_sql = f"INSERT INTO {table_name} ({insert_cols_str}) VALUES ({placeholders})"
    update_set_clauses = [f"[{col}] = ?" for col in all_sql_cols if col != "Student_ID"]
    update_sql = f"UPDATE {table_name} SET {', '.join(update_set_clauses)}, [UploadTimestamp] = GETDATE() WHERE [Student_ID] = ?"
    insert_count, update_count, error_count = 0, 0, 0; error_messages = []

    for record in records_to_upload:
        values_for_insert = [
            record.get("Student ID"), record.get("Student Name"),
            record.get("Age_Group"), record.get("Gender"),
            bool(record.get("Loose_Clothing", False)), # Ensure boolean for DB
            record.get("Timestamp")
        ] + \
        [bool(record.get(k, False)) for k in POSTURE_RECOMMENDATIONS.keys()] + \
        [float(record.get(k)) if record.get(k) is not None else None for k in metrics_sql_cols]

        student_id_val = record.get("Student ID")
        if not student_id_val:
            error_messages.append(f"Skipping record (missing Student ID): {record.get('Student Name', 'N/A')}")
            error_count +=1; continue
        try:
            cursor.execute(insert_sql, tuple(values_for_insert)); insert_count += 1
        except pyodbc.IntegrityError as e:
            if '2627' in str(e) or 'PRIMARY KEY constraint' in str(e).upper() or 'unique constraint' in str(e).upper():
                try: cursor.execute(update_sql, tuple(values_for_insert[1:] + [student_id_val])); update_count += 1
                except pyodbc.Error as ue: error_messages.append(f"Error updating '{student_id_val}': {ue}"); error_count += 1; conn.rollback() # Rollback on inner error
            else: error_messages.append(f"DB Integrity Error for '{student_id_val}': {e}"); error_count += 1; conn.rollback()
        except pyodbc.Error as e: error_messages.append(f"DB Error for '{student_id_val}': {e}"); error_count += 1; conn.rollback()
        except Exception as ex: error_messages.append(f"Unexpected error for '{student_id_val}': {ex}"); error_count += 1; conn.rollback()
        if error_count > 0: break # Stop processing this batch on first error after rollback

    final_status = {}
    if error_count > 0:
        # Rollback might have already happened in the loop
        final_status = {"type": "error", "message": f"{error_count} record(s) failed. Batch processing stopped. Errors: {'; '.join(error_messages)}"}
    elif insert_count > 0 or update_count > 0:
        try:
            conn.commit()
            msg_parts = []
            if insert_count > 0: msg_parts.append(f"Successfully inserted {insert_count} new record(s).")
            if update_count > 0: msg_parts.append(f"Successfully updated {update_count} existing record(s).")
            final_status = {"type": "success", "message": " ".join(msg_parts)}
        except pyodbc.Error as e:
            try: conn.rollback()
            except pyodbc.Error as rb_err: error_messages.append(f"Rollback after commit failure also failed: {rb_err}")
            final_status = {"type": "error", "message": f"Database commit error: {e}. Records not saved. Errors: {'; '.join(error_messages)}"}
    else:
        final_status = {"type": "info", "message": "No records were processed for upload (or all were skipped)."}
    try: cursor.close()
    except pyodbc.Error: pass
    return final_status

# --- Image Processing and Abnormality Detection ---
def process_image_for_view(image_pil, view_name="Unknown View"):
    if image_pil is None: return None, None, {}
    img_np = np.array(image_pil)
    if img_np.shape[-1] == 4: img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    elif len(img_np.shape) == 2: img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    results = pose_static.process(img_np)
    landmarked_pil_image = image_pil # Default to original if no landmarks
    if not results or not results.pose_landmarks:
        st.warning(f"No person/landmarks detected in {view_name}.")
        return None, landmarked_pil_image, {} # Return original image for display
    lm = results.pose_landmarks.landmark
    img_bgr_for_drawing = cv2.cvtColor(img_np.copy(), cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(img_bgr_for_drawing, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2))
    img_bgr_with_labels = add_landmark_labels(img_bgr_for_drawing, results)
    landmarked_pil_image = Image.fromarray(cv2.cvtColor(img_bgr_with_labels, cv2.COLOR_BGR2RGB))
    metrics = {}
    def is_visible(le): return lm[le.value].visibility > LANDMARK_VISIBILITY_THRESHOLD if le.value < len(lm) else False

    ear_lm_side, shoulder_lm_side, hip_lm_side, knee_lm_side, ankle_lm_side, heel_lm_side, foot_index_lm_side = None, None, None, None, None, None, None
    if view_name == 'Left Side View':
        ear_lm_side, shoulder_lm_side, hip_lm_side, knee_lm_side, ankle_lm_side, heel_lm_side, foot_index_lm_side = mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.LEFT_FOOT_INDEX
    elif view_name == 'Right Side View' or view_name == "Side View (Single)": # Treat single view as right side for consistency
        ear_lm_side, shoulder_lm_side, hip_lm_side, knee_lm_side, ankle_lm_side, heel_lm_side, foot_index_lm_side = mp_pose.PoseLandmark.RIGHT_EAR, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX

    if view_name in SIDE_VIEWS or view_name == "Side View (Single)":
        if ear_lm_side and shoulder_lm_side and hip_lm_side and all(is_visible(p) for p in [ear_lm_side, shoulder_lm_side, hip_lm_side]):
            metrics["ear_shoulder_hip_angle"] = calculate_angle([lm[ear_lm_side.value].x, lm[ear_lm_side.value].y], [lm[shoulder_lm_side.value].x, lm[shoulder_lm_side.value].y], [lm[hip_lm_side.value].x, lm[hip_lm_side.value].y])
        else: metrics["ear_shoulder_hip_angle"] = None
        if ear_lm_side and shoulder_lm_side and all(is_visible(p) for p in [ear_lm_side, shoulder_lm_side]):
            metrics["ear_shoulder_horizontal_distance"] = abs(lm[ear_lm_side.value].x - lm[shoulder_lm_side.value].x)
        else: metrics["ear_shoulder_horizontal_distance"] = None
        metrics["shoulder_z"] = lm[shoulder_lm_side.value].z if shoulder_lm_side and is_visible(shoulder_lm_side) else None
        metrics["hip_z"] = lm[hip_lm_side.value].z if hip_lm_side and is_visible(hip_lm_side) else None
        metrics["knee_z"] = lm[knee_lm_side.value].z if knee_lm_side and is_visible(knee_lm_side) else None
        metrics["foot_z_diff"] = abs(lm[heel_lm_side.value].z - lm[foot_index_lm_side.value].z) if heel_lm_side and foot_index_lm_side and all(is_visible(p) for p in [heel_lm_side, foot_index_lm_side]) else None

    if view_name in FRONT_BACK_VIEWS:
        metrics["shoulder_y_diff"] = abs(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y - lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) if all(is_visible(p) for p in [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]) else None
        metrics["hip_y_diff"] = abs(lm[mp_pose.PoseLandmark.LEFT_HIP.value].y - lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y) if all(is_visible(p) for p in [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]) else None # For future use
        metrics["ankle_x_diff"] = abs(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x - lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x) if all(is_visible(p) for p in [mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE]) else None
        metrics["knee_x_diff"] = abs(lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x - lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x) if all(is_visible(p) for p in [mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE]) else None
    return results.pose_landmarks, landmarked_pil_image, metrics

def apply_clothing_adjustment(base_threshold):
    return base_threshold * (CLOTHING_ADJUSTMENT_FACTOR if st.session_state.loose_clothing else 1.0)

def analyze_multi_view_data(multi_images_pil_dict, selected_abnormalities_config):
    all_metrics_by_view, all_landmarked_images_pil = {}, {}
    consolidated_metrics, final_abnormalities = {}, {k: False for k,v in selected_abnormalities_config.items() if v}
    primary_display_image = None
    for view_name, image_pil in multi_images_pil_dict.items():
        if image_pil:
            _, landmarked_img, view_metrics = process_image_for_view(image_pil, view_name)
            all_metrics_by_view[view_name] = view_metrics
            all_landmarked_images_pil[view_name] = landmarked_img
            if view_name == 'Left Side View' and landmarked_img: primary_display_image = landmarked_img
            elif not primary_display_image and landmarked_img: primary_display_image = landmarked_img # Fallback

    def get_metric(name, views_to_check, default=None, use_average=False):
        valid_metrics = [all_metrics_by_view[v_name][name] for v_name in views_to_check if v_name in all_metrics_by_view and all_metrics_by_view[v_name].get(name) is not None]
        if not valid_metrics: return default
        return np.nanmean(valid_metrics) if use_average else valid_metrics[0] # Prioritize first valid or average

    consolidated_metrics["shoulder_z"] = get_metric("shoulder_z", SIDE_VIEWS, use_average=True)
    consolidated_metrics["hip_z"] = get_metric("hip_z", SIDE_VIEWS, use_average=True)
    consolidated_metrics["knee_z"] = get_metric("knee_z", SIDE_VIEWS, use_average=True)
    consolidated_metrics["ear_shoulder_hip_angle"] = get_metric("ear_shoulder_hip_angle", SIDE_VIEWS, use_average=True, default=90.0) # Default to neutral if not visible
    consolidated_metrics["ear_shoulder_horizontal_distance"] = get_metric("ear_shoulder_horizontal_distance", SIDE_VIEWS, use_average=True, default=0.0)
    consolidated_metrics["foot_z_diff"] = get_metric("foot_z_diff", SIDE_VIEWS, use_average=True)
    consolidated_metrics["shoulder_y_diff"] = get_metric("shoulder_y_diff", ['Back View', 'Front View']) # Prefer Back View if available
    consolidated_metrics["ankle_x_diff"] = get_metric("ankle_x_diff", ['Front View', 'Back View'])
    consolidated_metrics["knee_x_diff"] = get_metric("knee_x_diff", ['Front View', 'Back View'])

    # Apply clothing adjustment to relevant thresholds
    eff_kyphosis_thresh = apply_clothing_adjustment(KYPHOSIS_THRESHOLD_SHOULDER_HIP_Z_DIFF)
    eff_lordosis_thresh = apply_clothing_adjustment(LORDOSIS_THRESHOLD_HIP_KNEE_Z_DIFF)
    eff_scoliosis_thresh = apply_clothing_adjustment(SCOLIOSIS_THRESHOLD_SHOULDER_Y_DIFF)

    if "Kyphosis" in final_abnormalities and all(consolidated_metrics.get(k) is not None for k in ["shoulder_z", "hip_z"]): final_abnormalities["Kyphosis"] = (consolidated_metrics["shoulder_z"] - consolidated_metrics["hip_z"]) > eff_kyphosis_thresh
    if "Lordosis" in final_abnormalities and all(consolidated_metrics.get(k) is not None for k in ["hip_z", "knee_z"]): final_abnormalities["Lordosis"] = (consolidated_metrics["hip_z"] - consolidated_metrics["knee_z"]) > eff_lordosis_thresh
    if "Tech Neck" in final_abnormalities and consolidated_metrics.get("ear_shoulder_hip_angle") is not None and consolidated_metrics.get("ear_shoulder_horizontal_distance") is not None:
        final_abnormalities["Tech Neck"] = (consolidated_metrics["ear_shoulder_hip_angle"] < TECH_NECK_MAX_ESH_ANGLE and consolidated_metrics["ear_shoulder_horizontal_distance"] > TECH_NECK_MIN_ESH_HORIZ_DIST)
    if "Scoliosis" in final_abnormalities and consolidated_metrics.get("shoulder_y_diff") is not None: final_abnormalities["Scoliosis"] = consolidated_metrics["shoulder_y_diff"] > eff_scoliosis_thresh
    if "Flat Feet" in final_abnormalities and consolidated_metrics.get("foot_z_diff") is not None: final_abnormalities["Flat Feet"] = consolidated_metrics["foot_z_diff"] < FLAT_FEET_THRESHOLD_FOOT_Z_DIFF
    if "Gait Abnormalities" in final_abnormalities and consolidated_metrics.get("ankle_x_diff") is not None: final_abnormalities["Gait Abnormalities"] = consolidated_metrics["ankle_x_diff"] > GAIT_ABNORMALITIES_THRESHOLD_ANKLE_X_DIFF
    if "Knock Knees" in final_abnormalities and all(consolidated_metrics.get(k) is not None for k in ["knee_x_diff", "ankle_x_diff"]) and consolidated_metrics.get("ankle_x_diff",0) != 0:
        final_abnormalities["Knock Knees"] = consolidated_metrics["knee_x_diff"] < (consolidated_metrics.get("ankle_x_diff",0) * KNOCK_KNEES_KNEE_ANKLE_RATIO_THRESHOLD)
    if "Bow Legs" in final_abnormalities and all(consolidated_metrics.get(k) is not None for k in ["knee_x_diff", "ankle_x_diff"]) and consolidated_metrics.get("knee_x_diff",0) != 0 : # and consolidated_metrics.get("ankle_x_diff", 0) != 0:
         # knee_x_diff > ankle_x_diff * some_factor, or ankle_x_diff / knee_x_diff < some_ratio
        if consolidated_metrics.get("ankle_x_diff", 0) == 0 and consolidated_metrics.get("knee_x_diff",0) > 0.05 : # Knees apart, ankles together
             final_abnormalities["Bow Legs"] = True
        elif consolidated_metrics.get("knee_x_diff",0) > 0 and consolidated_metrics.get("ankle_x_diff",0) / consolidated_metrics.get("knee_x_diff",1) < (1/BOW_LEGS_ANKLE_KNEE_RATIO_THRESHOLD) : # Original logic was ankle < knee * ratio, so ankle/knee < ratio
             final_abnormalities["Bow Legs"] = True


    if not primary_display_image and all_landmarked_images_pil: primary_display_image = next(iter(all_landmarked_images_pil.values()), None)
    return final_abnormalities, consolidated_metrics, primary_display_image, all_landmarked_images_pil

# --- Gemini API Integration ---
def get_gemini_suggestions(abnormalities_detected_dict, student_name, age_group, gender): # Made synchronous
    st.session_state.gemini_suggestions = None; st.session_state.gemini_suggestions_error = None
    detected_issues = [name for name, present in abnormalities_detected_dict.items() if present]
    if not detected_issues:
        return "No specific postural issues were detected. Focus on maintaining good overall posture and regular physical activity."

    prompt = f"""
    FitNurture has analyzed the posture for {student_name} (Age Group: {age_group}, Gender: {gender}) and identified the following potential postural issues: {', '.join(detected_issues)}.

    Please provide personalized and actionable advice for {student_name} to help improve these conditions, keeping their age group ({age_group}) and gender ({gender}) in mind where appropriate for exercise suggestions or lifestyle tips.
    Your response should be encouraging and easy to understand for a non-medical person (e.g., a parent).
    Structure your response in Markdown format as follows:

    ### ✨ Personalized Posture Plan for {student_name} ✨

    Based on the analysis, here are some suggestions to help improve posture:

    #### Corrective Exercises & Stretches:
    For each of the detected issues, list 2-3 specific exercises or stretches. Be very specific with exercise names.
    For example:
    **For Kyphosis:**
    1.  **Wall Angels:** Brief description of how to perform it.
    2.  **Prone Cobra:** Brief description.

    **For Tech Neck:**
    1.  **Chin Tucks:** Brief description.
    ... and so on for other detected issues.

    #### General Lifestyle Adjustments:
    Provide 3-5 general lifestyle tips that can help with the detected issues or overall posture improvement.
    For example:
    1.  **Screen Time Management:** Explanation.
    2.  **Ergonomic Setup (Study/Work Area):** Explanation.

    #### Important Note:
    Include a brief disclaimer that these suggestions are for informational purposes and a healthcare professional should be consulted for persistent issues or before starting any new exercise program.

    Keep the language positive and supportive.
    """
    try:
        chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
        payload = {"contents": chat_history}
        api_key = "" # Provided by Canvas environment
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        if (result.get('candidates') and result['candidates'][0].get('content') and
                result['candidates'][0]['content'].get('parts') and
                result['candidates'][0]['content']['parts'][0].get('text')):
            suggestions = result['candidates'][0]['content']['parts'][0]['text']
            st.session_state.gemini_suggestions = suggestions; return suggestions
        else:
            st.session_state.gemini_suggestions_error = f"Could not retrieve valid suggestions from the AI. Response structure unexpected. Raw: {str(result)[:500]}"
            st.error(st.session_state.gemini_suggestions_error); return None
    except requests.exceptions.RequestException as e:
        st.session_state.gemini_suggestions_error = f"Network or API error: {e}"
        st.error(st.session_state.gemini_suggestions_error); return None
    except Exception as e:
        st.session_state.gemini_suggestions_error = f"An unexpected error occurred while fetching AI suggestions: {e}"
        st.error(st.session_state.gemini_suggestions_error); return None

# --- Main Application UI ---
child_name = st.text_input("Enter Child's Name (Mandatory):", key="child_name_input", value=st.session_state.get('current_entry',{}).get('Student Name',''))

if child_name: # Only show subsequent inputs if name is present
    st.session_state.selected_age_group = st.selectbox("Select Age Group:", options=AGE_GROUPS, key="age_group_select", index=AGE_GROUPS.index(st.session_state.selected_age_group))
    st.session_state.selected_gender = st.radio("Select Gender:", options=GENDERS, key="gender_radio", index=GENDERS.index(st.session_state.selected_gender), horizontal=True)
    st.session_state.loose_clothing = st.checkbox("Subject is NOT wearing body-fitting clothes (this may affect accuracy for some measurements)", key="loose_clothing_checkbox", value=st.session_state.loose_clothing)
    st.markdown("---")

st.session_state.analysis_mode = st.radio("Select Analysis Mode:", ("Single View Analysis", "Multi-View Analysis (4 Views)"), key="analysis_mode_radio", on_change=clear_image_memory)
st.markdown("### Select Abnormalities to Detect")
select_all_current_value = all(st.session_state.selected_abnormalities.values())
select_all_checkbox_val = st.checkbox("Select All", value=select_all_current_value, key="select_all_checkbox")
if select_all_checkbox_val != select_all_current_value: # Check if the checkbox state changed
    st.session_state.selected_abnormalities = {k: select_all_checkbox_val for k in POSTURE_RECOMMENDATIONS.keys()}
    st.rerun()
st.markdown("---")
cols_abnorm = st.columns(2)
for i, (abn, val) in enumerate(st.session_state.selected_abnormalities.items()):
    with cols_abnorm[i % 2]: st.session_state.selected_abnormalities[abn] = st.checkbox(abn, value=val, key=f"cb_{abn}")
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
            if cd: single_image_data_pil = optimize_image(Image.fromarray(cv2.cvtColor(cv2.imdecode(np.asarray(bytearray(cd.read()),dtype=np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)))
    else: # Multi-View
        st.info("Provide images for all four views.")
        if st.session_state.input_mode == "Upload Image":
            cols_up = st.columns(min(len(VIEWS_SEQUENCE), 2)) # Max 2 columns for uploaders
            all_up_local = True
            for i, vn in enumerate(VIEWS_SEQUENCE):
                with cols_up[i % 2]: # Distribute into 2 columns
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
                cp = st.camera_input(f"Capture {vtc}", key=st.session_state.camera_input_key_multi) # Use dynamic key
                if cp:
                    st.session_state.captured_images_multi[vtc] = optimize_image(Image.fromarray(cv2.cvtColor(cv2.imdecode(np.asarray(bytearray(cp.read()),dtype=np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)))
                    st.session_state.capture_stage += 1
                    st.session_state.camera_input_key_multi = f"camera_multi_{st.session_state.capture_stage}" # Regenerate key
                    st.rerun()
            else: st.session_state.all_multi_images_captured = True; st.success("All 4 views captured!"); multi_images_data_pil = st.session_state.captured_images_multi # Use captured images
            if any(st.session_state.captured_images_multi.values()):
                st.markdown("---"); st.write("Captured Images Preview:"); cols_rev = st.columns(len(VIEWS_SEQUENCE))
                for i, vn_rev in enumerate(VIEWS_SEQUENCE):
                    if st.session_state.captured_images_multi[vn_rev]:
                        with cols_rev[i]: st.image(st.session_state.captured_images_multi[vn_rev],caption=vn_rev,width=150)
                if st.button("Retake All Camera Images",key="retake_multi_cam"): clear_image_memory(); st.rerun()
            st.markdown("---")
else: st.warning("Please enter the child's name and select their age group and gender to proceed.")

# --- Analysis Trigger ---
btn_label = "Analyze All 4 Views" if st.session_state.analysis_mode == "Multi-View Analysis (4 Views)" else "Analyze Posture"
enable_btn = bool(child_name and ( (st.session_state.analysis_mode == "Single View Analysis" and single_image_data_pil) or \
                                (st.session_state.analysis_mode == "Multi-View Analysis (4 Views)" and \
                                 (st.session_state.all_multi_images_uploaded or st.session_state.all_multi_images_captured) and \
                                 all(multi_images_data_pil.get(v) or st.session_state.captured_images_multi.get(v) for v in VIEWS_SEQUENCE) ) ) )

if st.button(btn_label, key="analyze_button", disabled=not enable_btn):
    st.session_state.processing_done = False; st.session_state.gemini_suggestions = None; st.session_state.gemini_suggestions_error = None
    sid = st.session_state.get("current_student_id") or f"FN-{random.randint(1000,9999)}"
    st.session_state.current_student_id = sid
    base_entry_info = {
        "Student ID": sid, "Student Name": child_name,
        "Age_Group": st.session_state.selected_age_group,
        "Gender": st.session_state.selected_gender,
        "Loose_Clothing": st.session_state.loose_clothing,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if st.session_state.analysis_mode == "Single View Analysis":
        if single_image_data_pil:
            _, landmarked_img, metrics = process_image_for_view(single_image_data_pil, "Side View (Single)")
            st.session_state.landmark_image = landmarked_img
            st.session_state.all_landmark_images["Single View"] = landmarked_img
            current_abnormalities = {k: False for k,v in st.session_state.selected_abnormalities.items() if v}

            eff_kyphosis_thresh = apply_clothing_adjustment(KYPHOSIS_THRESHOLD_SHOULDER_HIP_Z_DIFF)
            eff_lordosis_thresh = apply_clothing_adjustment(LORDOSIS_THRESHOLD_HIP_KNEE_Z_DIFF)
            eff_scoliosis_thresh = apply_clothing_adjustment(SCOLIOSIS_THRESHOLD_SHOULDER_Y_DIFF)

            if "Tech Neck" in current_abnormalities and metrics.get("ear_shoulder_hip_angle") is not None and metrics.get("ear_shoulder_horizontal_distance") is not None:
                current_abnormalities["Tech Neck"] = (metrics["ear_shoulder_hip_angle"] < TECH_NECK_MAX_ESH_ANGLE and metrics["ear_shoulder_horizontal_distance"] > TECH_NECK_MIN_ESH_HORIZ_DIST)
            if "Kyphosis" in current_abnormalities and metrics.get("shoulder_z") is not None and metrics.get("hip_z") is not None: current_abnormalities["Kyphosis"] = (metrics["shoulder_z"] - metrics["hip_z"]) > eff_kyphosis_thresh
            if "Lordosis" in current_abnormalities and metrics.get("hip_z") is not None and metrics.get("knee_z") is not None: current_abnormalities["Lordosis"] = (metrics["hip_z"] - metrics["knee_z"]) > eff_lordosis_thresh
            if "Flat Feet" in current_abnormalities and metrics.get("foot_z_diff") is not None: current_abnormalities["Flat Feet"] = metrics["foot_z_diff"] < FLAT_FEET_THRESHOLD_FOOT_Z_DIFF
            # For single view, Scoliosis, Gait, Knock Knees, Bow Legs are less reliable but can be attempted if front/back view is assumed or metrics are available
            if "Scoliosis" in current_abnormalities and metrics.get("shoulder_y_diff") is not None: current_abnormalities["Scoliosis"] = metrics["shoulder_y_diff"] > eff_scoliosis_thresh
            if "Gait Abnormalities" in current_abnormalities and metrics.get("ankle_x_diff") is not None: current_abnormalities["Gait Abnormalities"] = metrics["ankle_x_diff"] > GAIT_ABNORMALITIES_THRESHOLD_ANKLE_X_DIFF
            if "Knock Knees" in current_abnormalities and metrics.get("knee_x_diff") is not None and metrics.get("ankle_x_diff") is not None and metrics.get("ankle_x_diff",0) != 0:
                current_abnormalities["Knock Knees"] = metrics["knee_x_diff"] < (metrics.get("ankle_x_diff",0) * KNOCK_KNEES_KNEE_ANKLE_RATIO_THRESHOLD)
            if "Bow Legs" in current_abnormalities and metrics.get("knee_x_diff") is not None and metrics.get("ankle_x_diff") is not None and metrics.get("knee_x_diff",0) !=0:
                if metrics.get("ankle_x_diff", 0) == 0 and metrics.get("knee_x_diff",0) > 0.05 : current_abnormalities["Bow Legs"] = True
                elif metrics.get("knee_x_diff",0) > 0 and metrics.get("ankle_x_diff",0) / metrics.get("knee_x_diff",1) < (1/BOW_LEGS_ANKLE_KNEE_RATIO_THRESHOLD) : current_abnormalities["Bow Legs"] = True


            st.session_state.abnormalities = current_abnormalities
            entry_metrics = { "shoulder_z": metrics.get("shoulder_z"), "hip_z": metrics.get("hip_z"), "knee_z": metrics.get("knee_z"), "ear_shoulder_hip_angle": metrics.get("ear_shoulder_hip_angle"), "ear_shoulder_horizontal_distance": metrics.get("ear_shoulder_horizontal_distance"), "shoulder_y_diff": metrics.get("shoulder_y_diff"), "foot_z_diff": metrics.get("foot_z_diff"), "ankle_x_diff": metrics.get("ankle_x_diff"), "knee_x_diff": metrics.get("knee_x_diff") }
            st.session_state.current_entry = {**base_entry_info, **current_abnormalities, **entry_metrics}
            for ab_key in POSTURE_RECOMMENDATIONS.keys(): # Ensure all abnormality keys are present
                if ab_key not in st.session_state.current_entry: st.session_state.current_entry[ab_key] = False
            st.session_state.processing_done = True
    elif st.session_state.analysis_mode == "Multi-View Analysis (4 Views)":
        images_for_analysis_final = multi_images_data_pil if st.session_state.all_multi_images_uploaded else st.session_state.captured_images_multi
        if all(images_for_analysis_final.get(view) for view in VIEWS_SEQUENCE):
            final_abnormalities, consolidated_metrics, primary_display_img, all_landmarked_imgs_pil = \
                analyze_multi_view_data(images_for_analysis_final, st.session_state.selected_abnormalities)
            st.session_state.abnormalities = final_abnormalities
            st.session_state.landmark_image = primary_display_img # For potential single display later
            st.session_state.all_landmark_images = all_landmarked_imgs_pil
            st.session_state.current_entry = {**base_entry_info, **final_abnormalities, **consolidated_metrics}
            for ab_key in POSTURE_RECOMMENDATIONS.keys(): # Ensure all abnormality keys are present
                if ab_key not in st.session_state.current_entry: st.session_state.current_entry[ab_key] = False
            st.session_state.processing_done = True
        else: st.error("Not all images for multi-view analysis are available. Please ensure all 4 views are provided.")

# --- Helper function to get abnormality reason string ---
def get_abnormality_reason_string(condition_name, metrics_dict):
    reason = ""
    # Use base thresholds for display, PDF will note if adjustment was made
    if condition_name == "Kyphosis" and metrics_dict.get("shoulder_z") is not None and metrics_dict.get("hip_z") is not None:
        diff = metrics_dict["shoulder_z"] - metrics_dict["hip_z"]
        reason = f"(Sh-Hip Z: {diff:.2f} > {KYPHOSIS_THRESHOLD_SHOULDER_HIP_Z_DIFF})"
    elif condition_name == "Lordosis" and metrics_dict.get("hip_z") is not None and metrics_dict.get("knee_z") is not None:
        diff = metrics_dict["hip_z"] - metrics_dict["knee_z"]
        reason = f"(Hip-Knee Z: {diff:.2f} > {LORDOSIS_THRESHOLD_HIP_KNEE_Z_DIFF})"
    elif condition_name == "Tech Neck" and metrics_dict.get("ear_shoulder_hip_angle") is not None and metrics_dict.get("ear_shoulder_horizontal_distance") is not None:
        angle_val = metrics_dict.get("ear_shoulder_hip_angle", 90)
        dist_val = metrics_dict.get("ear_shoulder_horizontal_distance", 0)
        reason = f"(ESH Angle: {angle_val:.1f}° < {TECH_NECK_MAX_ESH_ANGLE}°, ESH Horiz Dist: {dist_val:.2f} > {TECH_NECK_MIN_ESH_HORIZ_DIST})"
    elif condition_name == "Scoliosis" and metrics_dict.get("shoulder_y_diff") is not None:
        reason = f"(Shoulder Y-diff: {metrics_dict['shoulder_y_diff']:.2f} > {SCOLIOSIS_THRESHOLD_SHOULDER_Y_DIFF})"
    elif condition_name == "Flat Feet" and metrics_dict.get("foot_z_diff") is not None:
        reason = f"(Foot Z-diff: {metrics_dict['foot_z_diff']:.2f} < {FLAT_FEET_THRESHOLD_FOOT_Z_DIFF})"
    elif condition_name == "Gait Abnormalities" and metrics_dict.get("ankle_x_diff") is not None:
        reason = f"(Ankle X-diff: {metrics_dict['ankle_x_diff']:.2f} > {GAIT_ABNORMALITIES_THRESHOLD_ANKLE_X_DIFF})"
    elif condition_name == "Knock Knees" and metrics_dict.get("knee_x_diff") is not None and metrics_dict.get("ankle_x_diff") is not None:
        reason = f"(Knee X-diff: {metrics_dict['knee_x_diff']:.2f}, Ankle X-diff: {metrics_dict['ankle_x_diff']:.2f})" # Ratio logic applied in detection
    elif condition_name == "Bow Legs" and metrics_dict.get("knee_x_diff") is not None and metrics_dict.get("ankle_x_diff") is not None:
        reason = f"(Knee X-diff: {metrics_dict['knee_x_diff']:.2f}, Ankle X-diff: {metrics_dict['ankle_x_diff']:.2f})" # Ratio logic applied in detection
    return reason

# --- Display Results, Save, PDF ---
if st.session_state.processing_done and st.session_state.get("current_entry") and st.session_state.current_entry.get("Student ID"):
    st.success("Analysis Complete!")
    if st.session_state.analysis_mode == "Multi-View Analysis (4 Views)" and st.session_state.all_landmark_images:
        st.write("### Processed Images from All Views:")
        cols_processed_imgs = st.columns(min(len(VIEWS_SEQUENCE), 4))
        for i, view_name in enumerate(VIEWS_SEQUENCE):
            if view_name in st.session_state.all_landmark_images and st.session_state.all_landmark_images[view_name]:
                with cols_processed_imgs[i % 4]:
                    st.image(st.session_state.all_landmark_images[view_name], caption=f"Processed: {view_name}", width=150)
        st.markdown("---")
    elif st.session_state.analysis_mode == "Single View Analysis" and st.session_state.landmark_image:
        st.image(st.session_state.landmark_image, caption="Landmarked Image (Single View)", use_container_width=True)

    display_abnormalities = st.session_state.get("abnormalities", {})
    current_metrics_for_display = st.session_state.current_entry
    if display_abnormalities:
        st.write(f"### Abnormality Detection for {st.session_state.current_entry['Student Name']} (ID: {st.session_state.current_entry['Student ID']}):")
        for cond, pres in display_abnormalities.items():
            reason_str = get_abnormality_reason_string(cond, current_metrics_for_display) if pres else ""
            st.markdown(f"- {cond}: {'**Present**' if pres else 'Not Present'} {reason_str}")
    else: st.info("No abnormalities selected/detected based on the analysis.")

    st.markdown("---")
    st.markdown('<div class="gemini-button">', unsafe_allow_html=True)
    if st.button("✨ Get AI Exercise & Lifestyle Tips", key="gemini_tips_button"):
        with st.spinner("✨ Our AI is crafting personalized tips for you..."):
            _ = get_gemini_suggestions(
                st.session_state.abnormalities,
                st.session_state.current_entry['Student Name'],
                st.session_state.current_entry['Age_Group'],
                st.session_state.current_entry['Gender']
            )
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.gemini_suggestions:
        st.markdown("### ✨ AI-Powered Suggestions"); st.markdown(st.session_state.gemini_suggestions)
    elif st.session_state.gemini_suggestions_error:
        st.error(f"Could not fetch AI suggestions: {st.session_state.gemini_suggestions_error}")
    st.markdown("---")

    col1_actions, col2_actions = st.columns(2)
    with col1_actions:
        if st.button("💾 Save Result Locally", key="save_result_button_main"):
            if st.session_state.gemini_suggestions: st.session_state.current_entry["Gemini_Suggestions"] = st.session_state.gemini_suggestions
            st.session_state.records.append(st.session_state.current_entry.copy()) # Save a copy
            st.success(f"Result for {st.session_state.current_entry['Student ID']} saved locally!")
            # Don't delete current_student_id here, allow multiple saves/PDFs for same student if needed before reset
    with col2_actions:
        if st.button("📄 Generate PDF Report", key="generate_pdf_button_main"):
            try:
                data_pdf = st.session_state.current_entry
                abn_pdf = st.session_state.abnormalities
                gemini_sug_pdf = st.session_state.get("gemini_suggestions")

                pdf = FPDF(); pdf.add_page(); pdf.set_auto_page_break(auto=True, margin=15)
                pdf.set_font("Arial", "B", 16); pdf.cell(0, 10, "FitNurture Posture Analysis Report", ln=1, align="C")
                pdf.set_font("Arial", "", 9); pdf.cell(0, 7, "www.futurenurture.in", ln=1, align="C", link="http://www.futurenurture.in"); pdf.ln(5)
                current_y_logo = pdf.get_y(); logo_pdf_path = next((p for p in logo_paths if os.path.exists(p)), None)
                if logo_pdf_path:
                    logo_width_pdf = 35; logo_height_pdf = 17.5
                    if current_y_logo + logo_height_pdf > pdf.page_break_trigger - 5: pdf.add_page(); current_y_logo = pdf.get_y()
                    pdf.image(logo_pdf_path, x=(210-logo_width_pdf)/2, y=current_y_logo, w=logo_width_pdf); pdf.set_y(current_y_logo + logo_height_pdf + 5)
                pdf.set_font("Arial", "B", 12)
                details_pdf = { # Use a different variable name to avoid conflict
                    "Student Name": data_pdf.get('Student Name'), "Student ID": data_pdf.get('Student ID'),
                    "Age Group": data_pdf.get('Age_Group'), "Gender": data_pdf.get('Gender'),
                    "Timestamp": data_pdf.get('Timestamp')
                }
                for k_pdf, v_pdf in details_pdf.items(): pdf.cell(0, 7, f"{k_pdf}: {v_pdf or 'N/A'}", ln=1)

                pdf.set_font("Arial", "", 10)
                clothing_status_pdf = "Yes" if data_pdf.get('Loose_Clothing') else "No"
                pdf.cell(0, 7, f"Wearing Non-Body-Fitting Clothes: {clothing_status_pdf}", ln=1)
                if data_pdf.get('Loose_Clothing'):
                    pdf.set_font("Arial", "I", 8)
                    pdf.multi_cell(0, 4, "Note: Indication of non-body-fitting clothes might slightly reduce the precision of some skeletal measurements. Thresholds for Kyphosis, Lordosis, and Scoliosis were adjusted accordingly.", 0, 'L', False); pdf.ln(1) # Added align L
                pdf.ln(5)

                if st.session_state.analysis_mode == "Multi-View Analysis (4 Views)" and st.session_state.all_landmark_images:
                    pdf.set_font("Arial", "B", 10); pdf.cell(0, 7, "Processed Views:", ln=1, align='C'); pdf.ln(2)
                    pdf.set_font("Arial", "", 8); img_display_width_pdf = 70; img_spacing_horizontal_pdf = 10
                    row_start_x_pdf = (pdf.w - (img_display_width_pdf * 2 + img_spacing_horizontal_pdf)) / 2
                    image_paths_to_delete_pdf = []
                    try: # Wrap multi-image processing in try-finally for cleanup
                        for i_row in range(0, len(VIEWS_SEQUENCE), 2):
                            current_y_for_row_pdf = pdf.get_y(); max_h_this_row_pdf = 0
                            for j_col in range(2): # Max 2 images per row
                                idx = i_row + j_col
                                if idx < len(VIEWS_SEQUENCE):
                                    view_name_pdf = VIEWS_SEQUENCE[idx]; pil_image_pdf = st.session_state.all_landmark_images.get(view_name_pdf)
                                    if pil_image_pdf:
                                        tmp_file_handle_multi = None
                                        tmp_file_path_multi = None
                                        try:
                                            tmp_file_handle_multi = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                                            tmp_file_path_multi = tmp_file_handle_multi.name
                                            
                                            pil_image_pdf.save(tmp_file_path_multi, format="JPEG")
                                            tmp_file_handle_multi.close() 
                                            tmp_file_handle_multi = None # Mark as closed

                                            image_paths_to_delete_pdf.append(tmp_file_path_multi) 
                                            
                                            o_w, o_h = pil_image_pdf.size; asp = o_h/o_w if o_w > 0 else 1; img_h = img_display_width_pdf * asp
                                            max_h_this_row_pdf = max(max_h_this_row_pdf, img_h)
                                            if current_y_for_row_pdf + img_h + 10 > pdf.page_break_trigger: 
                                                pdf.add_page()
                                                current_y_for_row_pdf = pdf.get_y()
                                            
                                            x_pos_img_pdf = row_start_x_pdf + j_col * (img_display_width_pdf + img_spacing_horizontal_pdf)
                                            pdf.image(tmp_file_path_multi, x=x_pos_img_pdf, y=current_y_for_row_pdf, w=img_display_width_pdf, h=img_h)
                                            pdf.set_xy(x_pos_img_pdf, current_y_for_row_pdf + img_h + 1) 
                                            pdf.multi_cell(img_display_width_pdf, 4, view_name_pdf, 0, 'C')
                                        except Exception as e_multi_img_pdf_item:
                                            st.warning(f"Error processing image {view_name_pdf} for PDF: {e_multi_img_pdf_item}")
                                            if tmp_file_handle_multi is not None and not tmp_file_handle_multi.closed:
                                                tmp_file_handle_multi.close()
                                            # If path was added before error, it will be cleaned up by the outer finally
                            if max_h_this_row_pdf > 0: pdf.set_y(current_y_for_row_pdf + max_h_this_row_pdf + 5 + 5) 
                            else: pdf.ln(5)
                    finally:
                        for path_del in image_paths_to_delete_pdf:
                            if path_del and os.path.exists(path_del):
                                try: os.unlink(path_del)
                                except Exception as e_unlink_multi_loop: st.warning(f"Could not delete temp PDF image {path_del} in loop: {e_unlink_multi_loop}")
                    pdf.ln(5)
                elif st.session_state.analysis_mode == "Single View Analysis" and st.session_state.get("landmark_image"):
                    pil_image_pdf_single = st.session_state.landmark_image
                    page_width_pdf = pdf.w - pdf.l_margin - pdf.r_margin; max_image_height_pdf_val = 80
                    original_w_px_pdf, original_h_px_pdf = pil_image_pdf_single.size; aspect_ratio_pdf = original_h_px_pdf / original_w_px_pdf if original_w_px_pdf > 0 else 1
                    img_w_pdf_val = page_width_pdf * 0.70; img_h_pdf_val = img_w_pdf_val * aspect_ratio_pdf
                    if img_h_pdf_val > max_image_height_pdf_val: img_h_pdf_val = max_image_height_pdf_val; img_w_pdf_val = img_h_pdf_val / aspect_ratio_pdf if aspect_ratio_pdf > 0 else max_image_height_pdf_val
                    
                    current_y_img_pdf = pdf.get_y()
                    if current_y_img_pdf + img_h_pdf_val > pdf.page_break_trigger - 5: 
                        pdf.add_page()
                        current_y_img_pdf = pdf.get_y()

                    tmp_file_handle_single = None
                    tmp_file_path_single_pdf = None # Renamed to avoid conflict
                    try:
                        tmp_file_handle_single = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                        tmp_file_path_single_pdf = tmp_file_handle_single.name
                        pil_image_pdf_single.save(tmp_file_path_single_pdf, format="JPEG")
                        tmp_file_handle_single.close()
                        tmp_file_handle_single = None # Mark as closed

                        img_x_pos_pdf = (pdf.w - img_w_pdf_val) / 2
                        pdf.image(tmp_file_path_single_pdf, x=img_x_pos_pdf, y=current_y_img_pdf, w=img_w_pdf_val, h=img_h_pdf_val)
                        pdf.set_y(current_y_img_pdf + img_h_pdf_val + 5)
                    finally:
                        if tmp_file_handle_single is not None and not tmp_file_handle_single.closed: # Should not happen if logic is correct
                            try: tmp_file_handle_single.close()
                            except: pass 
                        if tmp_file_path_single_pdf and os.path.exists(tmp_file_path_single_pdf):
                            try: os.unlink(tmp_file_path_single_pdf)
                            except Exception as e_unlink_single: st.warning(f"Could not delete temp PDF image {tmp_file_path_single_pdf}: {e_unlink_single}")
                    pdf.ln(3)


                pdf.set_font("Arial", "B", 11); pdf.cell(0, 7, "Detected Postural Issues:", ln=1)
                pdf.set_font("Arial", "", 9); detected_cond_pdf = []
                if abn_pdf:
                    for cond, pres in abn_pdf.items():
                        reason_str_pdf = get_abnormality_reason_string(cond, data_pdf) if pres else ""
                        pdf.cell(0, 5, f"- {cond}: {'Present' if pres else 'Not Present'} {reason_str_pdf}", ln=1)
                        if pres: detected_cond_pdf.append(cond)
                else: pdf.cell(0,5, "- No abnormalities selected for detection or none found.", ln=1)
                pdf.ln(2)
                if detected_cond_pdf:
                    pdf.set_font("Arial", "B", 11); pdf.cell(0, 7, "General Recommendations (Standard):", ln=1)
                    pdf.set_font("Arial", "", 9); available_width_pdf = pdf.w - pdf.l_margin - pdf.r_margin - 10 
                    for cond in detected_cond_pdf:
                        if cond in POSTURE_RECOMMENDATIONS:
                            pdf.set_font("Arial", "B", 9); pdf.multi_cell(available_width_pdf, 5, f"For {cond}:")
                            pdf.set_font("Arial", "", 9)
                            for rec_item in POSTURE_RECOMMENDATIONS[cond]:
                                clean_rec_item = rec_item.strip(); pdf.set_x(pdf.l_margin + 5); pdf.multi_cell(available_width_pdf - 5, 4, clean_rec_item) 
                            pdf.ln(1)
                if gemini_sug_pdf:
                    pdf.ln(3); pdf.set_font("Arial", "B", 11); pdf.cell(0, 7, "✨ AI-Powered Personalized Suggestions:", ln=1)
                    pdf.set_font("Arial", "", 9)
                    lines = gemini_sug_pdf.split('\n')
                    estimated_height = len(lines) * 4
                    if pdf.get_y() + estimated_height > pdf.page_break_trigger: pdf.add_page()
                    for line_pdf in lines: 
                        line_stripped = line_pdf.strip()
                        if line_stripped.startswith("### "): pdf.set_font("Arial", "B", 10); pdf.multi_cell(0, 5, line_stripped.replace("### ", ""), ln=1); pdf.set_font("Arial", "", 9)
                        elif line_stripped.startswith("#### "): pdf.set_font("Arial", "B", 9); pdf.multi_cell(0, 5, line_stripped.replace("#### ", ""), ln=1); pdf.set_font("Arial", "", 9)
                        elif line_stripped.startswith("**For"): pdf.set_font("Arial", "B", 9); pdf.multi_cell(0, 5, line_stripped.replace("**", ""), ln=1); pdf.set_font("Arial", "", 9)
                        elif line_stripped.startswith("* ") or line_stripped.startswith("- ") or (line_stripped and line_stripped[0].isdigit() and line_stripped[1:3] == ". "):
                            pdf.set_x(pdf.l_margin + 5); pdf.multi_cell(0, 4, line_stripped, ln=1); pdf.set_x(pdf.l_margin)
                        else: pdf.multi_cell(0, 4, line_stripped, ln=1)
                    pdf.ln(1)

                pdf.ln(2); disclaimer_height_estimate_pdf = 15
                if pdf.get_y() + disclaimer_height_estimate_pdf > pdf.page_break_trigger -5: pdf.add_page()
                pdf.set_font("Arial", "I", 7)
                disclaimer_text_pdf = "Disclaimer: This automated analysis, including AI-powered suggestions, is for informational purposes only and not a substitute for professional medical advice. Accuracy may be affected by factors like image quality and clothing. Consult a healthcare provider for health concerns or before starting new exercises."
                pdf.multi_cell(0, 3.5, disclaimer_text_pdf, align="C")
                pdf_output_data = pdf.output(dest='S')
                if isinstance(pdf_output_data, str): pdf_bytes_out = pdf_output_data.encode('latin-1')
                elif isinstance(pdf_output_data, (bytearray, bytes)): pdf_bytes_out = bytes(pdf_output_data)
                else: st.error(f"Unexpected PDF output type: {type(pdf_output_data)}"); pdf_bytes_out = b""
                if not pdf_bytes_out: st.error("Critical PDF Error: Output from FPDF is empty.")
                else:
                    st.success("PDF Report Generated!"); st.download_button(label="📥 Download Report PDF",data=pdf_bytes_out,file_name=f"posture_report_{data_pdf.get('Student ID', 'report') if data_pdf else 'report'}.pdf",mime="application/pdf",key="download_full_pdf_button")
            except Exception as e: st.error(f"Error during PDF generation process: {e}\n{traceback.format_exc()}")

# --- View Data Table and Cloud Upload ---
st.markdown("---"); st.subheader("📊 View Locally Saved Records")
if st.session_state.records:
    records_per_page = 10
    if 'current_page_local_records' not in st.session_state: st.session_state.current_page_local_records = 0

    all_keys = set()
    for record in st.session_state.records: all_keys.update(record.keys())
    df_cols_ordered = ["Student ID", "Student Name", "Age_Group", "Gender", "Loose_Clothing", "Timestamp"] + \
                      [k.replace(' ', '_').replace('-', '_') for k in POSTURE_RECOMMENDATIONS.keys()] + \
                      ["shoulder_z", "hip_z", "knee_z", "ear_shoulder_hip_angle", "ear_shoulder_horizontal_distance", "shoulder_y_diff", "foot_z_diff", "ankle_x_diff", "knee_x_diff"] + \
                      ["Gemini_Suggestions"] 
    df_cols_final = [col for col in df_cols_ordered if col in all_keys]
    display_records_df_full = pd.DataFrame(st.session_state.records, columns=df_cols_final)

    search_term = st.text_input("🔍 Search by Student Name or ID in local records", key="search_local")
    if search_term:
        display_records_df_filtered = display_records_df_full[
            display_records_df_full['Student Name'].astype(str).str.contains(search_term, case=False, na=False) |
            display_records_df_full['Student ID'].astype(str).str.contains(search_term, case=False, na=False)
        ]
    else:
        display_records_df_filtered = display_records_df_full

    total_filtered_records = len(display_records_df_filtered)
    total_filtered_pages = (total_filtered_records + records_per_page - 1) // records_per_page if total_filtered_records > 0 else 0

    if total_filtered_pages > 0:
        current_page_display = st.session_state.current_page_local_records + 1 # 1-based for display
        selected_page_display = st.number_input(
            "Select Page", 
            min_value=1, 
            max_value=total_filtered_pages, 
            value=current_page_display,
            key="local_records_page_selector_num_input"
        )
        st.session_state.current_page_local_records = selected_page_display - 1 # Convert back to 0-based
    else: 
         st.session_state.current_page_local_records = 0

    start_idx = st.session_state.current_page_local_records * records_per_page
    end_idx = start_idx + records_per_page
    if not display_records_df_filtered.empty: st.dataframe(display_records_df_filtered.iloc[start_idx:end_idx], use_container_width=True)
    elif search_term: st.info("No local records match your search criteria.")
    else: st.info("No local records to display.")
    
    @st.cache_data
    def convert_all_to_csv(records_list_cache):
        if not records_list_cache: return b""
        all_keys_csv = set()
        for rec_csv in records_list_cache: all_keys_csv.update(rec_csv.keys())
        df_cols_final_csv = [col for col in df_cols_ordered if col in all_keys_csv] # Use same ordered cols
        return pd.DataFrame(records_list_cache, columns=df_cols_final_csv).to_csv(index=False).encode("utf-8")

    if st.session_state.records:
        csv_all = convert_all_to_csv(list(st.session_state.records)) 
        st.download_button("📥 Download All Local Records (CSV)", data=csv_all, file_name="all_posture_records.csv", mime="text/csv", key="download_all_csv")
else: st.info("No records saved locally yet.")

st.markdown("---"); st.subheader("☁️ Cloud Data Storage")
if st.session_state.cloud_upload_status:
    status_type = st.session_state.cloud_upload_status.get("type", "info"); status_message = st.session_state.cloud_upload_status.get("message", "")
    if status_type == "success": st.success(status_message)
    elif status_type == "error": st.error(status_message)
    elif status_type == "warning": st.warning(status_message)
    else: st.info(status_message)

if st.session_state.get('records'):
    if st.button("⬆️ Upload All Saved Records to Cloud", key="upload_to_azure_button"):
        st.session_state.cloud_upload_status = None
        with st.spinner("Connecting to database and uploading records..."):
            conn_result = get_db_connection()
            if conn_result.get("type") == "success":
                conn = conn_result["connection"]
                records_for_sql = []
                for rec_sql in st.session_state.records: 
                    sql_rec_item = {k: v for k, v in rec_sql.items() if k != "Gemini_Suggestions"}
                    records_for_sql.append(sql_rec_item)
                upload_status = upload_records_to_sql(conn, records_for_sql)
                st.session_state.cloud_upload_status = upload_status
                try: conn.close()
                except pyodbc.Error as e_close: 
                    if st.session_state.cloud_upload_status and st.session_state.cloud_upload_status.get("type") == "error":
                        st.session_state.cloud_upload_status["message"] += f" (Also, minor error closing DB connection: {e_close})"
                    else: st.session_state.cloud_upload_status = {"type": "warning", "message": f"Minor error closing DB connection: {e_close}"}
                st.rerun()
            else: st.session_state.cloud_upload_status = conn_result; st.rerun()
else: st.info("No records saved locally to upload to Azure SQL.")

st.markdown("---")
button_col1_manual, button_col2_manual, button_col3_manual = st.columns([2, 1, 2])
with button_col2_manual:
    manual_path = os.path.join("assets", "FitNurture_User_Manual.pdf")
    if os.path.exists(manual_path):
        try:
            with open(manual_path, "rb") as f_manual:
                st.download_button(label="Download User Manual (PDF)", data=f_manual.read(), file_name="FitNurture_User_Manual.pdf", mime="application/pdf", key="download_manual_button")
        except Exception as e_manual: st.warning(f"Could not read user manual: {e_manual}") 
    else: st.warning("User manual PDF not found in assets folder (expected: assets/FitNurture_User_Manual.pdf).")

.st.markdown(f"""<div class="copyright-footer">© Copyright {datetime.now().year} FutureNurture | <a href="http://www.futurenurture.in" target="_blank">www.futurenurture.in</a></div>""", unsafe_allow_html=True)
