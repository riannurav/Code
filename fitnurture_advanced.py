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

# --- Application Constants ---
LANDMARK_VISIBILITY_THRESHOLD = 0.5
DB_TABLE_NAME = "PostureRecords"
TEXT_OFFSET_X = 70
ARROW_TIP_LENGTH = 0.3
TEXT_FONT_SCALE = 0.5
TEXT_THICKNESS = 1
HIGHLIGHT_COLOR_BGR = (0, 0, 255) 
TEXT_BG_COLOR_BGR = (255, 255, 255)
KYPHOSIS_THRESHOLD_SHOULDER_HIP_Z_DIFF = 0.15
LORDOSIS_THRESHOLD_HIP_KNEE_Z_DIFF = 0.1
# Revised Tech Neck Thresholds
TECH_NECK_MAX_ESH_ANGLE = 75.0  # Angle at SHOULDER (EAR-SHOULDER-HIP) should be LESS than this
TECH_NECK_MIN_ESH_HORIZ_DIST = 0.08 # Horizontal distance between EAR and SHOULDER should be GREATER
SCOLIOSIS_THRESHOLD_SHOULDER_Y_DIFF = 0.05 
FLAT_FEET_THRESHOLD_FOOT_Z_DIFF = 0.05
GAIT_ABNORMALITIES_THRESHOLD_ANKLE_X_DIFF = 0.25 
KNOCK_KNEES_KNEE_ANKLE_RATIO_THRESHOLD = 0.1 
BOW_LEGS_ANKLE_KNEE_RATIO_THRESHOLD = 1.5 


VIEWS_SEQUENCE = ['Front View', 'Left Side View', 'Right Side View', 'Back View']
SIDE_VIEWS = ['Left Side View', 'Right Side View']
FRONT_BACK_VIEWS = ['Front View', 'Back View']

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
if 'current_entry' not in st.session_state: st.session_state.current_entry = {}
if 'landmark_image' not in st.session_state: st.session_state.landmark_image = None 
if 'all_landmark_images' not in st.session_state: st.session_state.all_landmark_images = {} 
if 'abnormalities' not in st.session_state: st.session_state.abnormalities = {}
if 'records' not in st.session_state: st.session_state.records = []
if 'current_student_id' not in st.session_state: st.session_state.current_student_id = None
if 'analysis_mode' not in st.session_state: st.session_state.analysis_mode = "Single View Analysis"
if 'input_mode' not in st.session_state: st.session_state.input_mode = "Upload Image" 
if 'capture_stage' not in st.session_state: st.session_state.capture_stage = 0 
if 'captured_images_multi' not in st.session_state: st.session_state.captured_images_multi = {view: None for view in VIEWS_SEQUENCE}
if 'camera_input_key_multi' not in st.session_state: st.session_state.camera_input_key_multi = "camera_multi_0" 
if 'all_multi_images_captured' not in st.session_state: st.session_state.all_multi_images_captured = False
for view_name_key in [f"uploaded_image_{view.lower().replace(' ', '_')}" for view in VIEWS_SEQUENCE]:
    if view_name_key not in st.session_state: st.session_state[view_name_key] = None
if 'all_multi_images_uploaded' not in st.session_state: st.session_state.all_multi_images_uploaded = False
if 'processing_done' not in st.session_state: st.session_state.processing_done = False
if 'selected_abnormalities' not in st.session_state: st.session_state.selected_abnormalities = {k: True for k in POSTURE_RECOMMENDATIONS.keys()}
if 'cloud_upload_status' not in st.session_state: st.session_state.cloud_upload_status = None


# --- Memory Management Functions ---
def clear_image_memory():
    st.session_state.landmark_image = None
    st.session_state.all_landmark_images = {}
    st.session_state.current_entry = {}
    st.session_state.abnormalities = {}
    st.session_state.processing_done = False 
    st.session_state.capture_stage = 0
    st.session_state.captured_images_multi = {view: None for view in VIEWS_SEQUENCE}
    st.session_state.all_multi_images_captured = False
    st.session_state.camera_input_key_multi = "camera_multi_0" 
    for view_name_key in [f"uploaded_image_{view.lower().replace(' ', '_')}" for view in VIEWS_SEQUENCE]:
        st.session_state[view_name_key] = None
    st.session_state.all_multi_images_uploaded = False
    # Do not assign to widget keys directly. Streamlit manages their state.
    # If these keys exist, their widgets will re-render and manage their own values.
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
st.markdown("""<style>/* ... CSS ... */</style>""", unsafe_allow_html=True) 
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
        missing = [k for k, v in secrets_dict.items() if not v and k != "DB_DRIVER"]
        if missing: return {"type": "error", "message": f"DB secrets missing: {', '.join(missing)}."}
        db_driver = secrets_dict.get("DB_DRIVER") or "{ODBC Driver 17 for SQL Server}"
        conn_str = f"DRIVER={db_driver};SERVER={secrets_dict['DB_SERVER']};DATABASE={secrets_dict['DB_NAME']};UID={secrets_dict['DB_UID']};PWD={secrets_dict['DB_PWD']};Encrypt=yes;TrustServerCertificate=no;ConnectionTimeout=30;"
        return {"type": "success", "connection": pyodbc.connect(conn_str)}
    except pyodbc.Error as ex: return {"type": "error", "message": f"DB Connection Error: {ex.args[0]}."}
    except Exception as e: return {"type": "error", "message": f"Unexpected DB connection error: {e}"}


def create_table_if_not_exists(conn): 
    if conn is None: return False 
    cursor = conn.cursor(); table_name = DB_TABLE_NAME
    cols = {"Student_ID": "NVARCHAR(50) NOT NULL PRIMARY KEY", "Student_Name": "NVARCHAR(255) NULL", "Observation_Timestamp": "DATETIME2 NULL", "UploadTimestamp": "DATETIME2 DEFAULT GETDATE() NULL"}
    for key in POSTURE_RECOMMENDATIONS.keys(): cols[key.replace(' ', '_').replace('-', '_')] = "BIT NULL"
    metrics_keys = ["shoulder_z", "hip_z", "knee_z", "ear_shoulder_hip_angle", "ear_shoulder_horizontal_distance", "shoulder_y_diff", "foot_z_diff", "ankle_x_diff", "knee_x_diff"] # Updated metric names
    for key in metrics_keys: cols[key] = "FLOAT NULL"
    defs_list = [f"[{name}] {typedef}" for name, typedef in cols.items()]
    column_definitions_sql_string = ",\n        ".join(defs_list) 
    query = f"IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{table_name}' AND xtype='U') CREATE TABLE {table_name} ({column_definitions_sql_string});"
    try: cursor.execute(query); conn.commit(); return True
    except pyodbc.Error as e: st.error(f"Error creating table '{table_name}': {e}"); conn.rollback(); return False
    finally: cursor.close()

def upload_records_to_sql(conn, records_to_upload): 
    if not records_to_upload:
        return {"type": "info", "message": "No new records to upload."}
    if conn is None: 
        return {"type": "error", "message": "Database connection not established for upload."}
    
    if not create_table_if_not_exists(conn): 
        return {"type": "error", "message": "Failed to create or verify database table."}

    cursor = conn.cursor()
    table_name = DB_TABLE_NAME
    base_sql_cols = ["Student_ID", "Student_Name", "Observation_Timestamp"]
    abnormality_sql_cols = [k.replace(' ', '_').replace('-', '_') for k in POSTURE_RECOMMENDATIONS.keys()]
    metrics_sql_cols = ["shoulder_z", "hip_z", "knee_z", "ear_shoulder_hip_angle", "ear_shoulder_horizontal_distance", "shoulder_y_diff", "foot_z_diff", "ankle_x_diff", "knee_x_diff"] # Updated metric names
    all_sql_cols = base_sql_cols + abnormality_sql_cols + metrics_sql_cols
    insert_cols_str = ", ".join([f"[{col}]" for col in all_sql_cols])
    placeholders = ", ".join(["?" for _ in all_sql_cols])
    insert_sql = f"INSERT INTO {table_name} ({insert_cols_str}) VALUES ({placeholders})"
    update_set_clauses = [f"[{col}] = ?" for col in all_sql_cols if col != "Student_ID"]
    update_sql = f"UPDATE {table_name} SET {', '.join(update_set_clauses)}, [UploadTimestamp] = GETDATE() WHERE [Student_ID] = ?"
    insert_count, update_count, error_count = 0, 0, 0
    error_messages = []

    for record in records_to_upload:
        values_for_insert = [record.get("Student ID"), record.get("Student Name"), record.get("Timestamp")] + \
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
                except pyodbc.Error as ue: error_messages.append(f"Error updating '{student_id_val}': {ue}"); error_count += 1
            else: error_messages.append(f"DB Integrity Error for '{student_id_val}': {e}"); error_count += 1
        except pyodbc.Error as e: error_messages.append(f"DB Error for '{student_id_val}': {e}"); error_count += 1
        except Exception as ex: error_messages.append(f"Unexpected error for '{student_id_val}': {ex}"); error_count += 1
    
    final_status = {}
    if error_count > 0:
        try: conn.rollback()
        except pyodbc.Error as rb_err: error_messages.append(f"Rollback attempt also failed: {rb_err}")
        final_status = {"type": "error", "message": f"{error_count} record(s) failed. Batch rolled back. Errors: {'; '.join(error_messages)}"}
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
        final_status = {"type": "info", "message": "No records were processed for upload."}
    
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
    landmarked_pil_image = image_pil 
    if not results or not results.pose_landmarks:
        st.warning(f"No person/landmarks detected in {view_name}.")
        return None, landmarked_pil_image, {}
    lm = results.pose_landmarks.landmark
    img_bgr_for_drawing = cv2.cvtColor(img_np.copy(), cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(img_bgr_for_drawing, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2))
    img_bgr_with_labels = add_landmark_labels(img_bgr_for_drawing, results)
    landmarked_pil_image = Image.fromarray(cv2.cvtColor(img_bgr_with_labels, cv2.COLOR_BGR2RGB))
    metrics = {}
    def is_visible(le): return lm[le.value].visibility > LANDMARK_VISIBILITY_THRESHOLD if le.value < len(lm) else False
    
    # Determine which set of landmarks to use for side views
    ear_lm_side, shoulder_lm_side, hip_lm_side, knee_lm_side, ankle_lm_side, heel_lm_side, foot_index_lm_side = None, None, None, None, None, None, None
    if view_name == 'Left Side View':
        ear_lm_side = mp_pose.PoseLandmark.LEFT_EAR
        shoulder_lm_side = mp_pose.PoseLandmark.LEFT_SHOULDER
        hip_lm_side = mp_pose.PoseLandmark.LEFT_HIP
        knee_lm_side = mp_pose.PoseLandmark.LEFT_KNEE
        ankle_lm_side = mp_pose.PoseLandmark.LEFT_ANKLE
        heel_lm_side = mp_pose.PoseLandmark.LEFT_HEEL
        foot_index_lm_side = mp_pose.PoseLandmark.LEFT_FOOT_INDEX
    elif view_name == 'Right Side View' or view_name == "Side View (Single)": # Treat single view as right side for consistency here
        ear_lm_side = mp_pose.PoseLandmark.RIGHT_EAR
        shoulder_lm_side = mp_pose.PoseLandmark.RIGHT_SHOULDER
        hip_lm_side = mp_pose.PoseLandmark.RIGHT_HIP
        knee_lm_side = mp_pose.PoseLandmark.RIGHT_KNEE
        ankle_lm_side = mp_pose.PoseLandmark.RIGHT_ANKLE
        heel_lm_side = mp_pose.PoseLandmark.RIGHT_HEEL
        foot_index_lm_side = mp_pose.PoseLandmark.RIGHT_FOOT_INDEX

    if view_name in SIDE_VIEWS or view_name == "Side View (Single)":
        if ear_lm_side and shoulder_lm_side and hip_lm_side and all(is_visible(p) for p in [ear_lm_side, shoulder_lm_side, hip_lm_side]):
            metrics["ear_shoulder_hip_angle"] = calculate_angle(
                [lm[ear_lm_side.value].x, lm[ear_lm_side.value].y],
                [lm[shoulder_lm_side.value].x, lm[shoulder_lm_side.value].y],
                [lm[hip_lm_side.value].x, lm[hip_lm_side.value].y]
            )
        else: metrics["ear_shoulder_hip_angle"] = None # Or a default like 90

        if ear_lm_side and shoulder_lm_side and all(is_visible(p) for p in [ear_lm_side, shoulder_lm_side]):
            metrics["ear_shoulder_horizontal_distance"] = abs(lm[ear_lm_side.value].x - lm[shoulder_lm_side.value].x)
        else: metrics["ear_shoulder_horizontal_distance"] = None

        metrics["shoulder_z"] = lm[shoulder_lm_side.value].z if shoulder_lm_side and is_visible(shoulder_lm_side) else None
        metrics["hip_z"] = lm[hip_lm_side.value].z if hip_lm_side and is_visible(hip_lm_side) else None
        metrics["knee_z"] = lm[knee_lm_side.value].z if knee_lm_side and is_visible(knee_lm_side) else None
        metrics["foot_z_diff"] = abs(lm[heel_lm_side.value].z - lm[foot_index_lm_side.value].z) if heel_lm_side and foot_index_lm_side and all(is_visible(p) for p in [heel_lm_side, foot_index_lm_side]) else None

    if view_name in FRONT_BACK_VIEWS:
        metrics["shoulder_y_diff"] = abs(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y - lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) if all(is_visible(p) for p in [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]) else None
        metrics["hip_y_diff"] = abs(lm[mp_pose.PoseLandmark.LEFT_HIP.value].y - lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y) if all(is_visible(p) for p in [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]) else None
        metrics["ankle_x_diff"] = abs(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x - lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x) if all(is_visible(p) for p in [mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE]) else None
        metrics["knee_x_diff"] = abs(lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x - lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x) if all(is_visible(p) for p in [mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE]) else None
    
    return results.pose_landmarks, landmarked_pil_image, metrics


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
            elif not primary_display_image and landmarked_img: primary_display_image = landmarked_img
    
    def get_metric(name, views, default=None, use_average=False):
        valid_metrics = []
        for v_name in views:
            if v_name in all_metrics_by_view and all_metrics_by_view[v_name].get(name) is not None:
                if not use_average: return all_metrics_by_view[v_name][name]
                valid_metrics.append(all_metrics_by_view[v_name][name])
        if use_average and valid_metrics: return np.nanmean(valid_metrics)
        return default

    consolidated_metrics["shoulder_z"] = get_metric("shoulder_z", SIDE_VIEWS, use_average=True)
    consolidated_metrics["hip_z"] = get_metric("hip_z", SIDE_VIEWS, use_average=True)
    consolidated_metrics["knee_z"] = get_metric("knee_z", SIDE_VIEWS, use_average=True)
    consolidated_metrics["ear_shoulder_hip_angle"] = get_metric("ear_shoulder_hip_angle", SIDE_VIEWS, use_average=True, default=90.0) # Default to non-tech-neck angle
    consolidated_metrics["ear_shoulder_horizontal_distance"] = get_metric("ear_shoulder_horizontal_distance", SIDE_VIEWS, use_average=True, default=0.0)
    consolidated_metrics["foot_z_diff"] = get_metric("foot_z_diff", SIDE_VIEWS, use_average=True)
    
    consolidated_metrics["shoulder_y_diff"] = get_metric("shoulder_y_diff", ['Back View', 'Front View'])
    consolidated_metrics["hip_y_diff"] = get_metric("hip_y_diff", ['Back View', 'Front View']) 
    consolidated_metrics["ankle_x_diff"] = get_metric("ankle_x_diff", ['Front View', 'Back View'])
    consolidated_metrics["knee_x_diff"] = get_metric("knee_x_diff", ['Front View', 'Back View'])

    if "Kyphosis" in final_abnormalities and all(consolidated_metrics.get(k) is not None for k in ["shoulder_z", "hip_z"]): final_abnormalities["Kyphosis"] = (consolidated_metrics["shoulder_z"] - consolidated_metrics["hip_z"]) > KYPHOSIS_THRESHOLD_SHOULDER_HIP_Z_DIFF
    if "Lordosis" in final_abnormalities and all(consolidated_metrics.get(k) is not None for k in ["hip_z", "knee_z"]): final_abnormalities["Lordosis"] = (consolidated_metrics["hip_z"] - consolidated_metrics["knee_z"]) > LORDOSIS_THRESHOLD_HIP_KNEE_Z_DIFF
    if "Tech Neck" in final_abnormalities and consolidated_metrics.get("ear_shoulder_hip_angle") is not None and consolidated_metrics.get("ear_shoulder_horizontal_distance") is not None:
        final_abnormalities["Tech Neck"] = (consolidated_metrics["ear_shoulder_hip_angle"] < TECH_NECK_MAX_ESH_ANGLE and consolidated_metrics["ear_shoulder_horizontal_distance"] > TECH_NECK_MIN_ESH_HORIZ_DIST)
    if "Scoliosis" in final_abnormalities and consolidated_metrics.get("shoulder_y_diff") is not None: final_abnormalities["Scoliosis"] = consolidated_metrics["shoulder_y_diff"] > SCOLIOSIS_THRESHOLD_SHOULDER_Y_DIFF
    if "Flat Feet" in final_abnormalities and consolidated_metrics.get("foot_z_diff") is not None: final_abnormalities["Flat Feet"] = consolidated_metrics["foot_z_diff"] < FLAT_FEET_THRESHOLD_FOOT_Z_DIFF
    if "Gait Abnormalities" in final_abnormalities and consolidated_metrics.get("ankle_x_diff") is not None: final_abnormalities["Gait Abnormalities"] = consolidated_metrics["ankle_x_diff"] > GAIT_ABNORMALITIES_THRESHOLD_ANKLE_X_DIFF
    if "Knock Knees" in final_abnormalities and all(consolidated_metrics.get(k) is not None for k in ["knee_x_diff", "ankle_x_diff"]) and consolidated_metrics.get("ankle_x_diff",0)!=0: final_abnormalities["Knock Knees"] = consolidated_metrics["knee_x_diff"] < (consolidated_metrics.get("ankle_x_diff",0) * KNOCK_KNEES_KNEE_ANKLE_RATIO_THRESHOLD)
    if "Bow Legs" in final_abnormalities and all(consolidated_metrics.get(k) is not None for k in ["knee_x_diff", "ankle_x_diff"]) and consolidated_metrics.get("knee_x_diff",0)!=0: final_abnormalities["Bow Legs"] = consolidated_metrics.get("ankle_x_diff",0) < (consolidated_metrics.get("knee_x_diff",0) * BOW_LEGS_ANKLE_KNEE_RATIO_THRESHOLD)
    if not primary_display_image and all_landmarked_images_pil: primary_display_image = next(iter(all_landmarked_images_pil.values()), None)
    return final_abnormalities, consolidated_metrics, primary_display_image, all_landmarked_images_pil

# --- Main Application UI ---
child_name = st.text_input("Enter Child's Name (Mandatory):", key="child_name_input")
st.session_state.analysis_mode = st.radio("Select Analysis Mode:", ("Single View Analysis", "Multi-View Analysis (4 Views)"), key="analysis_mode_radio", on_change=clear_image_memory)
st.markdown("### Select Abnormalities to Detect")
select_all = st.checkbox("Select All", value=all(st.session_state.selected_abnormalities.values()), key="select_all_checkbox")
if select_all != all(st.session_state.selected_abnormalities.values()):
    st.session_state.selected_abnormalities = {k: select_all for k in POSTURE_RECOMMENDATIONS.keys()}
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
            cols_up = st.columns(2)
            all_up = True
            for i, vn in enumerate(VIEWS_SEQUENCE):
                with cols_up[i%2]:
                    uf_multi = st.file_uploader(f"Upload {vn}", type=["jpg","png","jpeg"], key=f"upload_{vn.lower().replace(' ','_')}")
                    if uf_multi: st.session_state[f"uploaded_image_{vn.lower().replace(' ','_')}"] = optimize_image(Image.open(uf_multi))
                    multi_images_data_pil[vn] = st.session_state.get(f"uploaded_image_{vn.lower().replace(' ','_')}")
                    if not multi_images_data_pil[vn]: all_up = False
            st.session_state.all_multi_images_uploaded = all_up
            if all_up: st.success("All 4 images uploaded.")
        else: # Camera for Multi-View
            cs = st.session_state.capture_stage
            if cs < len(VIEWS_SEQUENCE):
                vtc = VIEWS_SEQUENCE[cs]
                st.markdown(f"**Taking photo for: {vtc}**")
                if cs > 0: st.info(f"Photo for {VIEWS_SEQUENCE[cs-1]} captured. Turn for {vtc}.")
                cp = st.camera_input(f"Capture {vtc}", key=st.session_state.camera_input_key_multi)
                if cp:
                    st.session_state.captured_images_multi[vtc] = optimize_image(Image.fromarray(cv2.cvtColor(cv2.imdecode(np.asarray(bytearray(cp.read()),dtype=np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)))
                    st.session_state.capture_stage += 1
                    st.session_state.camera_input_key_multi = f"camera_multi_{st.session_state.capture_stage}"
                    st.rerun()
            else: st.session_state.all_multi_images_captured = True; st.success("All 4 views captured!"); multi_images_data_pil = st.session_state.captured_images_multi
            if any(st.session_state.captured_images_multi.values()):
                st.markdown("---"); st.write("Captured Images:"); cols_rev = st.columns(len(VIEWS_SEQUENCE))
                for i, vn_rev in enumerate(VIEWS_SEQUENCE):
                    if st.session_state.captured_images_multi[vn_rev]:
                        with cols_rev[i]: st.image(st.session_state.captured_images_multi[vn_rev],caption=vn_rev,width=150)
                if st.button("Retake All",key="retake_multi_cam"): clear_image_memory(); st.rerun()
            st.markdown("---")
else: st.warning("Enter child's name to proceed.")

# --- Analysis Trigger ---
btn_label = "Analyze All 4 Views" if st.session_state.analysis_mode == "Multi-View Analysis (4 Views)" else "Analyze Posture"
enable_btn = bool(child_name and ( (st.session_state.analysis_mode == "Single View Analysis" and single_image_data_pil) or \
                                (st.session_state.analysis_mode == "Multi-View Analysis (4 Views)" and \
                                 (st.session_state.all_multi_images_uploaded or st.session_state.all_multi_images_captured) and \
                                 all(multi_images_data_pil.get(v) for v in VIEWS_SEQUENCE) ) ) )


if st.button(btn_label, key="analyze_button", disabled=not enable_btn):
    st.session_state.processing_done = False 
    if st.session_state.analysis_mode == "Single View Analysis":
        if single_image_data_pil:
            _, landmarked_img, metrics = process_image_for_view(single_image_data_pil, "Side View (Single)") # Pass a generic side view name
            st.session_state.landmark_image = landmarked_img 
            st.session_state.all_landmark_images["Single View"] = landmarked_img
            current_abnormalities = {k: False for k,v in st.session_state.selected_abnormalities.items() if v}
            
            # Use new metrics for Tech Neck
            if "Tech Neck" in current_abnormalities and metrics.get("ear_shoulder_hip_angle") is not None and metrics.get("ear_shoulder_horizontal_distance") is not None:
                current_abnormalities["Tech Neck"] = (metrics["ear_shoulder_hip_angle"] < TECH_NECK_MAX_ESH_ANGLE and metrics["ear_shoulder_horizontal_distance"] > TECH_NECK_MIN_ESH_HORIZ_DIST)
            
            # Other abnormalities (ensure metrics keys match what process_image_for_view provides for a side view)
            if "Kyphosis" in current_abnormalities and metrics.get("shoulder_z") is not None and metrics.get("hip_z") is not None: current_abnormalities["Kyphosis"] = (metrics["shoulder_z"] - metrics["hip_z"]) > KYPHOSIS_THRESHOLD_SHOULDER_HIP_Z_DIFF
            if "Lordosis" in current_abnormalities and metrics.get("hip_z") is not None and metrics.get("knee_z") is not None: current_abnormalities["Lordosis"] = (metrics["hip_z"] - metrics["knee_z"]) > LORDOSIS_THRESHOLD_HIP_KNEE_Z_DIFF
            if "Flat Feet" in current_abnormalities and metrics.get("foot_z_diff") is not None: current_abnormalities["Flat Feet"] = metrics["foot_z_diff"] < FLAT_FEET_THRESHOLD_FOOT_Z_DIFF
            # Scoliosis, Gait, Knock Knees, Bow Legs are less reliable from single side view; might be mostly false or need specific instructions.
            # For single view, we might only populate metrics available from a side view.
            if "Scoliosis" in current_abnormalities and metrics.get("shoulder_y_diff") is not None: current_abnormalities["Scoliosis"] = metrics["shoulder_y_diff"] > SCOLIOSIS_THRESHOLD_SHOULDER_Y_DIFF 
            if "Gait Abnormalities" in current_abnormalities and metrics.get("ankle_x_diff") is not None: current_abnormalities["Gait Abnormalities"] = metrics["ankle_x_diff"] > GAIT_ABNORMALITIES_THRESHOLD_ANKLE_X_DIFF 
            if "Knock Knees" in current_abnormalities and metrics.get("knee_x_diff") is not None and metrics.get("ankle_x_diff") is not None and metrics.get("ankle_x_diff",0) != 0: current_abnormalities["Knock Knees"] = metrics["knee_x_diff"] < (metrics.get("ankle_x_diff",0) * KNOCK_KNEES_KNEE_ANKLE_RATIO_THRESHOLD)
            if "Bow Legs" in current_abnormalities and metrics.get("knee_x_diff") is not None and metrics.get("ankle_x_diff") is not None and metrics.get("knee_x_diff",0) != 0: current_abnormalities["Bow Legs"] = metrics.get("ankle_x_diff",0) < (metrics.get("knee_x_diff",0) * BOW_LEGS_ANKLE_KNEE_RATIO_THRESHOLD)

            st.session_state.abnormalities = current_abnormalities
            sid = st.session_state.get("current_student_id") or f"FN-{random.randint(1000,9999)}"; st.session_state.current_student_id = sid
            
            # Store all calculated metrics for the entry
            entry_metrics = {
                "shoulder_z": metrics.get("shoulder_z"), "hip_z": metrics.get("hip_z"), "knee_z": metrics.get("knee_z"),
                "ear_shoulder_hip_angle": metrics.get("ear_shoulder_hip_angle"), # New metric
                "ear_shoulder_horizontal_distance": metrics.get("ear_shoulder_horizontal_distance"), # New metric
                "shoulder_y_diff": metrics.get("shoulder_y_diff"), "foot_z_diff": metrics.get("foot_z_diff"),
                "ankle_x_diff": metrics.get("ankle_x_diff"), "knee_x_diff": metrics.get("knee_x_diff")
            }
            st.session_state.current_entry = {"Student ID":sid,"Student Name":child_name,"Timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),**current_abnormalities,**entry_metrics}
            for ab_key in POSTURE_RECOMMENDATIONS.keys():
                if ab_key not in st.session_state.current_entry: st.session_state.current_entry[ab_key] = False 
            st.session_state.processing_done = True

    elif st.session_state.analysis_mode == "Multi-View Analysis (4 Views)":
        images_for_analysis_final = multi_images_data_pil 
        if all(images_for_analysis_final.get(view) for view in VIEWS_SEQUENCE): 
            final_abnormalities, consolidated_metrics, primary_display_img, all_landmarked_imgs_pil = \
                analyze_multi_view_data(images_for_analysis_final, st.session_state.selected_abnormalities)
            st.session_state.abnormalities = final_abnormalities
            st.session_state.landmark_image = primary_display_img 
            st.session_state.all_landmark_images = all_landmarked_imgs_pil 
            sid = st.session_state.get("current_student_id") or f"FN-{random.randint(1000,9999)}"; st.session_state.current_student_id = sid
            st.session_state.current_entry = {"Student ID":sid,"Student Name":child_name,"Timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),**final_abnormalities,**consolidated_metrics}
            for ab_key in POSTURE_RECOMMENDATIONS.keys(): 
                if ab_key not in st.session_state.current_entry: st.session_state.current_entry[ab_key] = False
            st.session_state.processing_done = True
        else: st.error("Not all images for multi-view analysis are available.")

# --- Helper function to get abnormality reason string ---
def get_abnormality_reason_string(condition_name, metrics_dict):
    reason = ""
    # Use .get() with a default for metrics to avoid KeyErrors if a metric wasn't calculated
    if condition_name == "Kyphosis" and metrics_dict.get("shoulder_z") is not None and metrics_dict.get("hip_z") is not None:
        diff = metrics_dict["shoulder_z"] - metrics_dict["hip_z"]
        reason = f"(Sh-Hip Z: {diff:.2f} > {KYPHOSIS_THRESHOLD_SHOULDER_HIP_Z_DIFF})"
    elif condition_name == "Lordosis" and metrics_dict.get("hip_z") is not None and metrics_dict.get("knee_z") is not None:
        diff = metrics_dict["hip_z"] - metrics_dict["knee_z"]
        reason = f"(Hip-Knee Z: {diff:.2f} > {LORDOSIS_THRESHOLD_HIP_KNEE_Z_DIFF})"
    elif condition_name == "Tech Neck" and metrics_dict.get("ear_shoulder_hip_angle") is not None and metrics_dict.get("ear_shoulder_horizontal_distance") is not None:
        # Using new metrics for Tech Neck
        angle_val = metrics_dict.get("ear_shoulder_hip_angle", 90) # Default to a non-tech-neck angle if None
        dist_val = metrics_dict.get("ear_shoulder_horizontal_distance", 0)
        reason = f"(ESH Angle: {angle_val:.1f}° < {TECH_NECK_MAX_ESH_ANGLE}°, ESH Horiz Dist: {dist_val:.2f} > {TECH_NECK_MIN_ESH_HORIZ_DIST})"
    elif condition_name == "Scoliosis" and metrics_dict.get("shoulder_y_diff") is not None:
        reason = f"(Shoulder Y-diff: {metrics_dict['shoulder_y_diff']:.2f} > {SCOLIOSIS_THRESHOLD_SHOULDER_Y_DIFF})"
    elif condition_name == "Flat Feet" and metrics_dict.get("foot_z_diff") is not None:
        reason = f"(Foot Z-diff: {metrics_dict['foot_z_diff']:.2f} < {FLAT_FEET_THRESHOLD_FOOT_Z_DIFF})"
    elif condition_name == "Gait Abnormalities" and metrics_dict.get("ankle_x_diff") is not None: 
        reason = f"(Ankle X-diff: {metrics_dict['ankle_x_diff']:.2f} > {GAIT_ABNORMALITIES_THRESHOLD_ANKLE_X_DIFF})"
    elif condition_name == "Knock Knees" and metrics_dict.get("knee_x_diff") is not None and metrics_dict.get("ankle_x_diff") is not None:
        reason = f"(Knee X-diff: {metrics_dict['knee_x_diff']:.2f}, Ankle X-diff: {metrics_dict['ankle_x_diff']:.2f})"
    elif condition_name == "Bow Legs" and metrics_dict.get("knee_x_diff") is not None and metrics_dict.get("ankle_x_diff") is not None:
        reason = f"(Knee X-diff: {metrics_dict['knee_x_diff']:.2f}, Ankle X-diff: {metrics_dict['ankle_x_diff']:.2f})"
    return reason

# --- Display Results, Save, PDF ---
if st.session_state.processing_done and st.session_state.get("current_entry"):
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
        st.image(st.session_state.landmark_image, caption="Landmarked Image", use_container_width=True)

    display_abnormalities = st.session_state.get("abnormalities", {})
    current_metrics = st.session_state.current_entry 
    if display_abnormalities:
        st.write(f"### Abnormality Detection for {st.session_state.current_entry['Student Name']} (ID: {st.session_state.current_entry['Student ID']}):")
        for cond, pres in display_abnormalities.items():
            reason_str = ""
            if pres: 
                reason_str = get_abnormality_reason_string(cond, current_metrics)
            st.markdown(f"- {cond}: {'**Present**' if pres else 'Not Present'} {reason_str}")
    else: st.info("No abnormalities selected/detected based on the analysis.")
    
    col1_actions, col2_actions = st.columns(2) 
    with col1_actions:
        if st.button("💾 Save Result Locally", key="save_result_button_main"):
            st.session_state.records.append(st.session_state.current_entry)
            st.success(f"Result for {st.session_state.current_entry['Student ID']} saved locally!")
            if 'current_student_id' in st.session_state: del st.session_state['current_student_id']
            # Keep processing_done = True so PDF can be generated
    with col2_actions:
        if st.button("📄 Generate PDF Report", key="generate_pdf_button_main"):
            try:
                data_pdf = st.session_state.current_entry
                abn_pdf = st.session_state.abnormalities
                pdf = FPDF(); pdf.add_page(); pdf.set_auto_page_break(auto=True, margin=15) 
                pdf.set_font("Arial", "B", 16); pdf.cell(0, 10, "FitNurture Posture Analysis Report", ln=1, align="C")
                pdf.set_font("Arial", "", 9); pdf.cell(0, 7, "www.futurenurture.in", ln=1, align="C", link="http://www.futurenurture.in"); pdf.ln(5) 
                current_y_logo = pdf.get_y(); logo_pdf_path = next((p for p in logo_paths if os.path.exists(p)), None)
                if logo_pdf_path:
                    logo_width_pdf = 35; logo_height_pdf = 17.5 
                    if current_y_logo + logo_height_pdf > pdf.page_break_trigger - 5: pdf.add_page(); current_y_logo = pdf.get_y() 
                    pdf.image(logo_pdf_path, x=(210-logo_width_pdf)/2, y=current_y_logo, w=logo_width_pdf); pdf.set_y(current_y_logo + logo_height_pdf + 5) 
                pdf.set_font("Arial", "B", 12)
                details = {"Student Name": data_pdf.get('Student Name'), "Student ID": data_pdf.get('Student ID'), "Timestamp": data_pdf.get('Timestamp')}
                for k_pdf, v_pdf in details.items(): pdf.cell(0, 7, f"{k_pdf}: {v_pdf or 'N/A'}", ln=1)
                pdf.ln(5)

                if st.session_state.analysis_mode == "Multi-View Analysis (4 Views)" and st.session_state.all_landmark_images:
                    pdf.set_font("Arial", "B", 10)
                    pdf.cell(0, 7, "Processed Views:", ln=1, align='C')
                    pdf.ln(2)
                    pdf.set_font("Arial", "", 8)
                    img_display_width_pdf = 70 
                    img_spacing_horizontal_pdf = 10
                    row_start_x_pdf = (pdf.w - (img_display_width_pdf * 2 + img_spacing_horizontal_pdf)) / 2
                    image_paths_to_delete_pdf = []
                    
                    for i in range(0, len(VIEWS_SEQUENCE), 2):
                        current_y_for_row_pdf = pdf.get_y()
                        max_h_this_row_pdf = 0
                        view_name1_pdf = VIEWS_SEQUENCE[i]
                        pil_image1_pdf = st.session_state.all_landmark_images.get(view_name1_pdf)
                        if pil_image1_pdf:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_f1:
                                pil_image1_pdf.save(tmp_f1.name, format="JPEG"); image_paths_to_delete_pdf.append(tmp_f1.name)
                                o_w, o_h = pil_image1_pdf.size; asp = o_h/o_w if o_w > 0 else 1
                                img_h = img_display_width_pdf * asp
                                max_h_this_row_pdf = max(max_h_this_row_pdf, img_h)
                                if current_y_for_row_pdf + img_h + 10 > pdf.page_break_trigger: pdf.add_page(); current_y_for_row_pdf = pdf.get_y()
                                pdf.image(tmp_f1.name, x=row_start_x_pdf, y=current_y_for_row_pdf, w=img_display_width_pdf, h=img_h)
                                pdf.set_xy(row_start_x_pdf, current_y_for_row_pdf + img_h + 1)
                                pdf.multi_cell(img_display_width_pdf, 4, view_name1_pdf, 0, 'C')
                        if i + 1 < len(VIEWS_SEQUENCE):
                            view_name2_pdf = VIEWS_SEQUENCE[i+1]
                            pil_image2_pdf = st.session_state.all_landmark_images.get(view_name2_pdf)
                            if pil_image2_pdf:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_f2:
                                    pil_image2_pdf.save(tmp_f2.name, format="JPEG"); image_paths_to_delete_pdf.append(tmp_f2.name)
                                    o_w, o_h = pil_image2_pdf.size; asp = o_h/o_w if o_w > 0 else 1
                                    img_h = img_display_width_pdf * asp
                                    max_h_this_row_pdf = max(max_h_this_row_pdf, img_h)
                                    x_pos_img2_pdf = row_start_x_pdf + img_display_width_pdf + img_spacing_horizontal_pdf
                                    pdf.image(tmp_f2.name, x=x_pos_img2_pdf, y=current_y_for_row_pdf, w=img_display_width_pdf, h=img_h)
                                    pdf.set_xy(x_pos_img2_pdf, current_y_for_row_pdf + img_h + 1)
                                    pdf.multi_cell(img_display_width_pdf, 4, view_name2_pdf, 0, 'C')
                        if max_h_this_row_pdf > 0: pdf.set_y(current_y_for_row_pdf + max_h_this_row_pdf + 5 + 5) 
                        else: pdf.ln(5)
                    for path in image_paths_to_delete_pdf:
                        try: os.unlink(path)
                        except Exception: pass
                    pdf.ln(5)
                elif st.session_state.analysis_mode == "Single View Analysis" and st.session_state.get("landmark_image"): 
                    pil_image_pdf = st.session_state.landmark_image 
                    page_width_pdf = pdf.w - pdf.l_margin - pdf.r_margin; max_image_height_pdf_val = 80 
                    original_w_px_pdf, original_h_px_pdf = pil_image_pdf.size
                    aspect_ratio_pdf = original_h_px_pdf / original_w_px_pdf if original_w_px_pdf > 0 else 1
                    img_w_pdf_val = page_width_pdf * 0.70; img_h_pdf_val = img_w_pdf_val * aspect_ratio_pdf
                    if img_h_pdf_val > max_image_height_pdf_val: img_h_pdf_val = max_image_height_pdf_val; img_w_pdf_val = img_h_pdf_val / aspect_ratio_pdf if aspect_ratio_pdf > 0 else max_image_height_pdf_val
                    current_y_img_pdf = pdf.get_y()
                    if current_y_img_pdf + img_h_pdf_val > pdf.page_break_trigger - 5: pdf.add_page(); current_y_img_pdf = pdf.get_y()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img_f_pdf:
                        pil_image_pdf.save(tmp_img_f_pdf.name, format="JPEG")
                        img_x_pos_pdf = (pdf.w - img_w_pdf_val) / 2 
                        pdf.image(tmp_img_f_pdf.name, x=img_x_pos_pdf, y=current_y_img_pdf, w=img_w_pdf_val, h=img_h_pdf_val)
                    os.unlink(tmp_img_f_pdf.name); pdf.set_y(current_y_img_pdf + img_h_pdf_val + 5) 
                pdf.ln(3)
                pdf.set_font("Arial", "B", 11); pdf.cell(0, 7, "Detected Postural Issues:", ln=1) 
                pdf.set_font("Arial", "", 9); detected_cond_pdf = []
                if abn_pdf: 
                    for cond, pres in abn_pdf.items(): 
                        reason_str_pdf = ""
                        if pres: reason_str_pdf = get_abnormality_reason_string(cond, data_pdf) 
                        pdf.cell(0, 5, f"- {cond}: {'Present' if pres else 'Not Present'} {reason_str_pdf}", ln=1) 
                        if pres: detected_cond_pdf.append(cond) 
                else: pdf.cell(0,5, "- No abnormalities selected for detection or none found.", ln=1)
                pdf.ln(2) 
                if detected_cond_pdf:
                    pdf.set_font("Arial", "B", 11); pdf.cell(0, 7, "General Recommendations:", ln=1) 
                    pdf.set_font("Arial", "", 9); available_width_pdf = pdf.w - pdf.l_margin - pdf.r_margin - 5 
                    for cond in detected_cond_pdf:
                        if cond in POSTURE_RECOMMENDATIONS:
                            pdf.set_font("Arial", "B", 9); pdf.multi_cell(available_width_pdf, 5, f"For {cond}:") 
                            pdf.set_font("Arial", "", 9)
                            for rec_item in POSTURE_RECOMMENDATIONS[cond]:
                                clean_rec_item = rec_item.strip(); pdf.set_x(pdf.l_margin + 5); pdf.multi_cell(available_width_pdf - 5, 4, clean_rec_item) 
                            pdf.ln(1) 
                pdf.ln(2) 
                disclaimer_height_estimate_pdf = 15 
                if pdf.get_y() + disclaimer_height_estimate_pdf > pdf.page_break_trigger -5: pdf.add_page()
                pdf.set_font("Arial", "I", 7) 
                disclaimer_text_pdf = "Disclaimer: This automated analysis is for informational purposes only and not a substitute for professional medical advice. Consult a healthcare provider for health concerns."
                pdf.multi_cell(0, 3.5, disclaimer_text_pdf, align="C") 
                pdf_output_data = pdf.output(dest='S') 
                if isinstance(pdf_output_data, str): pdf_bytes_out = pdf_output_data.encode('latin-1')
                elif isinstance(pdf_output_data, bytearray): pdf_bytes_out = bytes(pdf_output_data)
                elif isinstance(pdf_output_data, bytes): pdf_bytes_out = pdf_output_data
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
    total_records = len(st.session_state.records)
    total_pages = (total_records + records_per_page - 1) // records_per_page if total_records > 0 else 0

    if total_pages > 0:
        st.session_state.current_page_local_records = st.selectbox("Select Page", options=range(total_pages), format_func=lambda x: f"Page {x+1}", index=st.session_state.current_page_local_records, key="local_records_page_selector")
    
    search_term = st.text_input("🔍 Search by Student Name or ID in local records", key="search_local")
    display_records_df = pd.DataFrame(st.session_state.records) 
    if search_term:
        display_records_df = display_records_df[
            display_records_df['Student Name'].str.contains(search_term, case=False, na=False) | 
            display_records_df['Student ID'].str.contains(search_term, case=False, na=False)
        ]
    total_filtered_records = len(display_records_df)
    total_filtered_pages = (total_filtered_records + records_per_page - 1) // records_per_page if total_filtered_records > 0 else 0
    if st.session_state.current_page_local_records >= total_filtered_pages and total_filtered_pages > 0:
        st.session_state.current_page_local_records = total_filtered_pages - 1
    elif total_filtered_pages == 0: st.session_state.current_page_local_records = 0
    start_idx = st.session_state.current_page_local_records * records_per_page
    end_idx = start_idx + records_per_page
    if not display_records_df.empty: st.dataframe(display_records_df.iloc[start_idx:end_idx], use_container_width=True)
    elif search_term: st.info("No local records match your search criteria.")
    else: st.info("No local records to display.")
    @st.cache_data 
    def convert_all_to_csv(records_list_cache): 
        if not records_list_cache: return b""
        return pd.DataFrame(records_list_cache).to_csv(index=False).encode("utf-8")
    if st.session_state.records: 
        csv_all = convert_all_to_csv(list(st.session_state.records)) 
        st.download_button("📥 Download All Local Records (CSV)", data=csv_all, file_name="all_posture_records.csv", mime="text/csv", key="download_all_csv")
else: st.info("No records saved locally yet.")

st.markdown("---"); st.subheader("☁️ Cloud Data Storage")
# Display cloud upload status if it exists
if st.session_state.cloud_upload_status:
    status_type = st.session_state.cloud_upload_status.get("type", "info")
    status_message = st.session_state.cloud_upload_status.get("message", "")
    if status_type == "success":
        st.success(status_message)
    elif status_type == "error":
        st.error(status_message)
    elif status_type == "warning":
        st.warning(status_message)
    else:
        st.info(status_message)

if st.session_state.get('records'):
    if st.button("⬆️ Upload All Saved Records to Cloud", key="upload_to_azure_button"):
        st.session_state.cloud_upload_status = None # Clear previous status
        with st.spinner("Connecting to database and uploading records..."):
            conn_result = get_db_connection() 
            if conn_result.get("type") == "success":
                conn = conn_result["connection"]
                upload_status = upload_records_to_sql(conn, list(st.session_state.records))
                st.session_state.cloud_upload_status = upload_status # Store status
                try: conn.close()
                except pyodbc.Error as e: 
                    if st.session_state.cloud_upload_status and st.session_state.cloud_upload_status.get("type") == "error":
                        st.session_state.cloud_upload_status["message"] += f" (Also, minor error closing DB connection: {e})"
                    else:
                        st.session_state.cloud_upload_status = {"type": "warning", "message": f"Minor error closing DB connection: {e}"}
                st.rerun() 
            else:
                st.session_state.cloud_upload_status = conn_result 
                st.rerun() 
else: st.info("No records saved locally to upload to Azure SQL.")

st.markdown("---") 
button_col1_manual, button_col2_manual, button_col3_manual = st.columns([2, 1, 2]) 
with button_col2_manual:
    manual_path = os.path.join("assets", "FitNurture_User_Manual.pdf")
    if os.path.exists(manual_path):
        try:
            with open(manual_path, "rb") as f_manual:
                st.download_button(label="Download User Manual (PDF)", data=f_manual.read(), file_name="FitNurture_User_Manual.pdf", mime="application/pdf", key="download_manual_button")
        except Exception as e: st.warning(f"Could not read user manual: {e}")
    else: st.warning("User manual PDF not found in assets folder (expected: assets/FitNurture_User_Manual.pdf).")

st.markdown("""<div class="copyright-footer">© Copyright 2025 FutureNurture | <a href="http://www.futurenurture.in" target="_blank">www.futurenurture.in</a></div>""", unsafe_allow_html=True)
