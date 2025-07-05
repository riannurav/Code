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

# --- MediaPipe Initialization ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# --- Application Constants ---
LANDMARK_VISIBILITY_THRESHOLD = 0.5
DB_TABLE_NAME = "PostureRecords"
TEXT_OFFSET_X = 70
ARROW_TIP_LENGTH = 0.3
TEXT_FONT_SCALE = 0.5
TEXT_THICKNESS = 1
HIGHLIGHT_COLOR_BGR = (0, 0, 255)
TEXT_BG_COLOR_BGR = (255, 255, 255)

# Original/Default Thresholds
DEFAULT_THRESHOLDS = {
    "kyphosis": 0.15,
    "lordosis": 0.10,
    "tech_neck_angle": 75.0, # Angle should be LESS than this
    "tech_neck_dist": 0.08, # Ear should be MORE forward than this
    "scoliosis": 0.05,
    "flat_feet": 0.05, # Arch height (z-diff) LESS than this
    "gait": 0.25,
    "knock_knees": 0.10, # Ratio of knee_x_diff to ankle_x_diff
    "bow_legs": 1.5, # Ratio of knee_x_diff to ankle_x_diff
}

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

# --- Session State Initialization (Done once at the top) ---
def initialize_session_state():
    default_session_states = {
        'page': 'Posture Analysis',
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
        'thresholds': DEFAULT_THRESHOLDS.copy(),
    }
    for view_name_key_init in [f"uploaded_image_{view.lower().replace(' ', '_')}" for view in VIEWS_SEQUENCE]:
        default_session_states[view_name_key_init] = None

    for key, value in default_session_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- SHARED FUNCTIONS AND UTILITIES ---
def clear_image_memory():
    """Clears image and processing related states, keeps student info and settings."""
    st.session_state.landmark_image = None
    st.session_state.all_landmark_images = {}
    st.session_state.abnormalities = {}
    st.session_state.processing_done = False
    st.session_state.capture_stage = 0
    st.session_state.captured_images_multi = {view: None for view in VIEWS_SEQUENCE}
    st.session_state.all_multi_images_captured = False
    st.session_state.camera_input_key_multi = "camera_multi_0"
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
    return mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

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
    cols = {
        "Student_ID": "NVARCHAR(50) NOT NULL PRIMARY KEY", "Student_Name": "NVARCHAR(255) NULL",
        "Age_Group": "NVARCHAR(50) NULL", "Gender": "NVARCHAR(20) NULL", "Loose_Clothing": "BIT NULL",
        "Observation_Timestamp": "DATETIME2 NULL", "UploadTimestamp": "DATETIME2 DEFAULT GETDATE() NULL"
    }
    for key in POSTURE_RECOMMENDATIONS.keys(): cols[key.replace(' ', '_').replace('-', '_')] = "BIT NULL"
    metrics_keys = ["shoulder_z", "hip_z", "knee_z", "ear_shoulder_hip_angle", "ear_shoulder_horizontal_distance", "shoulder_y_diff", "foot_z_diff", "ankle_x_diff", "knee_x_diff"]
    for key in metrics_keys: cols[key] = "FLOAT NULL"
    for key in DEFAULT_THRESHOLDS.keys(): cols[f"Thresh_{key}"] = "FLOAT NULL"
    
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
    base_sql_cols = ["Student_ID", "Student_Name", "Age_Group", "Gender", "Loose_Clothing", "Observation_Timestamp"]
    abnormality_sql_cols = [k.replace(' ', '_').replace('-', '_') for k in POSTURE_RECOMMENDATIONS.keys()]
    metrics_sql_cols = ["shoulder_z", "hip_z", "knee_z", "ear_shoulder_hip_angle", "ear_shoulder_horizontal_distance", "shoulder_y_diff", "foot_z_diff", "ankle_x_diff", "knee_x_diff"]
    threshold_sql_cols = [f"Thresh_{key}" for key in DEFAULT_THRESHOLDS.keys()]
    
    all_sql_cols = base_sql_cols + abnormality_sql_cols + metrics_sql_cols + threshold_sql_cols
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
            bool(record.get("Loose_Clothing", False)),
            record.get("Timestamp")
        ] + [bool(record.get(k, False)) for k in POSTURE_RECOMMENDATIONS.keys()] + \
        [float(record.get(k)) if record.get(k) is not None else None for k in metrics_sql_cols] + \
        [float(record.get(f"Thresh_{k}")) if record.get(f"Thresh_{k}") is not None else None for k in DEFAULT_THRESHOLDS.keys()]

        student_id_val = record.get("Student ID")
        if not student_id_val:
            error_messages.append(f"Skipping record (missing Student ID): {record.get('Student Name', 'N/A')}"); error_count +=1; continue
        try:
            cursor.execute(insert_sql, tuple(values_for_insert)); insert_count += 1
        except pyodbc.IntegrityError:
            try: cursor.execute(update_sql, tuple(values_for_insert[1:] + [student_id_val])); update_count += 1
            except pyodbc.Error as ue: error_messages.append(f"Error updating '{student_id_val}': {ue}"); error_count += 1; conn.rollback()
        except pyodbc.Error as e: error_messages.append(f"DB Error for '{student_id_val}': {e}"); error_count += 1; conn.rollback()
        except Exception as ex: error_messages.append(f"Unexpected error for '{student_id_val}': {ex}"); error_count += 1; conn.rollback()
        if error_count > 0: break

    final_status = {}
    if error_count > 0: final_status = {"type": "error", "message": f"{error_count} record(s) failed. Batch processing stopped. Errors: {'; '.join(error_messages)}"}
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
    else: final_status = {"type": "info", "message": "No records were processed for upload."}
    try: cursor.close()
    except pyodbc.Error: pass
    return final_status

def process_image_for_view(image_pil, view_name="Unknown View"):
    if image_pil is None: return None, None, {}
    img_np = np.array(image_pil);
    if img_np.shape[-1] == 4: img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    elif len(img_np.shape) == 2: img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    results = pose_static.process(img_np)
    landmarked_pil_image = image_pil
    if not results or not results.pose_landmarks:
        st.warning(f"No person/landmarks detected in {view_name}."); return None, landmarked_pil_image, {}
    lm = results.pose_landmarks.landmark
    img_bgr_for_drawing = cv2.cvtColor(img_np.copy(), cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(img_bgr_for_drawing, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2))
    img_bgr_with_labels = add_landmark_labels(img_bgr_for_drawing, results)
    landmarked_pil_image = Image.fromarray(cv2.cvtColor(img_bgr_with_labels, cv2.COLOR_BGR2RGB))
    metrics = {}
    def is_visible(le): return lm[le.value].visibility > LANDMARK_VISIBILITY_THRESHOLD if le.value < len(lm) else False

    ear_lm, shoulder_lm, hip_lm, knee_lm, ankle_lm, heel_lm, foot_lm = (mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.LEFT_FOOT_INDEX) \
        if view_name == 'Left Side View' else (mp_pose.PoseLandmark.RIGHT_EAR, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)

    if view_name in SIDE_VIEWS or view_name == "Side View (Single)":
        if all(is_visible(p) for p in [ear_lm, shoulder_lm, hip_lm]):
            metrics["ear_shoulder_hip_angle"] = calculate_angle([lm[ear_lm.value].x, lm[ear_lm.value].y], [lm[shoulder_lm.value].x, lm[shoulder_lm.value].y], [lm[hip_lm.value].x, lm[hip_lm.value].y])
        if all(is_visible(p) for p in [ear_lm, shoulder_lm]):
            metrics["ear_shoulder_horizontal_distance"] = abs(lm[ear_lm.value].x - lm[shoulder_lm.value].x)
        if is_visible(shoulder_lm): metrics["shoulder_z"] = lm[shoulder_lm.value].z
        if is_visible(hip_lm): metrics["hip_z"] = lm[hip_lm.value].z
        if is_visible(knee_lm): metrics["knee_z"] = lm[knee_lm.value].z
        if all(is_visible(p) for p in [heel_lm, foot_lm]): metrics["foot_z_diff"] = abs(lm[heel_lm.value].z - lm[foot_lm.value].z)

    if view_name in FRONT_BACK_VIEWS:
        if all(is_visible(p) for p in [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]): metrics["shoulder_y_diff"] = abs(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y - lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
        if all(is_visible(p) for p in [mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE]): metrics["ankle_x_diff"] = abs(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x - lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x)
        if all(is_visible(p) for p in [mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE]): metrics["knee_x_diff"] = abs(lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x - lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x)

    return results.pose_landmarks, landmarked_pil_image, metrics

def apply_clothing_adjustment(base_threshold):
    return base_threshold * (CLOTHING_ADJUSTMENT_FACTOR if st.session_state.loose_clothing else 1.0)

def analyze_multi_view_data(multi_images_pil_dict, selected_abnormalities_config, thresholds):
    all_metrics_by_view, all_landmarked_images_pil = {}, {}
    consolidated_metrics, final_abnormalities = {}, {k: False for k,v in selected_abnormalities_config.items() if v}

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

    eff_kyphosis = apply_clothing_adjustment(thresholds['kyphosis'])
    eff_lordosis = apply_clothing_adjustment(thresholds['lordosis'])
    eff_scoliosis = apply_clothing_adjustment(thresholds['scoliosis'])

    if "Kyphosis" in final_abnormalities and all(consolidated_metrics.get(k) is not None for k in ["shoulder_z", "hip_z"]): final_abnormalities["Kyphosis"] = (consolidated_metrics["shoulder_z"] - consolidated_metrics["hip_z"]) > eff_kyphosis
    if "Lordosis" in final_abnormalities and all(consolidated_metrics.get(k) is not None for k in ["hip_z", "knee_z"]): final_abnormalities["Lordosis"] = (consolidated_metrics["hip_z"] - consolidated_metrics["knee_z"]) > eff_lordosis
    if "Tech Neck" in final_abnormalities and all(consolidated_metrics.get(k) is not None for k in ["ear_shoulder_hip_angle", "ear_shoulder_horizontal_distance"]):
        final_abnormalities["Tech Neck"] = (consolidated_metrics["ear_shoulder_hip_angle"] < thresholds['tech_neck_angle'] and consolidated_metrics["ear_shoulder_horizontal_distance"] > thresholds['tech_neck_dist'])
    if "Scoliosis" in final_abnormalities and consolidated_metrics.get("shoulder_y_diff") is not None: final_abnormalities["Scoliosis"] = consolidated_metrics["shoulder_y_diff"] > eff_scoliosis
    if "Flat Feet" in final_abnormalities and consolidated_metrics.get("foot_z_diff") is not None: final_abnormalities["Flat Feet"] = consolidated_metrics["foot_z_diff"] < thresholds['flat_feet']
    if "Gait Abnormalities" in final_abnormalities and consolidated_metrics.get("ankle_x_diff") is not None: final_abnormalities["Gait Abnormalities"] = consolidated_metrics["ankle_x_diff"] > thresholds['gait']
    if "Knock Knees" in final_abnormalities and all(consolidated_metrics.get(k) is not None for k in ["knee_x_diff", "ankle_x_diff"]) and consolidated_metrics.get("ankle_x_diff",0) > 0:
        final_abnormalities["Knock Knees"] = consolidated_metrics["knee_x_diff"] < (consolidated_metrics.get("ankle_x_diff",0) * thresholds['knock_knees'])
    if "Bow Legs" in final_abnormalities and all(consolidated_metrics.get(k) is not None for k in ["knee_x_diff", "ankle_x_diff"]):
        ankle_x = consolidated_metrics.get("ankle_x_diff", 0); knee_x = consolidated_metrics.get("knee_x_diff", 0)
        if ankle_x > 0 and knee_x > 0 and (ankle_x / knee_x) < (1 / thresholds['bow_legs']): final_abnormalities["Bow Legs"] = True

    primary_img = all_landmarked_images_pil.get('Left Side View') or next(iter(all_landmarked_images_pil.values()), None)
    return final_abnormalities, consolidated_metrics, primary_img, all_landmarked_images_pil

def get_gemini_suggestions(abnormalities_detected_dict, student_name, age_group, gender):
    st.session_state.gemini_suggestions = None; st.session_state.gemini_suggestions_error = None
    detected_issues = [name for name, present in abnormalities_detected_dict.items() if present]
    if not detected_issues: return "No specific postural issues were detected. Focus on maintaining good overall posture and regular physical activity."

    prompt = f"""
    FitNurture has analyzed the posture for {student_name} (Age Group: {age_group}, Gender: {gender}) and identified: {', '.join(detected_issues)}.
    Provide personalized, actionable advice for {student_name} to improve these conditions, keeping their age group and gender in mind.
    Your response should be encouraging, easy to understand, and structured in Markdown as follows:
    ### ✨ Personalized Posture Plan for {student_name} ✨
    Based on the analysis, here are some suggestions:
    #### Corrective Exercises & Stretches:
    For each issue, list 2-3 specific exercises (e.g., **Wall Angels:** description).
    #### General Lifestyle Adjustments:
    Provide 3-5 general tips (e.g., **Screen Time Management:** explanation).
    #### Important Note:
    Include a disclaimer about this being informational advice and to consult a professional.
    """
    try:
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        if result.get('candidates') and result['candidates'][0].get('content'):
            suggestions = result['candidates'][0]['content']['parts'][0]['text']
            st.session_state.gemini_suggestions = suggestions; return suggestions
        else:
            st.session_state.gemini_suggestions_error = "Could not retrieve valid suggestions from the AI."; return None
    except Exception as e:
        st.session_state.gemini_suggestions_error = f"An unexpected error occurred while fetching AI suggestions: {e}"; return None

# --- Main App Page ---
def main_app_page():
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
    </style>
    """, unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font-size: 24px; margin-bottom: 20px;'>FitNurture : Posture Detection</h2>", unsafe_allow_html=True)
    
    child_name = st.text_input("Enter Child's Name (Mandatory):", key="child_name_input", value=st.session_state.get('current_entry',{}).get('Student Name',''))

    if child_name:
        st.session_state.selected_age_group = st.selectbox("Select Age Group:", options=AGE_GROUPS, key="age_group_select", index=AGE_GROUPS.index(st.session_state.selected_age_group))
        st.session_state.selected_gender = st.radio("Select Gender:", options=GENDERS, key="gender_radio", index=GENDERS.index(st.session_state.selected_gender), horizontal=True)
        st.session_state.loose_clothing = st.checkbox("Subject is NOT wearing body-fitting clothes", key="loose_clothing_checkbox", value=st.session_state.loose_clothing)
        st.markdown("---")

    st.session_state.analysis_mode = st.radio("Select Analysis Mode:", ("Single View Analysis", "Multi-View Analysis (4 Views)"), key="analysis_mode_radio", on_change=clear_image_memory)
    st.markdown("### Select Abnormalities to Detect")
    if st.checkbox("Select All", value=all(st.session_state.selected_abnormalities.values()), key="select_all_checkbox"):
        if not all(st.session_state.selected_abnormalities.values()):
            st.session_state.selected_abnormalities = {k: True for k in POSTURE_RECOMMENDATIONS.keys()}; st.rerun()
    else:
        if all(st.session_state.selected_abnormalities.values()):
            st.session_state.selected_abnormalities = {k: False for k in POSTURE_RECOMMENDATIONS.keys()}; st.rerun()

    cols_abnorm = st.columns(2)
    for i, (abn, val) in enumerate(st.session_state.selected_abnormalities.items()):
        with cols_abnorm[i % 2]: st.session_state.selected_abnormalities[abn] = st.checkbox(abn, value=val, key=f"cb_{abn}")
    st.markdown("---")

    with st.expander("⚙️ Advanced: Adjust Detection Thresholds"):
        st.info("Adjust these values to change the sensitivity of the detection. Higher values for differences/distances and lower values for angles generally make detection stricter.")
        thresh = st.session_state.thresholds
        
        thresh['kyphosis'] = st.number_input("Kyphosis (Sh-Hip Z-Diff >)", min_value=0.01, max_value=0.5, value=thresh['kyphosis'], step=0.01, format="%.2f", help="Higher value requires more forward shoulder slouch.")
        thresh['lordosis'] = st.number_input("Lordosis (Hip-Knee Z-Diff >)", min_value=0.01, max_value=0.5, value=thresh['lordosis'], step=0.01, format="%.2f", help="Higher value requires more pronounced lower back curve.")
        col_tn1, col_tn2 = st.columns(2)
        with col_tn1: thresh['tech_neck_angle'] = st.number_input("Tech Neck (ESH Angle <)", min_value=45.0, max_value=90.0, value=thresh['tech_neck_angle'], step=0.5, format="%.1f", help="Lower value requires more forward head tilt.")
        with col_tn2: thresh['tech_neck_dist'] = st.number_input("Tech Neck (ESH Horiz Dist >)", min_value=0.0, max_value=0.3, value=thresh['tech_neck_dist'], step=0.01, format="%.2f", help="Higher value requires ear to be more forward of shoulder.")
        thresh['scoliosis'] = st.number_input("Scoliosis (Shoulder Y-Diff >)", min_value=0.01, max_value=0.2, value=thresh['scoliosis'], step=0.01, format="%.2f", help="Higher value requires a greater height difference between shoulders.")
        thresh['flat_feet'] = st.number_input("Flat Feet (Foot Arch <)", min_value=0.01, max_value=0.15, value=thresh['flat_feet'], step=0.01, format="%.2f", help="Lower value requires a flatter foot arch.")
        thresh['gait'] = st.number_input("Gait Abnormality (Ankle X-Diff >)", min_value=0.1, max_value=0.5, value=thresh['gait'], step=0.01, format="%.2f", help="Higher value means feet are wider apart.")
        col_kk, col_bl = st.columns(2)
        with col_kk: thresh['knock_knees'] = st.number_input("Knock Knees (Knee/Ankle Ratio <)", min_value=0.05, max_value=0.95, value=thresh['knock_knees'], step=0.01, format="%.2f", help="Lower value means knees must be much closer than ankles.")
        with col_bl: thresh['bow_legs'] = st.number_input("Bow Legs (Knee/Ankle Ratio >)", min_value=1.1, max_value=3.0, value=thresh['bow_legs'], step=0.05, format="%.2f", help="Higher value means knees must be much wider than ankles.")
        
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
                if cd: single_image_data_pil = optimize_image(Image.fromarray(cv2.cvtColor(cv2.imdecode(np.asarray(bytearray(cd.read()),dtype=np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)))
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
                        st.session_state.captured_images_multi[vtc] = optimize_image(Image.fromarray(cv2.cvtColor(cv2.imdecode(np.asarray(bytearray(cp.read()),dtype=np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)))
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

    btn_label = "Analyze Posture"
    enable_btn = bool(child_name and ( (st.session_state.analysis_mode == "Single View Analysis" and single_image_data_pil) or \
                                    (st.session_state.analysis_mode == "Multi-View Analysis (4 Views)" and \
                                    (st.session_state.all_multi_images_uploaded or st.session_state.all_multi_images_captured) and \
                                    all(multi_images_data_pil.get(v) or st.session_state.captured_images_multi.get(v) for v in VIEWS_SEQUENCE) ) ) )

    if st.button(btn_label, key="analyze_button", disabled=not enable_btn):
        st.session_state.processing_done = False; st.session_state.gemini_suggestions = None
        sid = st.session_state.get("current_student_id") or f"FN-{random.randint(1000,9999)}"
        st.session_state.current_student_id = sid
        base_entry_info = {"Student ID": sid, "Student Name": child_name, "Age_Group": st.session_state.selected_age_group,
                           "Gender": st.session_state.selected_gender, "Loose_Clothing": st.session_state.loose_clothing,
                           "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        current_thresholds = st.session_state.thresholds
        threshold_entry_info = {f"Thresh_{k}": v for k, v in current_thresholds.items()}

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
                if "Bow Legs" in current_abnormalities and all(metrics.get(k) is not None for k in ["knee_x_diff", "ankle_x_diff"]):
                    ankle_x = metrics.get("ankle_x_diff", 0); knee_x = metrics.get("knee_x_diff", 0)
                    if ankle_x > 0 and knee_x > 0 and (ankle_x / knee_x) < (1/current_thresholds['bow_legs']): current_abnormalities["Bow Legs"] = True

                st.session_state.abnormalities = current_abnormalities
                st.session_state.current_entry = {**base_entry_info, **current_abnormalities, **metrics, **threshold_entry_info}
                st.session_state.processing_done = True
        else:
            images_for_analysis = multi_images_data_pil if st.session_state.all_multi_images_uploaded else st.session_state.captured_images_multi
            if all(images_for_analysis.get(v) for v in VIEWS_SEQUENCE):
                final_abns, cons_metrics, p_img, all_l_imgs = analyze_multi_view_data(images_for_analysis, st.session_state.selected_abnormalities, current_thresholds)
                st.session_state.abnormalities = final_abns
                st.session_state.landmark_image = p_img
                st.session_state.all_landmark_images = all_l_imgs
                st.session_state.current_entry = {**base_entry_info, **final_abns, **cons_metrics, **threshold_entry_info}
                st.session_state.processing_done = True
            else: st.error("Not all images for multi-view analysis are available.")

    def get_abnormality_reason_string(condition_name, metrics_dict, thresholds):
        reason = ""
        try:
            if condition_name == "Kyphosis" and all(k in metrics_dict and metrics_dict[k] is not None for k in ["shoulder_z", "hip_z"]):
                reason = f"(Sh-Hip Z: {metrics_dict['shoulder_z'] - metrics_dict['hip_z']:.2f} > {thresholds['kyphosis']})"
            elif condition_name == "Lordosis" and all(k in metrics_dict and metrics_dict[k] is not None for k in ["hip_z", "knee_z"]):
                reason = f"(Hip-Knee Z: {metrics_dict['hip_z'] - metrics_dict['knee_z']:.2f} > {thresholds['lordosis']})"
            elif condition_name == "Tech Neck" and all(k in metrics_dict and metrics_dict[k] is not None for k in ["ear_shoulder_hip_angle", "ear_shoulder_horizontal_distance"]):
                reason = f"(Angle: {metrics_dict['ear_shoulder_hip_angle']:.1f}° < {thresholds['tech_neck_angle']}°, Dist: {metrics_dict['ear_shoulder_horizontal_distance']:.2f} > {thresholds['tech_neck_dist']})"
            elif condition_name == "Scoliosis" and "shoulder_y_diff" in metrics_dict and metrics_dict['shoulder_y_diff'] is not None:
                reason = f"(Shoulder Y-diff: {metrics_dict['shoulder_y_diff']:.2f} > {thresholds['scoliosis']})"
            elif condition_name == "Flat Feet" and "foot_z_diff" in metrics_dict and metrics_dict['foot_z_diff'] is not None:
                reason = f"(Foot Arch: {metrics_dict['foot_z_diff']:.2f} < {thresholds['flat_feet']})"
            elif condition_name == "Gait Abnormalities" and "ankle_x_diff" in metrics_dict and metrics_dict['ankle_x_diff'] is not None:
                reason = f"(Ankle X-diff: {metrics_dict['ankle_x_diff']:.2f} > {thresholds['gait']})"
            elif condition_name in ["Knock Knees", "Bow Legs"] and all(k in metrics_dict and metrics_dict[k] is not None for k in ["knee_x_diff", "ankle_x_diff"]):
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
        st.markdown('<div class="gemini-button">', unsafe_allow_html=True)
        if st.button("✨ Get AI Exercise & Lifestyle Tips", key="gemini_tips_button"):
            with st.spinner("✨ Our AI is crafting personalized tips..."):
                get_gemini_suggestions(st.session_state.abnormalities, st.session_state.current_entry['Student Name'],
                                       st.session_state.current_entry['Age_Group'], st.session_state.current_entry['Gender'])
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.gemini_suggestions: st.markdown("### ✨ AI-Powered Suggestions"); st.markdown(st.session_state.gemini_suggestions)
        elif st.session_state.gemini_suggestions_error: st.error(f"Could not fetch AI suggestions: {st.session_state.gemini_suggestions_error}")
        
        st.markdown("---")
        col1_actions, col2_actions = st.columns(2)
        with col1_actions:
            if st.button("💾 Save Result Locally", key="save_result_button"):
                st.session_state.current_entry["Gemini_Suggestions"] = st.session_state.gemini_suggestions
                st.session_state.records.append(st.session_state.current_entry.copy())
                st.success(f"Result for {st.session_state.current_entry['Student ID']} saved locally!")
        with col2_actions:
            if st.button("📄 Generate PDF Report", key="generate_pdf_button"):
                with st.spinner("Generating PDF..."):
                    try:
                        data_pdf = st.session_state.current_entry
                        abn_pdf = st.session_state.abnormalities
                        gemini_sug_pdf = st.session_state.get("gemini_suggestions")

                        pdf = FPDF(); pdf.add_page(); pdf.set_auto_page_break(auto=True, margin=15)
                        pdf.set_font("Arial", "B", 16); pdf.cell(0, 10, "FitNurture Posture Analysis Report", ln=1, align="C")
                        pdf.set_font("Arial", "", 9); pdf.cell(0, 7, "www.futurenurture.in", ln=1, align="C", link="http://www.futurenurture.in"); pdf.ln(5)
                        
                        logo_paths = [os.path.join("assets", name) for name in ["logo.jpg", "logo.JPG", "logo.png", "logo.PNG"]]
                        current_y_logo = pdf.get_y(); logo_pdf_path = next((p for p in logo_paths if os.path.exists(p)), None)
                        if logo_pdf_path:
                            logo_width_pdf = 35; logo_height_pdf = 17.5
                            if current_y_logo + logo_height_pdf > pdf.page_break_trigger - 5: pdf.add_page(); current_y_logo = pdf.get_y()
                            pdf.image(logo_pdf_path, x=(210-logo_width_pdf)/2, y=current_y_logo, w=logo_width_pdf); pdf.set_y(current_y_logo + logo_height_pdf + 5)
                        
                        pdf.set_font("Arial", "B", 12)
                        details_pdf = { "Student Name": data_pdf.get('Student Name'), "Student ID": data_pdf.get('Student ID'), "Age Group": data_pdf.get('Age_Group'), "Gender": data_pdf.get('Gender'), "Timestamp": data_pdf.get('Timestamp')}
                        for k_pdf, v_pdf in details_pdf.items(): pdf.cell(0, 7, f"{k_pdf}: {v_pdf or 'N/A'}", ln=1)

                        pdf.set_font("Arial", "", 10); clothing_status_pdf = "Yes" if data_pdf.get('Loose_Clothing') else "No"
                        pdf.cell(0, 7, f"Wearing Non-Body-Fitting Clothes: {clothing_status_pdf}", ln=1)
                        if data_pdf.get('Loose_Clothing'):
                            pdf.set_font("Arial", "I", 8); pdf.multi_cell(0, 4, "Note: Thresholds for Kyphosis, Lordosis, and Scoliosis were adjusted accordingly.", 0, 'L', False); pdf.ln(1)
                        pdf.ln(5)

                        image_paths_to_delete_pdf = []
                        try:
                            if st.session_state.analysis_mode == "Multi-View Analysis (4 Views)" and st.session_state.all_landmark_images:
                                pdf.set_font("Arial", "B", 10); pdf.cell(0, 7, "Processed Views:", ln=1, align='C'); pdf.ln(2)
                                pdf.set_font("Arial", "", 8); img_display_width_pdf = 70; img_spacing_horizontal_pdf = 10
                                row_start_x_pdf = (pdf.w - (img_display_width_pdf * 2 + img_spacing_horizontal_pdf)) / 2
                                for i_row in range(0, len(VIEWS_SEQUENCE), 2):
                                    current_y_for_row_pdf = pdf.get_y(); max_h_this_row_pdf = 0
                                    for j_col in range(2):
                                        idx = i_row + j_col
                                        if idx < len(VIEWS_SEQUENCE):
                                            view_name_pdf = VIEWS_SEQUENCE[idx]; pil_image_pdf = st.session_state.all_landmark_images.get(view_name_pdf)
                                            if pil_image_pdf:
                                                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_f:
                                                    pil_image_pdf.save(tmp_f.name, format="JPEG"); image_paths_to_delete_pdf.append(tmp_f.name)
                                                    o_w, o_h = pil_image_pdf.size; asp = o_h/o_w if o_w > 0 else 1; img_h = img_display_width_pdf * asp
                                                    max_h_this_row_pdf = max(max_h_this_row_pdf, img_h)
                                                    if current_y_for_row_pdf + img_h + 10 > pdf.page_break_trigger: pdf.add_page(); current_y_for_row_pdf = pdf.get_y()
                                                    x_pos_img_pdf = row_start_x_pdf + j_col * (img_display_width_pdf + img_spacing_horizontal_pdf)
                                                    pdf.image(tmp_f.name, x=x_pos_img_pdf, y=current_y_for_row_pdf, w=img_display_width_pdf, h=img_h)
                                                    pdf.set_xy(x_pos_img_pdf, current_y_for_row_pdf + img_h + 1); pdf.multi_cell(img_display_width_pdf, 4, view_name_pdf, 0, 'C')
                                    if max_h_this_row_pdf > 0: pdf.set_y(current_y_for_row_pdf + max_h_this_row_pdf + 10)
                                    else: pdf.ln(5)
                                pdf.ln(5)
                            elif st.session_state.analysis_mode == "Single View Analysis" and st.session_state.get("landmark_image"):
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_f:
                                    st.session_state.landmark_image.save(tmp_f.name, format="JPEG"); image_paths_to_delete_pdf.append(tmp_f.name)
                                    page_width_pdf = pdf.w - pdf.l_margin - pdf.r_margin; max_h = 80
                                    w, h = st.session_state.landmark_image.size; asp = h/w if w>0 else 1
                                    img_w = page_width_pdf * 0.7; img_h = img_w * asp
                                    if img_h > max_h: img_h = max_h; img_w = img_h / asp if asp > 0 else max_h
                                    cy = pdf.get_y();
                                    if cy + img_h > pdf.page_break_trigger - 5: pdf.add_page(); cy = pdf.get_y()
                                    pdf.image(tmp_f.name, x=(pdf.w - img_w) / 2, y=cy, w=img_w, h=img_h)
                                    pdf.set_y(cy + img_h + 5); pdf.ln(3)
                        finally:
                            for path in image_paths_to_delete_pdf:
                                try: os.unlink(path)
                                except Exception: pass

                        pdf.set_font("Arial", "B", 11); pdf.cell(0, 7, "Detected Postural Issues:", ln=1)
                        pdf.set_font("Arial", "", 9); detected_cond_pdf = []
                        if abn_pdf:
                            for cond, pres in abn_pdf.items():
                                reason_str_pdf = get_abnormality_reason_string(cond, data_pdf, data_pdf) if pres else ""
                                pdf.cell(0, 5, f"- {cond}: {'Present' if pres else 'Not Present'} {reason_str_pdf}", ln=1)
                                if pres: detected_cond_pdf.append(cond)
                        else: pdf.cell(0,5, "- No abnormalities selected for detection or none found.", ln=1)
                        pdf.ln(2)

                        pdf.set_font("Arial", "B", 10); pdf.cell(0, 7, "Thresholds Used for this Analysis:", ln=1)
                        pdf.set_font("Arial", "", 8)
                        for k, v in DEFAULT_THRESHOLDS.items():
                             pdf.cell(0, 4, f"- {k.replace('_', ' ').title()}: {data_pdf.get(f'Thresh_{k}', v):.2f}", ln=1)
                        pdf.ln(3)

                        if gemini_sug_pdf:
                            pdf.ln(3); pdf.set_font("Arial", "B", 11); pdf.cell(0, 7, "✨ AI-Powered Personalized Suggestions:", ln=1)
                            pdf.set_font("Arial", "", 9)
                            for line_pdf in gemini_sug_pdf.split('\n'):
                                if pdf.get_y() > 270: pdf.add_page()
                                line_stripped = line_pdf.strip()
                                if line_stripped.startswith("### "): pdf.set_font("Arial", "B", 10); pdf.multi_cell(0, 5, line_stripped.replace("### ", ""), ln=1); pdf.set_font("Arial", "", 9)
                                elif line_stripped.startswith("#### "): pdf.set_font("Arial", "B", 9); pdf.multi_cell(0, 5, line_stripped.replace("#### ", ""), ln=1); pdf.set_font("Arial", "", 9)
                                elif line_stripped.startswith("**For"): pdf.set_font("Arial", "B", 9); pdf.multi_cell(0, 5, line_stripped.replace("**", ""), ln=1); pdf.set_font("Arial", "", 9)
                                elif line_stripped.startswith("* ") or line_stripped.startswith("- ") or (line_stripped and line_stripped[0].isdigit() and line_stripped[1:3] == ". "):
                                    pdf.set_x(pdf.l_margin + 5); pdf.multi_cell(0, 4, line_stripped, ln=1); pdf.set_x(pdf.l_margin)
                                else: pdf.multi_cell(0, 4, line_stripped, ln=1)
                            pdf.ln(1)

                        if pdf.get_y() > 275: pdf.add_page()
                        pdf.set_font("Arial", "I", 7)
                        disclaimer_text_pdf = "Disclaimer: This automated analysis, including AI-powered suggestions, is for informational purposes only and not a substitute for professional medical advice. Accuracy may be affected by factors like image quality and clothing. Consult a healthcare provider for health concerns or before starting new exercises."
                        pdf.multi_cell(0, 3.5, disclaimer_text_pdf, align="C")
                        pdf_output_data = pdf.output(dest='S')
                        pdf_bytes_out = pdf_output_data.encode('latin-1')
                        
                        st.download_button(label="📥 Download Report PDF",data=pdf_bytes_out,file_name=f"posture_report_{data_pdf.get('Student ID', 'report')}.pdf",mime="application/pdf",key="download_full_pdf_button")

                    except Exception as e:
                        st.error(f"Error during PDF generation: {e}\n{traceback.format_exc()}")
    
    st.markdown("---"); st.subheader("📊 View Locally Saved Records")
    if st.session_state.records:
        df_display = pd.DataFrame(st.session_state.records).copy()
        for col in df_display.columns:
            if 'Thresh_' in col:
                df_display[col] = df_display[col].round(3)
        st.dataframe(df_display)

    st.markdown("---"); st.subheader("☁️ Cloud Data Storage")
    if st.session_state.get('records'):
        if st.button("⬆️ Upload All Saved Records to Cloud", key="upload_to_azure_button"):
            with st.spinner("Connecting to database and uploading..."):
                conn_result = get_db_connection()
                if conn_result.get("type") == "success":
                    conn = conn_result["connection"]
                    records_for_sql = []
                    for rec in st.session_state.records:
                        rec_copy = rec.copy()
                        if "Gemini_Suggestions" in rec_copy:
                            del rec_copy["Gemini_Suggestions"]
                        records_for_sql.append(rec_copy)

                    upload_status = upload_records_to_sql(conn, records_for_sql)
                    st.session_state.cloud_upload_status = upload_status
                    try: conn.close()
                    except pyodbc.Error as e_close: 
                        if st.session_state.cloud_upload_status and st.session_state.cloud_upload_status.get("type") == "error":
                            st.session_state.cloud_upload_status["message"] += f" (Also, minor error closing DB connection: {e_close})"
                        else: st.session_state.cloud_upload_status = {"type": "warning", "message": f"Minor error closing DB connection: {e_close}"}
                    st.rerun()
                else: st.session_state.cloud_upload_status = conn_result; st.rerun()
    else: st.info("No local records saved locally to upload.")

# --- USER MANUAL PAGE ---
def user_manual_page():
    st.markdown("<h2 style='text-align: center; font-size: 24px; margin-bottom: 20px;'>FitNurture User Manual</h2>", unsafe_allow_html=True)
    
    st.header("1. Capturing an Image")
    st.write("""
        For the most accurate analysis, image quality is crucial. Please follow these guidelines:
        - **Good Lighting:** Ensure the subject is well-lit from the front. Avoid strong backlighting.
        - **Full Body Visibility:** The entire body, from head to feet, should be visible in the frame.
        - **Neutral Background:** Use a plain, non-cluttered background if possible.
        - **Form-Fitting Clothes:** For best results, the subject should wear body-fitting clothes. If not possible, check the "Subject is NOT wearing body-fitting clothes" box, which will adjust the calculations, though this may slightly reduce accuracy.
    """)

    st.subheader("Single View Analysis")
    st.write("""
        This mode is quick but provides less data. It's best for analyzing posture from the side.
        - **Instruction:** Have the subject stand in a relaxed side profile, with their left or right side facing the camera. They should look straight ahead.
    """)

    st.subheader("Multi-View Analysis (Recommended)")
    st.write("""
        This mode provides a comprehensive 360° analysis and is highly recommended for the most accurate results. You will be prompted to capture four images in the following order:
        1.  **Front View:** Subject faces the camera.
        2.  **Left Side View:** Subject turns so their left side faces the camera.
        3.  **Right Side View:** Subject turns so their right side faces the camera.
        4.  **Back View:** Subject faces away from the camera.
        
        The subject should stand in a relaxed, natural posture for each shot.
    """)

    st.header("2. How to Analyze and Read Results")
    st.write("""
        1.  **Fill Details:** Enter the subject's name and select their age and gender.
        2.  **Select Analysis Mode:** Choose "Single View" or "Multi-View".
        3.  **Provide Image(s):** Use the "Upload Image" or "Use Camera" option as instructed.
        4.  **Click Analyze:** Press the "Analyze Posture" button.
        5.  **Review Results:** The app will display a list of potential abnormalities.
            - **Present:** The analysis indicates a potential issue based on the measured metrics and set thresholds.
            - **Not Present:** The analysis did not detect this specific issue.
            - **Metric Details:** Next to a "Present" result, you will see the calculated metric versus the threshold that was used, like `(Sh-Hip Z: 0.21 > 0.15)`. This shows *why* the issue was flagged.
    """)

    st.header("3. Managing Results")
    st.write("""
    After a successful analysis, you have several options:
    - **Get AI Tips:** Click the `✨ Get AI Exercise & Lifestyle Tips` button to receive personalized suggestions from our AI based on the detected issues.
    - **Save Result Locally:** Click the `💾 Save Result Locally` button to add the complete analysis (including metrics, abnormalities, and any AI suggestions) to a table at the bottom of the page. This is useful for analyzing multiple students before uploading.
    - **Generate PDF Report:** Click the `📄 Generate PDF Report` button to create a comprehensive, downloadable PDF of the current analysis. This report is ideal for printing or sharing.
    - **Upload to Cloud:** After saving one or more records locally, the `⬆️ Upload All Saved Records to Cloud` button becomes available. Click this to securely save all the locally stored records to the central database.
    """)


    st.header("4. Advanced Settings (For Professional Use)")
    st.write("""
        The `Advanced: Adjust Detection Thresholds` section allows a professional, such as a physiotherapist, to fine-tune the sensitivity of the detection algorithm. **It is not recommended for general users to change these values.**
    """)
    
    st.subheader("Understanding the Thresholds")
    st.write("""
        The thresholds are based on normalized distances, angles, and ratios calculated from the person's skeletal landmarks in the image. "Normalized" means the values are relative to the person's own body size, not fixed pixels or inches.
    """)
    
    threshold_data = {
        "Abnormality": ["Kyphosis", "Lordosis", "Tech Neck (Angle)", "Tech Neck (Distance)", "Scoliosis", "Flat Feet", "Gait Abnormality", "Knock Knees", "Bow Legs"],
        "Metric Measures...": [
            "Forward slouch of shoulders relative to hips.",
            "Forward curve of lower back (hips relative to knees).",
            "Forward tilt angle of the head.",
            "Forward position of the ear relative to the shoulder.",
            "Height difference between the left and right shoulders.",
            "The height of the foot arch.",
            "How wide apart the ankles are when standing.",
            "Ratio of knee distance to ankle distance.",
            "Ratio of knee distance to ankle distance."
        ],
        "How the Input Box Works": [
            "**Stricter:** A higher value requires a more pronounced slouch to be flagged.",
            "**Stricter:** A higher value requires a more pronounced back curve to be flagged.",
            "**More Lenient:** A higher value allows for more head tilt before being flagged.",
            "**Stricter:** A higher value requires the head to be further forward to be flagged.",
            "**Stricter:** A higher value requires a larger height difference to be flagged.",
            "**Stricter:** A higher value requires a more defined arch to pass (less likely to be flat).",
            "**Stricter:** A higher value requires feet to be wider apart to be flagged.",
            "**More Lenient:** A higher ratio (closer to 1.0) allows knees to be closer relative to the ankles.",
            "**Stricter:** A higher ratio requires knees to be much wider than the ankles."
        ]
    }
    st.table(threshold_data)


# --- Main Application Logic (Router) ---
initialize_session_state()

# Sidebar Navigation
with st.sidebar:
    logo_paths = [os.path.join("assets", name) for name in ["logo.jpg", "logo.JPG", "logo.png", "logo.PNG"]]
    logo_found = False
    for logo_path in logo_paths:
        if os.path.exists(logo_path):
            st.image(logo_path)
            logo_found = True
            break
    
    st.title("Choose an Option")
    st.session_state.page = st.radio(
        "Menu", 
        ["Posture Analysis", "User Manual"],
        label_visibility="collapsed",
        index=["Posture Analysis", "User Manual"].index(st.session_state.page)
    )
    st.markdown("---")
    st.markdown(f"<div style='text-align: center; margin-top: 10px;'>© Copyright {datetime.now().year} FutureNurture</div>", unsafe_allow_html=True)


pose_static = load_pose_model()

if st.session_state.page == "Posture Analysis":
    main_app_page()
elif st.session_state.page == "User Manual":
    user_manual_page()
