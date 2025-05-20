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
from fpdf import FPDF # Using fpdf as per user's working history
from PIL import Image
import tempfile
import os
import random
import pandas as pd
from datetime import datetime
import gc  # Import garbage collector
import pyodbc # Added for Azure SQL connection
import traceback # Added for PDF error debugging


# --- Memory Management Functions ---
def clear_image_memory():
    """Clear image-related data from session state"""
    if 'landmark_image' in st.session_state:
        del st.session_state['landmark_image']
    if 'current_entry' in st.session_state:
        del st.session_state['current_entry']
    if 'abnormalities' in st.session_state:
        del st.session_state['abnormalities']
    gc.collect()  # Force garbage collection

def optimize_image(image, max_size=800):
    """Resize image while maintaining aspect ratio"""
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image
    
    if img.size[0] == 0 or img.size[1] == 0: 
        return Image.new('RGB', (100,100), color = 'lightgray') 

    if max(img.size) > 0 : 
        ratio = max_size / max(img.size)
        if ratio < 1:  
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.LANCZOS) 
    
    return img


# Add this custom CSS after your existing page config
st.markdown("""
    <style>
    .camera-container {
        position: relative;
        width: fit-content;
        margin: auto;
    }
    /* ... (rest of CSS remains the same) ... */
    .copyright-footer {
        text-align: center;
        padding: 20px 0;
        margin-top: 30px;
        border-top: 1px solid #e5e5e5;
        color: #666;
        font-size: 14px;
    }
    .copyright-footer a {
        color: #666;
        text-decoration: none;
    }
    .copyright-footer a:hover {
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Logo and Title Section ---
st.markdown("<h2 style='text-align: center; font-size: 24px; margin-bottom: 20px;'>FitNurture : Posture Detection</h2>", unsafe_allow_html=True)

col1_logo, col2_logo, col3_logo = st.columns([1.2, 1, 1.2]) 
with col2_logo:
    logo_paths = [
        os.path.join("assets", "logo.jpg"), os.path.join("assets", "logo.JPG"),
        os.path.join("assets", "logo.png"), os.path.join("assets", "logo.PNG")
    ]
    logo_found = False
    for logo_path in logo_paths:
        if os.path.exists(logo_path):
            try:
                st.image(logo_path, width=225, use_container_width=True)
                logo_found = True; break
            except Exception as e:
                st.warning(f"Could not load logo {logo_path}: {e}"); continue
    if not logo_found:
        st.warning("Logo not found. Please ensure the logo file (logo.jpg, logo.png, etc.) is in the assets directory.")
st.markdown("<br>", unsafe_allow_html=True)

# --- Function Definitions ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    if not (a.shape == (2,) and b.shape == (2,) and c.shape == (2,)): return 0.0 
    ba, bc = a - b, c - b
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0: return 0.0
    cosine_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def add_landmark_labels(image, landmarks):
    img = image.copy(); h, w = img.shape[:2]
    landmark_labels = {
        mp_pose.PoseLandmark.NOSE: "Head", mp_pose.PoseLandmark.LEFT_SHOULDER: "L Shoulder", 
        mp_pose.PoseLandmark.RIGHT_SHOULDER: "R Shoulder", mp_pose.PoseLandmark.LEFT_ELBOW: "L Elbow", 
        mp_pose.PoseLandmark.RIGHT_ELBOW: "R Elbow", mp_pose.PoseLandmark.LEFT_HIP: "L Hip", 
        mp_pose.PoseLandmark.RIGHT_HIP: "R Hip", mp_pose.PoseLandmark.LEFT_KNEE: "L Knee", 
        mp_pose.PoseLandmark.RIGHT_KNEE: "R Knee", mp_pose.PoseLandmark.LEFT_ANKLE: "L Ankle", 
        mp_pose.PoseLandmark.RIGHT_ANKLE: "R Ankle"
    }
    for landmark_id, label in landmark_labels.items():
        if landmark_id.value < len(landmarks.landmark):
            landmark = landmarks.landmark[landmark_id.value]
            if landmark.visibility > 0.5:
                px, py = int(landmark.x * w), int(landmark.y * h)
                offset_x = -70 if px < w/2 else 70
                text_align = 'right' if px < w/2 else 'left'
                cv2.arrowedLine(img, (px + (offset_x//2), py), (px, py), (0,0,255), 1, tipLength=0.3)
                text_x = px + offset_x
                text_anchor = (text_x - 5 if text_align == 'right' else text_x + 5, py + 5)
                (t_w, t_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                bg_p1 = (text_anchor[0] - t_w - 4, text_anchor[1] - t_h - 4) if text_align == 'right' else (text_anchor[0] - 4, text_anchor[1] - t_h - 4)
                bg_p2 = (text_anchor[0] + 4, text_anchor[1] + 4) if text_align == 'right' else (text_anchor[0] + t_w + 4, text_anchor[1] + 4)
                cv2.rectangle(img, bg_p1, bg_p2, (255,255,255), -1)
                cv2.putText(img, label, text_anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    return img

# --- App Config ---
@st.cache_resource
def load_pose_model():
    return mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
mp_pose, mp_drawing, pose_static = mp.solutions.pose, mp.solutions.drawing_utils, load_pose_model()

for key in ['records', 'current_entry', 'landmark_image', 'abnormalities']:
    if key not in st.session_state: st.session_state[key] = [] if key == 'records' else {}

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

# --- Database Connection and Upload Functions ---
# ... (Database functions remain the same as previous correct version) ...
def get_db_connection():
    try:
        secrets = {k: st.secrets.get(k) for k in ["DB_DRIVER", "DB_SERVER", "DB_NAME", "DB_UID", "DB_PWD"]}
        missing_secrets = []
        if not secrets.get("DB_SERVER"): missing_secrets.append("DB_SERVER")
        if not secrets.get("DB_NAME"): missing_secrets.append("DB_NAME")
        if not secrets.get("DB_UID"): missing_secrets.append("DB_UID")
        if not secrets.get("DB_PWD"): missing_secrets.append("DB_PWD")

        if missing_secrets:
            st.error(f"⚠️ Database credentials missing in secrets: {', '.join(missing_secrets)}. Please configure them in Streamlit Cloud settings.")
            return None
        
        db_driver = secrets.get("DB_DRIVER") or "{ODBC Driver 17 for SQL Server}" 
        
        conn_str = f"DRIVER={db_driver};SERVER={secrets['DB_SERVER']};DATABASE={secrets['DB_NAME']};UID={secrets['DB_UID']};PWD={secrets['DB_PWD']};Encrypt=yes;TrustServerCertificate=no;ConnectionTimeout=30;"
        return pyodbc.connect(conn_str)
    except pyodbc.Error as ex:
        st.error(f"⚠️ Database Connection Error: {ex.args[0]}. Check configuration.")
        return None
    except Exception as e:
        st.error(f"⚠️ An unexpected error occurred during database connection: {e}")
        return None

def create_table_if_not_exists(conn):
    if conn is None: return
    cursor = conn.cursor()
    table_name = "PostureRecords"
    
    column_definitions_dict = {
        "Student_ID": "NVARCHAR(50) NOT NULL PRIMARY KEY",
        "Student_Name": "NVARCHAR(255) NULL",
        "Observation_Timestamp": "DATETIME2 NULL", 
        "UploadTimestamp": "DATETIME2 DEFAULT GETDATE() NULL" 
    }

    for key in POSTURE_RECOMMENDATIONS.keys():
        column_definitions_dict[key.replace(' ', '_').replace('-', '_')] = "BIT NULL"
    
    metrics_base_keys = ["shoulder_z", "hip_z", "knee_z", "neck_angle", "ear_shoulder_distance", "shoulder_y_diff", "foot_z_diff", "ankle_x_diff", "knee_x_diff"]
    for key in metrics_base_keys:
        column_definitions_dict[key] = "FLOAT NULL"

    cols_sql_definitions = [f"[{name}] {typedef}" for name, typedef in column_definitions_dict.items()]
    
    join_separator = ",\n        "
    create_table_query = f"""
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{table_name}' AND xtype='U')
    CREATE TABLE {table_name} (
        {join_separator.join(cols_sql_definitions)}
    );"""
    try:
        cursor.execute(create_table_query)
        conn.commit()
    except pyodbc.Error as e:
        st.error(f"⚠️ Error creating/checking table '{table_name}': {e}")
        conn.rollback()
    finally:
        cursor.close()

def upload_records_to_sql(conn, records_to_upload):
    if not records_to_upload or conn is None:
        if not records_to_upload: st.info("No new records to upload.")
        if conn is None: st.error("Database connection is not available for upload.")
        return

    create_table_if_not_exists(conn)
    cursor = conn.cursor()
    table_name = "PostureRecords"

    ordered_py_keys = ["Student ID", "Student Name", "Timestamp"] 
    sql_cols_for_std_fields = ["Student_ID", "Student_Name", "Observation_Timestamp"] 
    all_sql_cols = list(sql_cols_for_std_fields) 

    abnormality_sql_cols = [k.replace(' ', '_').replace('-', '_') for k in POSTURE_RECOMMENDATIONS.keys()]
    all_sql_cols.extend(abnormality_sql_cols)
    
    metrics_sql_cols = ["shoulder_z", "hip_z", "knee_z", "neck_angle", "ear_shoulder_distance", "shoulder_y_diff", "foot_z_diff", "ankle_x_diff", "knee_x_diff"]
    all_sql_cols.extend(metrics_sql_cols)

    insert_cols_str = ", ".join([f"[{col}]" for col in all_sql_cols])
    placeholders = ", ".join(["?" for _ in all_sql_cols])
    insert_sql = f"INSERT INTO {table_name} ({insert_cols_str}) VALUES ({placeholders})"

    update_set_clauses = [f"[{col}] = ?" for col in all_sql_cols if col != "Student_ID"]
    # Also update UploadTimestamp when updating a record
    update_sql = f"UPDATE {table_name} SET {', '.join(update_set_clauses)}, [UploadTimestamp] = GETDATE() WHERE [Student_ID] = ?"


    insert_count = 0
    update_count = 0
    error_count = 0

    for record in records_to_upload:
        values_for_insert = []
        values_for_insert.append(record.get("Student ID"))
        values_for_insert.append(record.get("Student Name"))
        values_for_insert.append(record.get("Timestamp")) 
        for key in POSTURE_RECOMMENDATIONS.keys(): 
            values_for_insert.append(bool(record.get(key)) if record.get(key) is not None else None)
        for key in metrics_sql_cols: 
            values_for_insert.append(float(record.get(key)) if record.get(key) is not None else None)
        
        student_id_val = record.get("Student ID")
        if not student_id_val:
            st.warning(f"Skipping record due to missing Student ID: {record.get('Student Name', 'N/A')}")
            error_count +=1
            continue

        try:
            cursor.execute(insert_sql, tuple(values_for_insert))
            insert_count += 1
        except pyodbc.IntegrityError as e:
            if '2627' in str(e) or 'PRIMARY KEY constraint' in str(e).upper() or 'unique constraint' in str(e).upper() : 
                values_for_update_set = values_for_insert[1:] # All values except Student_ID for SET clause
                values_for_update = tuple(values_for_update_set + [student_id_val]) # Add Student_ID for WHERE clause
                try:
                    cursor.execute(update_sql, values_for_update)
                    update_count += 1
                except pyodbc.Error as ue:
                    st.error(f"⚠️ Error updating record for Student ID '{student_id_val}': {ue}")
                    error_count += 1; conn.rollback()
            else: 
                st.error(f"⚠️ Database Integrity Error for Student ID '{student_id_val}': {e}")
                error_count += 1; conn.rollback()
        except pyodbc.Error as e: 
            st.error(f"⚠️ Database Error inserting record for Student ID '{student_id_val}': {e}")
            error_count += 1; conn.rollback()
        except Exception as ex_generic:
            st.error(f"⚠️ Unexpected error processing record for Student ID '{student_id_val}': {ex_generic}")
            error_count += 1; conn.rollback()

    if error_count == 0 and (insert_count > 0 or update_count > 0):
        try:
            conn.commit()
            if insert_count > 0: st.success(f"✅ Successfully inserted {insert_count} new record(s).")
            if update_count > 0: st.success(f"✅ Successfully updated {update_count} existing record(s).")
        except pyodbc.Error as e:
            conn.rollback()
            st.error(f"⚠️ Database commit error: {e}. Records were not saved.")
    elif error_count > 0:
        st.warning(f"{error_count} record(s) encountered errors. Any successful operations in this batch were rolled back.")
        conn.rollback() 
    
    cursor.close()

# --- Input Form and Image Processing ---
# ... (Input form and image processing remains the same) ...
container = st.container()
with container:
    col1_form, col2_form, col3_form = st.columns([1,2,1]) 
    with col2_form:
        child_name = st.text_input("Whats the Child's Name? (This is a mandatory field)", key="child_name")
        
        if not child_name and (st.session_state.get('camera_data') is not None or st.session_state.get('_file_uploader_key') is not None):
            st.error("⚠️ Please enter the child's name before proceeding")
            if 'camera_data' in st.session_state: del st.session_state['camera_data'] 
            if st.session_state.get('_file_uploader_key') is not None: st.session_state['_file_uploader_key'] = None 
            st.rerun()

        st.markdown("**Note:** If you're using a mobile device, the camera input is more reliable than file uploads.")
        st.markdown("### Select Abnormalities to Detect")
        
        if 'selected_abnormalities' not in st.session_state:
            st.session_state.selected_abnormalities = {k: True for k in POSTURE_RECOMMENDATIONS.keys()}
        
        select_all_current_value = all(st.session_state.selected_abnormalities.values())
        select_all = st.checkbox("Select All", value=select_all_current_value, key="select_all_checkbox")
        
        if st.session_state.select_all_checkbox != select_all_current_value : 
            st.session_state.selected_abnormalities = {k: st.session_state.select_all_checkbox for k in st.session_state.selected_abnormalities}
            st.rerun() 

        st.markdown("---")
        cols_abnorm = st.columns(2) 
        abnormality_list = list(st.session_state.selected_abnormalities.keys())
        half = len(abnormality_list) // 2
        
        with cols_abnorm[0]:
            for abnormality in abnormality_list[:half]:
                st.session_state.selected_abnormalities[abnormality] = st.checkbox(abnormality, value=st.session_state.selected_abnormalities[abnormality], key=f"cb_{abnormality}")
        with cols_abnorm[1]:
            for abnormality in abnormality_list[half:]:
                st.session_state.selected_abnormalities[abnormality] = st.checkbox(abnormality, value=st.session_state.selected_abnormalities[abnormality], key=f"cb_{abnormality}")
        st.markdown("---")

if 'previous_mode' not in st.session_state: st.session_state.previous_mode = None
input_mode = st.radio("Choose Input Mode", ["Upload Image", "Use Camera (Recommended for Mobile)"])
image_data = None

if st.session_state.previous_mode != input_mode:
    st.session_state.previous_mode = input_mode
    if "camera_data" in st.session_state: del st.session_state["camera_data"]
    if "_file_uploader_key" in st.session_state: del st.session_state["_file_uploader_key"]
    clear_image_memory(); st.rerun()

if input_mode == "Upload Image":
    if not child_name: st.error("⚠️ Please enter the child's name before uploading an image")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="_file_uploader_key")
    if uploaded_file:
        if child_name: image_data = optimize_image(Image.open(uploaded_file))
        else: st.warning("Please enter child's name first."); st.session_state['_file_uploader_key'] = None; st.rerun()
else:  # Camera mode
    if not child_name: st.error("⚠️ Please enter the child's name before using the camera")
    else:
        camera_data_val = st.camera_input("Take a picture using device", key="camera_data")
        if camera_data_val:
            if child_name:
                frame = cv2.imdecode(np.asarray(bytearray(camera_data_val.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None: image_data = optimize_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                else: st.error("Could not decode image from camera.")
            else: st.warning("Please enter child's name first."); st.session_state['camera_data'] = None; st.rerun()

if image_data and child_name: 
    img_np = np.array(image_data)
    if img_np.shape[-1] == 4: img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    elif len(img_np.shape) == 2: img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    message_placeholder = st.empty()
    try: results = pose_static.process(img_np)
    except Exception as e: st.error(f"Error processing image with MediaPipe: {str(e)}"); results = None
    
    if not results or not results.pose_landmarks:
        message_placeholder.error("⚠️ No person detected or pose landmarks found. Ensure full body visibility, good lighting, and clear image.")
    else:
        message_placeholder.empty(); lm = results.pose_landmarks.landmark
        img_bgr_with_landmarks = cv2.cvtColor(img_np.copy(), cv2.COLOR_RGB2BGR) 
        mp_drawing.draw_landmarks(img_bgr_with_landmarks, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2))
        img_bgr_with_landmarks = add_landmark_labels(img_bgr_with_landmarks, results.pose_landmarks) 
        st.session_state.landmark_image = Image.fromarray(cv2.cvtColor(img_bgr_with_landmarks, cv2.COLOR_BGR2RGB)) 

        def is_visible(le): return lm[le.value].visibility > 0.5 if le.value < len(lm) else False
        
        neck_angle = calculate_angle(*[[lm[p.value].x, lm[p.value].y] for p in [mp_pose.PoseLandmark.RIGHT_EAR, mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_SHOULDER]]) if all(is_visible(p) for p in [mp_pose.PoseLandmark.RIGHT_EAR, mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_SHOULDER]) else 0.0
        ear_shoulder_dist = abs(lm[mp_pose.PoseLandmark.RIGHT_EAR.value].x - lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) if all(is_visible(p) for p in [mp_pose.PoseLandmark.RIGHT_EAR, mp_pose.PoseLandmark.RIGHT_SHOULDER]) else 0.0

        metrics = {"neck_angle": neck_angle, "ear_shoulder_distance": ear_shoulder_dist}
        for k, v_lm_l, v_lm_r in [("shoulder_z", mp_pose.PoseLandmark.LEFT_SHOULDER, None), 
                                  ("hip_z", mp_pose.PoseLandmark.LEFT_HIP, None), 
                                  ("knee_z", mp_pose.PoseLandmark.LEFT_KNEE, None)]:
            metrics[k] = lm[v_lm_l.value].z if is_visible(v_lm_l) else None
        
        if all(is_visible(p) for p in [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]):
            metrics["shoulder_y_diff"] = abs(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y - lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
        else: metrics["shoulder_y_diff"] = None

        if all(is_visible(p) for p in [mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.LEFT_FOOT_INDEX]):
            metrics["foot_z_diff"] = abs(lm[mp_pose.PoseLandmark.LEFT_HEEL.value].z - lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z)
        else: metrics["foot_z_diff"] = None
            
        for k, v_lm_l, v_lm_r in [("ankle_x_diff", mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE), 
                                  ("knee_x_diff", mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE)]:
            metrics[k] = abs(lm[v_lm_l.value].x - lm[v_lm_r.value].x) if all(is_visible(p) for p in [v_lm_l, v_lm_r]) else None

        current_abnormalities = {k: False for k,v in st.session_state.selected_abnormalities.items() if v}
        def check_metric(m): return metrics.get(m) is not None

        if "Kyphosis" in current_abnormalities and all(check_metric(m) for m in ["shoulder_z", "hip_z"]): current_abnormalities["Kyphosis"] = metrics["shoulder_z"] - metrics["hip_z"] > 0.15
        if "Lordosis" in current_abnormalities and all(check_metric(m) for m in ["hip_z", "knee_z"]): current_abnormalities["Lordosis"] = metrics["hip_z"] - metrics["knee_z"] > 0.1
        if "Tech Neck" in current_abnormalities: current_abnormalities["Tech Neck"] = (metrics["neck_angle"] > 45 and metrics["ear_shoulder_distance"] > 0.15)
        if "Scoliosis" in current_abnormalities and check_metric("shoulder_y_diff"): current_abnormalities["Scoliosis"] = metrics["shoulder_y_diff"] > 0.05
        if "Flat Feet" in current_abnormalities and check_metric("foot_z_diff"): current_abnormalities["Flat Feet"] = metrics["foot_z_diff"] < 0.05
        if "Gait Abnormalities" in current_abnormalities and check_metric("ankle_x_diff"): current_abnormalities["Gait Abnormalities"] = metrics["ankle_x_diff"] > 0.25
        if "Knock Knees" in current_abnormalities and all(check_metric(m) for m in ["knee_x_diff", "ankle_x_diff"]) and metrics["ankle_x_diff"] != 0: current_abnormalities["Knock Knees"] = metrics["knee_x_diff"] < metrics["ankle_x_diff"] * 0.7
        if "Bow Legs" in current_abnormalities and all(check_metric(m) for m in ["knee_x_diff", "ankle_x_diff"]) and metrics["knee_x_diff"] != 0: current_abnormalities["Bow Legs"] = metrics["ankle_x_diff"] < metrics["knee_x_diff"] * 0.7
        
        student_id = st.session_state.get("current_student_id")
        if not student_id: 
            student_id = f"FN-{random.randint(1000,9999)}"
            st.session_state.current_student_id = student_id 


        entry = {"Student ID": student_id, "Student Name": child_name, "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **current_abnormalities, **metrics}
        for ab_key in POSTURE_RECOMMENDATIONS.keys():
            if ab_key not in entry: entry[ab_key] = False 

        st.session_state.current_entry = entry
        st.session_state.abnormalities = current_abnormalities 

        visible_metrics_count = sum(1 for k in ["shoulder_z", "hip_z", "knee_z", "shoulder_y_diff", "foot_z_diff", "ankle_x_diff", "knee_x_diff"] if metrics[k] is not None)
        if visible_metrics_count < 7: st.warning(f"⚠️ Partial visibility ({visible_metrics_count}/7 key metrics calculated). Accuracy may be affected.")

if st.session_state.get("current_entry") and st.session_state.get("landmark_image"):
    st.success("Analysis Complete!")
    st.image(st.session_state.landmark_image, caption="Landmarked Image", use_container_width=True)
    display_abnormalities = st.session_state.get("abnormalities", {})
    if display_abnormalities:
        st.write(f"### Abnormality Detection for {st.session_state.current_entry['Student Name']} (ID: {st.session_state.current_entry['Student ID']}):")
        for cond, pres in display_abnormalities.items(): st.markdown(f"- {cond}: {'**Present**' if pres else 'Not Present'}")
    else: st.info("No abnormalities selected/detected.")

    col1_actions, col2_actions = st.columns(2) 
    with col1_actions:
        if st.button("💾 Save Result Locally", key="save_result_button"):
            st.session_state.records.append(st.session_state.current_entry)
            st.success(f"Result for {st.session_state.current_entry['Student ID']} saved locally!")
            if 'current_student_id' in st.session_state:
                del st.session_state['current_student_id']

    with col2_actions:
        if st.button("📄 Generate PDF Report", key="generate_pdf_button"):
            try:
                data_pdf = st.session_state.current_entry
                abn_pdf = st.session_state.abnormalities
                
                pdf = FPDF()
                pdf.add_page()
                pdf.set_auto_page_break(auto=True, margin=15) 

                # --- Header ---
                pdf.set_font("Arial", "B", 16)
                pdf.cell(0, 10, "FitNurture Posture Analysis Report", ln=1, align="C")
                
                pdf.set_font("Arial", "", 9) 
                pdf.cell(0, 7, "www.futurenurture.in", ln=1, align="C", link="http://www.futurenurture.in")
                pdf.ln(5) 

                # --- Logo ---
                current_y_logo = pdf.get_y()
                logo_pdf_path = next((p for p in logo_paths if os.path.exists(p)), None)
                if logo_pdf_path:
                    logo_width_pdf = 35 
                    logo_height_pdf = 17.5 
                    if current_y_logo + logo_height_pdf > pdf.page_break_trigger - 5: 
                        pdf.add_page()
                        current_y_logo = pdf.get_y() 
                    pdf.image(logo_pdf_path, x=(210-logo_width_pdf)/2, y=current_y_logo, w=logo_width_pdf) 
                    pdf.set_y(current_y_logo + logo_height_pdf + 5) 
                
                # --- Student Details ---
                pdf.set_font("Arial", "B", 12)
                details = {
                    "Student Name": data_pdf.get('Student Name'), 
                    "Student ID": data_pdf.get('Student ID'), 
                    "Timestamp": data_pdf.get('Timestamp')
                }
                for k_pdf, v_pdf in details.items():
                    pdf.cell(0, 7, f"{k_pdf}: {v_pdf or 'N/A'}", ln=1)
                pdf.ln(5)

                # --- Landmarked Image ---
                if st.session_state.get("landmark_image"):
                    pil_image = st.session_state.landmark_image 
                    
                    page_width = pdf.w - pdf.l_margin - pdf.r_margin 
                    max_image_height_pdf = 80 
                    
                    original_w_px, original_h_px = pil_image.size
                    aspect_ratio = original_h_px / original_w_px if original_w_px > 0 else 1

                    img_w_pdf = page_width * 0.70 
                    img_h_pdf = img_w_pdf * aspect_ratio

                    if img_h_pdf > max_image_height_pdf:
                        img_h_pdf = max_image_height_pdf
                        img_w_pdf = img_h_pdf / aspect_ratio if aspect_ratio > 0 else max_image_height_pdf

                    current_y_img = pdf.get_y()
                    if current_y_img + img_h_pdf > pdf.page_break_trigger - 5: 
                        pdf.add_page()
                        current_y_img = pdf.get_y()

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img_f:
                        pil_image.save(tmp_img_f.name, format="JPEG")
                        img_x_pos = (pdf.w - img_w_pdf) / 2 
                        pdf.image(tmp_img_f.name, x=img_x_pos, y=current_y_img, w=img_w_pdf, h=img_h_pdf)
                    os.unlink(tmp_img_f.name)
                    pdf.set_y(current_y_img + img_h_pdf + 5) 
                pdf.ln(3)

                # --- Results ---
                pdf.set_font("Arial", "B", 11) 
                pdf.cell(0, 7, "Detected Postural Issues:", ln=1) 
                pdf.set_font("Arial", "", 9) 
                detected_cond_pdf = []
                if abn_pdf: 
                    for cond, pres in abn_pdf.items():
                        pdf.cell(0, 5, f"- {cond}: {'Present' if pres else 'Not Present'}", ln=1) 
                        if pres: detected_cond_pdf.append(cond)
                else:
                    pdf.cell(0,5, "- No abnormalities selected for detection or none found.", ln=1)
                pdf.ln(2) 

                # --- Recommendations ---
                if detected_cond_pdf:
                    pdf.set_font("Arial", "B", 11) 
                    pdf.cell(0, 7, "General Recommendations:", ln=1) 
                    pdf.set_font("Arial", "", 9) 
                    available_width = pdf.w - pdf.l_margin - pdf.r_margin - 5 
                    for cond in detected_cond_pdf:
                        if cond in POSTURE_RECOMMENDATIONS:
                            pdf.set_font("Arial", "B", 9) 
                            pdf.multi_cell(available_width, 5, f"For {cond}:") 
                            pdf.set_font("Arial", "", 9)
                            for rec_item in POSTURE_RECOMMENDATIONS[cond]:
                                clean_rec_item = rec_item.strip() 
                                pdf.set_x(pdf.l_margin + 5) 
                                pdf.multi_cell(available_width - 5, 4, clean_rec_item) 
                            pdf.ln(1) 
                pdf.ln(2) 
                
                # --- Disclaimer (Moved to main body, before potential page break for footer) ---
                disclaimer_height_estimate = 15 
                if pdf.get_y() + disclaimer_height_estimate > pdf.page_break_trigger -5: 
                    pdf.add_page()
                
                pdf.set_font("Arial", "I", 7) 
                disclaimer_text = "Disclaimer: This automated analysis is for informational purposes only and not a substitute for professional medical advice. Consult a healthcare provider for health concerns."
                pdf.multi_cell(0, 3.5, disclaimer_text, align="C") 
                
                # Output PDF
                pdf_output_data = pdf.output(dest='S') 
                # The error indicates pdf_output_data is bytearray, so convert to bytes
                pdf_bytes_out = bytes(pdf_output_data)

                if not pdf_bytes_out: 
                    st.error("Critical PDF Error: Output from FPDF is empty or None. No PDF data generated.")
                else:
                    st.success("PDF Report Generated!")
                    st.download_button(
                        label="📥 Download Report PDF",
                        data=pdf_bytes_out,
                        file_name=f"posture_report_{data_pdf.get('Student ID', 'report') if data_pdf else 'report'}.pdf",
                        mime="application/pdf",
                        key="download_full_pdf_button"
                    )
            except Exception as e: 
                st.error(f"Error during PDF generation process: {e}\n{traceback.format_exc()}")

# --- View Data Table and Cloud Upload ---
# ... (rest of the script remains the same) ...
st.markdown("---"); st.subheader("📊 View Locally Saved Records")
if st.session_state.records:
    records_per_page = 10
    if 'current_page_local_records' not in st.session_state: st.session_state.current_page_local_records = 0
    total_records = len(st.session_state.records)
    total_pages = (total_records + records_per_page - 1) // records_per_page if total_records > 0 else 0

    if total_pages > 0:
        st.session_state.current_page_local_records = st.selectbox("Select Page", options=range(total_pages), format_func=lambda x: f"Page {x+1}", index=st.session_state.current_page_local_records, key="local_records_page_selector")
    
    search_term = st.text_input("🔍 Search by Student Name or ID in local records", key="search_local")
    display_records = st.session_state.records
    if search_term:
        display_records = [r for r in st.session_state.records if (search_term.lower() in r.get('Student Name', '').lower() or search_term.lower() in r.get('Student ID', '').lower())]
        total_records = len(display_records) 
        total_pages = (total_records + records_per_page - 1) // records_per_page if total_records > 0 else 0
        if st.session_state.current_page_local_records >= total_pages and total_pages > 0 : st.session_state.current_page_local_records = total_pages -1
        elif total_pages == 0 : st.session_state.current_page_local_records = 0


    start_idx = st.session_state.current_page_local_records * records_per_page
    end_idx = start_idx + records_per_page
    df_display = pd.DataFrame(display_records[start_idx:end_idx])

    if not df_display.empty: st.dataframe(df_display, use_container_width=True)
    elif search_term: st.info("No local records match your search criteria.")
    else: st.info("No local records to display.")

    @st.cache_data 
    def convert_all_to_csv(records_list):
        if not records_list: return b""
        return pd.DataFrame(records_list).to_csv(index=False).encode("utf-8")
    if st.session_state.records: 
        csv_all = convert_all_to_csv(st.session_state.records) 
        st.download_button("📥 Download All Local Records (CSV)", data=csv_all, file_name="all_posture_records.csv", mime="text/csv", key="download_all_csv")
else: st.info("No records saved locally yet.")

st.markdown("---"); st.subheader("☁️ Cloud Storage (Azure SQL)")
if st.session_state.get('records'):
    if st.button("⬆️ Upload All Saved Local Records to Azure SQL", key="upload_to_azure_button"):
        with st.spinner("Connecting to database and uploading records..."):
            conn = get_db_connection() 
            if conn:
                upload_records_to_sql(conn, list(st.session_state.records))
                try: conn.close()
                except pyodbc.Error as e: st.warning(f"Minor error closing DB connection: {e}") 
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

