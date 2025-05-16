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

# --- Page Config ---
st.set_page_config(
    page_title="FitNurture : Posture Detection",
    page_icon="🧘‍♀️",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
        .main > div {
            padding: 0rem 1rem 1rem 1rem;
        }
        /* Center logo container */
        .logo-container {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            margin: 0 auto;
            padding: 1rem 0;
        }
        /* Ensure image is centered */
        .logo-container > div {
            display: flex !important;
            justify-content: center !important;
            width: 100% !important;
        }
        .logo-container img {
            max-width: 200px !important;
            margin: 0 auto !important;
        }
        /* Center title text */
        .title-text {
            text-align: center;
            font-size: 24px;
            width: 100%;
            margin: 1rem 0;
        }
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .logo-container {
                padding: 0 10px;
            }
        }
        /* Copyright footer styling */
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
st.markdown('<div class="logo-container">', unsafe_allow_html=True)
# Check for logo in different possible formats
logo_paths = [
    os.path.join("assets", "logo.jpg"),
    os.path.join("assets", "logo.JPG"),
    os.path.join("assets", "logo.png"),
    os.path.join("assets", "logo.PNG")
]

logo_found = False
for logo_path in logo_paths:
    if os.path.exists(logo_path):
        try:
            st.image(logo_path, width=200, use_container_width=False)
            logo_found = True
            break
        except Exception as e:
            continue

if not logo_found:
    st.warning("Logo not found. Please ensure the logo file is in the assets directory.")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="title-text">FitNurture : Posture Detection</div>', unsafe_allow_html=True)

# --- Function Definitions ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_rad = np.arccos(cosine_angle)
    return np.degrees(angle_rad)

# --- App Config ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_static = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

if "records" not in st.session_state:
    st.session_state.records = []
if "current_entry" not in st.session_state:
    st.session_state.current_entry = {}
if "landmark_image" not in st.session_state:
    st.session_state.landmark_image = None
if "abnormalities" not in st.session_state:
    st.session_state.abnormalities = {}

# --- Input Form and Image Processing ---
child_name = st.text_input("Enter the Child's Name")
st.markdown("**Note:** If you're using a mobile device, the camera input is more reliable than file uploads.")
input_mode = st.radio("Choose Input Mode", ["Upload Image", "Use Camera (Recommended for Mobile)"])

image_data = None

# Clear the other input type when switching modes
if input_mode == "Upload Image":
    if "camera" in st.session_state:
        del st.session_state["camera"]
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image_data = Image.open(uploaded_file)
else:  # Camera mode
    if "upload" in st.session_state:
        del st.session_state["upload"]
    camera_data = st.camera_input("Take a picture using device")
    if camera_data:
        file_bytes = np.asarray(bytearray(camera_data.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_data = Image.fromarray(frame_rgb)

# Process image if available and name is provided
if image_data and child_name:
    img_np = np.array(image_data)
    if img_np.shape[-1] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    elif len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    results = pose_static.process(img_np)
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        img_with_landmarks = img_np.copy()
        mp_drawing.draw_landmarks(
            img_with_landmarks, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
        st.session_state.landmark_image = img_with_landmarks

        metrics = {
            "shoulder_z": lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z,
            "hip_z": lm[mp_pose.PoseLandmark.LEFT_HIP.value].z,
            "knee_z": lm[mp_pose.PoseLandmark.LEFT_KNEE.value].z,
            "tech_neck_angle": calculate_angle(
                [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y],
                [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                [lm[mp_pose.PoseLandmark.LEFT_EAR.value].x, lm[mp_pose.PoseLandmark.LEFT_EAR.value].y]),
            "shoulder_y_diff": abs(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y - lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y),
            "foot_z_diff": abs(lm[mp_pose.PoseLandmark.LEFT_HEEL.value].z - lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z),
            "ankle_x_diff": abs(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x - lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x),
            "knee_x_diff": abs(lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x - lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x)
        }

        abnormalities = {
            "Kyphosis": metrics["shoulder_z"] - metrics["hip_z"] > 0.15,
            "Lordosis": metrics["hip_z"] - metrics["knee_z"] > 0.1,
            "Tech Neck": metrics["tech_neck_angle"] < 15,
            "Scoliosis": metrics["shoulder_y_diff"] > 0.05,
            "Flat Feet": metrics["foot_z_diff"] < 0.05,
            "Gait Abnormalities": metrics["ankle_x_diff"] > 0.25,
            "Knock Knees": metrics["knee_x_diff"] < metrics["ankle_x_diff"] * 0.7,
            "Bow Legs": metrics["ankle_x_diff"] < metrics["knee_x_diff"] * 0.7
        }

        entry = {
            "Student Name": child_name,
            "Student ID": f"STU{random.randint(1000,9999)}",
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **abnormalities,
            **metrics
        }

        st.session_state.current_entry = entry
        st.session_state.abnormalities = abnormalities

# Add Save Button
if st.session_state.get("current_entry"):
    st.success("Analysis Complete")
    st.image(st.session_state.landmark_image, caption="Landmarked Image", use_column_width=True)
    st.write(f"### Abnormality Detection for {st.session_state.current_entry['Student Name']}:")
    for condition, present in st.session_state.abnormalities.items():
        st.markdown(f"- {condition}: {'Yes' if present else 'No'}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Result"):
            st.session_state.records.append(st.session_state.current_entry)
            st.success("Result saved successfully!")
    
    with col2:
        if st.button("Generate PDF Report"):
            data = st.session_state.current_entry
            stored_abnormalities = st.session_state.abnormalities
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Posture Analysis Report for {data['Student Name']}", ln=True, align='C')
            pdf.ln(10)
            pdf.cell(200, 10, txt="Detected Issues:", ln=True)
            for condition in stored_abnormalities:
                value = "Yes" if data.get(condition) else "No"
                pdf.cell(200, 10, txt=f"- {condition}: {value}", ln=True)
            pdf.ln(10)
            pdf.cell(200, 10, txt="Recommendations:", ln=True)
            if data.get("Kyphosis"):
                pdf.cell(200, 10, txt="- Strengthen upper back and stretch chest muscles.", ln=True)
            if data.get("Lordosis"):
                pdf.cell(200, 10, txt="- Strengthen core and hamstrings.", ln=True)
            if data.get("Tech Neck"):
                pdf.cell(200, 10, txt="- Reduce screen time, raise device height.", ln=True)
            if data.get("Scoliosis"):
                pdf.cell(200, 10, txt="- Seek professional orthopedic assessment.", ln=True)
            if data.get("Flat Feet"):
                pdf.cell(200, 10, txt="- Use orthotic insoles or see podiatrist.", ln=True)
            if data.get("Gait Abnormalities"):
                pdf.cell(200, 10, txt="- Consult physiotherapist for gait correction.", ln=True)
            if data.get("Knock Knees"):
                pdf.cell(200, 10, txt="- Strengthen outer thigh and hip muscles.", ln=True)
            if data.get("Bow Legs"):
                pdf.cell(200, 10, txt="- Use bracing if advised by specialist.", ln=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                pdf_output_path = tmp_file.name
                pdf.output(pdf_output_path)
            with open(pdf_output_path, "rb") as pdf_file:
                st.download_button("Download PDF Report", pdf_file, file_name=f"Posture_Report_{data['Student Name'].replace(' ', '_')}.pdf", mime="application/pdf")
            os.remove(pdf_output_path)

# --- View Data Table at End ---
st.markdown("---")
st.subheader("📊 View Collected Records")
if st.session_state.records:
    search_term = st.text_input("🔍 Search by Student Name or ID", key="search")
    df = pd.DataFrame(st.session_state.records)
    if search_term:
        df = df[df['Student Name'].str.contains(search_term, case=False) | df['Student ID'].str.contains(search_term, case=False)]
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv, file_name="posture_records.csv", mime="text/csv")
else:
    st.info("No records to display yet.")

# Add copyright footer
st.markdown("""
    <div class="copyright-footer">
        © Copyright 2025 FutureNurture | <a href="http://www.futurenurture.in" target="_blank">www.futurenurture.in</a>
    </div>
""", unsafe_allow_html=True)
