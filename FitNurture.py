import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from fpdf import FPDF
from PIL import Image
import tempfile
import os
import math
import time

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

def check_kyphosis(landmarks):
    try:
        shoulder_z = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z
        hip_z = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z
        return shoulder_z - hip_z > 0.15
    except (AttributeError, IndexError):
        return False

def check_lordosis(landmarks):
    try:
        hip_z = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z
        knee_z = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z
        return hip_z - knee_z > 0.1
    except (AttributeError, IndexError):
        return False

def check_tech_neck(landmarks, threshold_angle=15):
    try:
        ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        if not all([ear, shoulder, hip]):
            return False
        angle = calculate_angle([hip.x, hip.y], [shoulder.x, shoulder.y], [ear.x, ear.y])
        return angle < threshold_angle
    except (AttributeError, IndexError, ValueError):
        return False

def check_scoliosis(landmarks, threshold_y_diff=0.05):
    try:
        l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        return abs(l_shoulder.y - r_shoulder.y) > threshold_y_diff
    except (AttributeError, IndexError):
        return False

def check_flat_feet(landmarks, threshold_z_diff=0.05):
    try:
        heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value]
        toe = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
        return abs(heel.z - toe.z) < threshold_z_diff
    except (AttributeError, IndexError):
        return False

def check_gait_abnormalities(landmarks, threshold_x_diff=0.25):
    try:
        l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        return abs(l_ankle.x - r_ankle.x) > threshold_x_diff
    except (AttributeError, IndexError):
        return False

def check_knock_knees(landmarks):
    try:
        l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        knee_distance = abs(l_knee.x - r_knee.x)
        ankle_distance = abs(l_ankle.x - r_ankle.x)
        return knee_distance < ankle_distance * 0.7
    except (AttributeError, IndexError):
        return False

def check_bow_legs(landmarks):
    try:
        l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        knee_distance = abs(l_knee.x - r_knee.x)
        ankle_distance = abs(l_ankle.x - r_ankle.x)
        return ankle_distance < knee_distance * 0.7
    except (AttributeError, IndexError):
        return False

# --- Streamlit App Logic ---

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_static = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

st.set_page_config(page_title="FitNurture : Posture Detection")
st.title("FitNurture : Posture Detection")

child_name = st.text_input("Enter the Child's Name")
input_mode = st.radio("Choose Input Mode", ["Upload Image", "Use Webcam"])

image = None

if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)

elif input_mode == "Use Webcam":
    st.info("Use the button below to take a snapshot from your webcam.")
    img_data = st.camera_input("Take a picture using webcam")

    if img_data is not None:
        file_bytes = np.asarray(bytearray(img_data.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
        else:
            st.error("Could not decode the image. Please try capturing again.")

if image is not None:
    st.image(image, caption="Image for Analysis", use_column_width=True)
    img_np = np.array(image)

    if img_np.shape[-1] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    elif len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    results = pose_static.process(img_np)

    abnormalities = {
        "Kyphosis": False,
        "Lordosis": False,
        "Tech Neck": False,
        "Scoliosis": False,
        "Flat Feet": False,
        "Gait Abnormalities": False,
        "Knock Knees": False,
        "Bow Legs": False
    }

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        if check_kyphosis(lm):
            abnormalities["Kyphosis"] = True
        if check_lordosis(lm):
            abnormalities["Lordosis"] = True
        if check_tech_neck(lm):
            abnormalities["Tech Neck"] = True
        if check_scoliosis(lm):
            abnormalities["Scoliosis"] = True
        if check_flat_feet(lm):
            abnormalities["Flat Feet"] = True
        if check_gait_abnormalities(lm):
            abnormalities["Gait Abnormalities"] = True
        if check_knock_knees(lm):
            abnormalities["Knock Knees"] = True
        if check_bow_legs(lm):
            abnormalities["Bow Legs"] = True

        st.success("Analysis Complete")
        st.write(f"### Abnormality Detection for {child_name}:")
        for condition, present in abnormalities.items():
            icon = "✅" if present else "❌"
            st.markdown(f"- {condition}: {icon}")

        img_with_landmarks = img_np.copy()
        mp_drawing.draw_landmarks(
            img_with_landmarks,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
        st.image(img_with_landmarks, caption="Analyzed Image with Landmarks", channels="RGB", use_column_width=True)

        if st.button("Generate PDF Summary"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Posture Analysis Report for {child_name}", ln=True, align='C')
            pdf.ln(10)

            pdf.cell(200, 10, txt="Detected Issues:", ln=True)
            issues_found = False
            for condition, present in abnormalities.items():
                if present:
                    pdf.cell(200, 10, txt=f"- {condition}: Yes", ln=True)
                    issues_found = True

            if not issues_found:
                pdf.cell(200, 10, txt="No significant posture abnormalities detected.", ln=True)

            pdf.ln(10)
            pdf.cell(200, 10, txt="Recommendations:", ln=True)
            recommendations_added = False
            if abnormalities["Kyphosis"]:
                pdf.cell(200, 10, txt="- Strengthen upper back and stretch chest muscles.", ln=True)
                recommendations_added = True
            if abnormalities["Lordosis"]:
                pdf.cell(200, 10, txt="- Strengthen core and hamstrings.", ln=True)
                recommendations_added = True
            if abnormalities["Tech Neck"]:
                pdf.cell(200, 10, txt="- Reduce screen time, raise device height.", ln=True)
                recommendations_added = True
            if abnormalities["Scoliosis"]:
                pdf.cell(200, 10, txt="- Seek professional orthopedic assessment.", ln=True)
                recommendations_added = True
            if abnormalities["Flat Feet"]:
                pdf.cell(200, 10, txt="- Use orthotic insoles or see podiatrist.", ln=True)
                recommendations_added = True
            if abnormalities["Gait Abnormalities"]:
                pdf.cell(200, 10, txt="- Consult physiotherapist for gait correction.", ln=True)
                recommendations_added = True
            if abnormalities["Knock Knees"]:
                pdf.cell(200, 10, txt="- Strengthen outer thigh and hip muscles.", ln=True)
                recommendations_added = True
            if abnormalities["Bow Legs"]:
                pdf.cell(200, 10, txt="- Use bracing if advised by specialist.", ln=True)
                recommendations_added = True

            if not recommendations_added:
                pdf.cell(200, 10, txt="No specific recommendations based on detected issues.", ln=True)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                pdf_output_path = tmp_file.name
                pdf.output(pdf_output_path)

            with open(pdf_output_path, "rb") as pdf_file:
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_file,
                    file_name=f"Posture_Report_{child_name.replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )

            os.remove(pdf_output_path)
    else:
        st.warning("Could not detect pose landmarks in the image. Please ensure the person is clearly visible.")
