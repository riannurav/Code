import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from fpdf import FPDF
from PIL import Image
import tempfile
import os

# Mediapipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_static = mp_pose.Pose(static_image_mode=True)
pose_live = mp_pose.Pose(static_image_mode=False)

def check_kyphosis(landmarks):
    shoulder_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y +
                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2
    hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y +
             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
    return shoulder_y - hip_y > 0.1

def check_lordosis(landmarks):
    hip_x = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x +
             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2
    knee_x = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x +
              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x) / 2
    return abs(hip_x - knee_x) > 0.03

def check_tech_neck(landmarks):
    nose_x = landmarks[mp_pose.PoseLandmark.NOSE.value].x
    shoulder_x = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x +
                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2
    return abs(nose_x - shoulder_x) > 0.08

def check_scoliosis(landmarks):
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
    lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x
    rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x
    return abs((ls - rs) - (lh - rh)) > 0.05

def detect_posture_static(image):
    results_dict = {}
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose_static.process(image_rgb)
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = result.pose_landmarks.landmark
        results_dict = {
            "Kyphosis": check_kyphosis(lm),
            "Lordosis": check_lordosis(lm),
            "Tech Neck": check_tech_neck(lm),
            "Scoliosis": check_scoliosis(lm)
        }
    return image, results_dict

def detect_posture_live():
    cap = cv2.VideoCapture(0)
    result_dict = {}
    frame = None
    stframe = st.empty()
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        frame = image.copy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = pose_live.process(image_rgb)
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = result.pose_landmarks.landmark
            result_dict = {
                "Kyphosis": check_kyphosis(lm),
                "Lordosis": check_lordosis(lm),
                "Tech Neck": check_tech_neck(lm),
                "Scoliosis": check_scoliosis(lm)
            }
        stframe.image(image[:, :, ::-1], channels='RGB')
        if st.button("Capture and Analyze"):
            break
    cap.release()
    return frame, result_dict

def generate_pdf(results, save_path):
    recs = {
        "Kyphosis": "Encourage yoga, swimming, and upright sitting. Limit screen time.",
        "Lordosis": "Strengthen core. Consult a physiotherapist.",
        "Tech Neck": "Raise screen to eye level. Take regular breaks.",
        "Scoliosis": "Consult a pediatric orthopedic. Monitor posture regularly."
    }
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Posture Screening Summary Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    pdf.cell(0, 10, "Detected Postural Abnormalities:", ln=True)
    for k, v in results.items():
        pdf.cell(0, 8, f"- {k}: {'Detected' if v else 'Not Detected'}", ln=True)
    pdf.ln(8)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Recommendations:", ln=True)
    pdf.set_font("Arial", "", 11)
    for k, v in results.items():
        if v:
            pdf.multi_cell(0, 8, f"{k}: {recs[k]}")
            pdf.ln(1)
    pdf.output(save_path)
    return save_path

def main():
    st.title("🧕 Posture Disorder Detector (Kids)")
    st.subheader("Detect Kyphosis, Lordosis, Tech Neck, Scoliosis")

    input_mode = st.radio("Choose Input Mode", ["Upload Image", "Use Webcam"])

    result_dict = {}
    final_img = None

    if input_mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            final_img, result_dict = detect_posture_static(image)
    elif input_mode == "Use Webcam":
        st.info("Start webcam and press 'Capture and Analyze' to detect")
        final_img, result_dict = detect_posture_live()

    if result_dict:
        st.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), caption="Posture Analysis", use_column_width=True)
        for k, v in result_dict.items():
            st.markdown(f"**{k}**: {'✅ Normal' if not v else '⚠️ Detected'}")

        with st.expander("Export PDF Report"):
            file_path = st.text_input("Enter file name (e.g., output.pdf):", "Posture_Summary_Report.pdf")
            if file_path:
                full_path = os.path.join(tempfile.gettempdir(), file_path)
                pdf_path = generate_pdf(result_dict, full_path)
                st.success("PDF Report Generated")
                with open(pdf_path, "rb") as file:
                    st.download_button("Download PDF Report", file, file_name=file_path)

if __name__ == '__main__':
    main()