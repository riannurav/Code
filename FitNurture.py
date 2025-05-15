import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from fpdf import FPDF
from PIL import Image
import tempfile
import os
import math # Import math for calculating angles if needed later
import time # Import time for potential delays

# --- Function Definitions ---

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a) # First point
    b = np.array(b) # Mid point
    c = np.array(c) # End point

    # Calculate vectors
    ba = a - b
    bc = c - b

    # Calculate cosine of the angle using dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # Ensure the value is within the valid range for arccos
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_rad = np.arccos(cosine_angle)

    # Convert radians to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def check_kyphosis(landmarks):
    """
    Checks for potential kyphosis based on shoulder and hip Z-axis position.
    Note: This is a simplified check and may not be accurate without a proper sagittal view.
    """
    # Using Z-axis for a very rough estimate from a frontal view
    # A more accurate check requires a side view and angle calculations
    try:
        shoulder_z = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z
        hip_z = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z
        # If shoulder is significantly behind the hip (larger Z value), could indicate kyphosis
        return shoulder_z - hip_z > 0.15 # Threshold might need adjustment
    except (AttributeError, IndexError):
        return False # Handle cases where landmarks are not detected or missing attributes

def check_lordosis(landmarks):
    """
    Checks for potential lordosis based on hip and knee Z-axis position.
    Note: This is a simplified check and may not be accurate without a proper sagittal view.
    """
    # Using Z-axis for a very rough estimate from a frontal view
    # A more accurate check requires a side view and angle calculations
    try:
        hip_z = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z
        knee_z = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z
        # If hip is significantly behind the knee (larger Z value), could indicate lordosis
        return hip_z - knee_z > 0.1 # Threshold might need adjustment
    except (AttributeError, IndexError):
        return False # Handle cases where landmarks are not detected or missing attributes


def check_tech_neck(landmarks, threshold_angle=15):
    """
    Checks for potential tech neck based on the angle between ear, shoulder, and hip.
    A smaller angle might indicate the head is forward.
    Requires LEFT_EAR, LEFT_SHOULDER, and LEFT_HIP landmarks.
    """
    try:
        ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

        # Calculate the angle formed by hip, shoulder, and ear
        # Ensure points are valid before calculating
        if not all([ear, shoulder, hip]):
             return False
        angle = calculate_angle(np.array([hip.x, hip.y]), np.array([shoulder.x, shoulder.y]), np.array([ear.x, ear.y]))

        # A smaller angle than the threshold might indicate tech neck
        # The threshold value might need tuning based on typical posture angles
        return angle < threshold_angle
    except (AttributeError, IndexError, ValueError): # Added ValueError for potential issues in calculate_angle
        return False # Handle cases where landmarks are not detected or calculation fails


def check_scoliosis(landmarks, threshold_y_diff=0.05):
    """
    Checks for potential scoliosis based on the vertical difference between shoulders.
    Requires LEFT_SHOULDER and RIGHT_SHOULDER landmarks.
    """
    try:
        l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        # Check if the vertical position (y-coordinate) of shoulders is significantly different
        return abs(l_shoulder.y - r_shoulder.y) > threshold_y_diff # Threshold might need adjustment
    except (AttributeError, IndexError):
        return False # Handle cases where landmarks are not detected

def check_flat_feet(landmarks, threshold_z_diff=0.05):
    """
    Checks for potential flat feet based on the Z-axis difference between heel and toe.
    Note: This is a simplified check from a frontal view and may not be accurate.
    Requires LEFT_HEEL and LEFT_FOOT_INDEX landmarks.
    """
    try:
        heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value]
        toe = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
        # A smaller Z-axis difference might indicate less arch height
        return abs(heel.z - toe.z) < threshold_z_diff # Threshold might need adjustment
    except (AttributeError, IndexError):
        return False # Handle cases where landmarks are not detected

def check_gait_abnormalities(landmarks, threshold_x_diff=0.25):
    """
    Checks for potential gait abnormalities based on the horizontal distance between ankles.
    Note: This is a simplified check from a static image and is more indicative of stance width.
    Requires LEFT_ANKLE and RIGHT_ANKLE landmarks.
    """
    try:
        l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        # A large horizontal distance might indicate a wide stance
        return abs(l_ankle.x - r_ankle.x) > threshold_x_diff # Threshold might need adjustment
    except (AttributeError, IndexError):
        return False # Handle cases where landmarks are not detected

def check_knock_knees(landmarks):
    """
    Checks for potential knock knees based on the horizontal distance between knees and ankles.
    Requires LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, and RIGHT_ANKLE landmarks.
    """
    try:
        l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # Calculate horizontal distances
        knee_distance = abs(l_knee.x - r_knee.x)
        ankle_distance = abs(l_ankle.x - r_ankle.x)

        # If knees are significantly closer than ankles, it might indicate knock knees
        return knee_distance < ankle_distance * 0.7 # Ratio might need adjustment
    except (AttributeError, IndexError):
        return False # Handle cases where landmarks are not detected

def check_bow_legs(landmarks):
    """
    Checks for potential bow legs based on the horizontal distance between knees and ankles.
    Requires LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, and RIGHT_ANKLE landmarks.
    """
    try:
        l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # Calculate horizontal distances
        knee_distance = abs(l_knee.x - r_knee.x)
        ankle_distance = abs(l_ankle.x - r_ankle.x)

        # If ankles are significantly closer than knees, it might indicate bow legs
        return ankle_distance < knee_distance * 0.7 # Ratio might need adjustment
    except (AttributeError, IndexError):
        return False # Handle cases where landmarks are not detected

# --- Streamlit App Logic ---

# Mediapipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
# Using static_image_mode=True for image upload, False for webcam (though webcam capture is static here)
pose_static = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose_live = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)


st.set_page_config(page_title="FitNurture : Posture Detection")
st.title("FitNurture : Posture Detection")

child_name = st.text_input("Enter the Child's Name")
input_mode = st.radio("Choose Input Mode", ["Upload Image", "Use Webcam"])

# Initialize session state for webcam capture and capture trigger
if 'is_capturing' not in st.session_state:
    st.session_state.is_capturing = False
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'capture_triggered' not in st.session_state:
    st.session_state.capture_triggered = False


image = None # This variable will hold the final image for analysis

if input_mode == "Upload Image":
    # Reset webcam state if switching mode
    st.session_state.is_capturing = False
    st.session_state.captured_image = None
    st.session_state.capture_triggered = False # Reset capture trigger

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="upload")
    if uploaded_file:
        image = Image.open(uploaded_file)

elif input_mode == "Use Webcam":
    frame_placeholder = st.empty()

    if not st.session_state.is_capturing and not st.session_state.capture_triggered:
        # Show button to start webcam if not already capturing or capture triggered
        if st.button("Start Webcam", key="start_webcam"):
            st.session_state.is_capturing = True
            st.session_state.captured_image = None # Clear previous capture
            st.session_state.capture_triggered = False # Ensure capture trigger is false
            st.rerun() # Rerun the script to start capture loop
    elif st.session_state.is_capturing:
        # If capturing, show the live feed and capture button
        st.write("Live Webcam Feed:")
        cap = cv2.VideoCapture(0) # Start video capture

        if not cap.isOpened():
            st.error("Cannot access webcam. Please ensure it's connected and not in use by another application.")
            st.session_state.is_capturing = False # Reset state
            st.session_state.capture_triggered = False
            st.rerun() # Rerun to show the start button again
        else:
            # Place the Capture button outside the loop
            capture_button = st.button("Capture and Analyze", key="webcam_capture_analyze")

            if capture_button:
                 st.session_state.capture_triggered = True # Set trigger
                 st.session_state.is_capturing = False # Stop capturing
                 # No rerun here, the loop will finish and then the code below will run

            while st.session_state.is_capturing:
                ret, frame = cap.read() # Read a frame

                if ret:
                    # Convert frame to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Process the frame for landmarks
                    results_live = pose_live.process(frame_rgb)

                    # Draw landmarks on the frame
                    if results_live.pose_landmarks:
                         mp_drawing.draw_landmarks(
                             frame_rgb,
                             results_live.pose_landmarks,
                             mp_pose.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                         )

                    # Display the live frame with landmarks
                    frame_placeholder.image(frame_rgb, caption="Live Feed with Landmarks", channels="RGB", width=600) # Adjust width as needed
                    st.session_state.current_frame = frame_rgb # Store current frame

                else:
                    st.warning("Could not read frame from webcam.")
                    st.session_state.is_capturing = False # Stop capturing
                    st.session_state.capture_triggered = False
                    break # Exit the loop

            # After the loop stops (either by button or error)
            cap.release() # Release the camera

            # If capture was triggered, process the last frame
            if st.session_state.capture_triggered:
                 if 'current_frame' in st.session_state and st.session_state.current_frame is not None:
                      st.session_state.captured_image = st.session_state.current_frame
                 st.session_state.capture_triggered = False # Reset trigger
                 st.rerun() # Rerun to process the captured image

# Use the captured image from session state if available and input mode is webcam
if input_mode == "Use Webcam" and st.session_state.captured_image is not None:
    image = Image.fromarray(st.session_state.captured_image)
    # Clear the captured image from session state after processing to avoid re-analysis
    # st.session_state.captured_image = None # Keep it to display the captured image below


# --- Posture Analysis and PDF Generation ---
if image is not None: # Check if an image is available for analysis
    # Display the image being analyzed (either uploaded or captured)
    st.image(image, caption="Image for Analysis", use_column_width=True)

    img_np = np.array(image)

    # Convert image to RGB for Mediapipe if it's not already
    if img_np.shape[-1] == 4: # Check if image has alpha channel
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    elif len(img_np.shape) == 2: # Check if image is grayscale
         img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)


    results = pose_static.process(img_np) # Use pose_static for the uploaded/captured image

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

        # Perform checks using the defined functions
        if check_kyphosis(lm):
            abnormalities["Kyphosis"] = True
        if check_lordosis(lm):
            abnormalities["Lordosis"] = True
        # Using the angle-based check for tech neck
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

        # --- Draw Landmarks on the Analyzed Image ---
        # Create a copy to draw on so the original uploaded image isn't modified
        img_with_landmarks = img_np.copy()
        mp_drawing.draw_landmarks(
            img_with_landmarks,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
        st.image(img_with_landmarks, caption="Analyzed Image with Landmarks", channels="RGB", use_column_width=True)


        # --- PDF Generation ---
        if st.button("Generate PDF Summary", key="generate_pdf"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Posture Analysis Report for {child_name}", ln=True, align='C')
            pdf.ln(10)

            pdf.cell(200, 10, txt="Detected Issues:", ln=True)
            issues_found = False
            for condition, present in abnormalities.items():
                if present:
                    # Use simple ASCII characters for PDF
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


            # Using tempfile to handle PDF in memory for download
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                pdf_output_path = tmp_file.name
                pdf.output(pdf_output_path)

            # Provide a download button for the PDF
            with open(pdf_output_path, "rb") as pdf_file:
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_file,
                    file_name=f"Posture_Report_{child_name.replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )

            # Clean up the temporary file after download (Streamlit handles this well)
            os.remove(pdf_output_path)


    else:
        st.warning("Could not detect pose landmarks in the image. Please ensure the person is clearly visible.")

