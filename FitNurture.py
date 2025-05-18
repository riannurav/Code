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
import gc  # Import garbage collector

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
    
    # Calculate new size maintaining aspect ratio
    ratio = max_size / max(img.size)
    if ratio < 1:  # Only resize if image is larger than max_size
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.LANCZOS)
    
    return img

# --- Page Config ---
st.set_page_config(
    page_title="FitNurture : Posture Detection",
    page_icon="🧘‍♀️",
    layout="wide"
)

# --- Custom CSS for Copyright Only ---
st.markdown("""
    <style>
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
# Title centered at the top
st.markdown("<h2 style='text-align: center; font-size: 24px; margin-bottom: 20px;'>FitNurture : Posture Detection</h2>", unsafe_allow_html=True)

# Center the logo using columns
col1, col2, col3 = st.columns([1.2, 1, 1.2])
with col2:
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
                st.image(logo_path, width=225, use_container_width=True)
                logo_found = True
                break
            except Exception as e:
                continue
    
    if not logo_found:
        st.warning("Logo not found. Please ensure the logo file is in the assets directory.")

# Add some spacing after the logo
st.markdown("<br>", unsafe_allow_html=True)

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

def generate_user_manual_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=14)

    # Add logo if available
    logo_paths = [
        os.path.join("assets", "logo.jpg"),
        os.path.join("assets", "logo.JPG"),
        os.path.join("assets", "logo.png"),
        os.path.join("assets", "logo.PNG")
    ]
    logo_path = next((p for p in logo_paths if os.path.exists(p)), None)
    if logo_path:
        pdf.image(logo_path, x=80, y=10, w=50)
        pdf.ln(40)
    else:
        pdf.ln(20)

    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 10, "FitNurture Posture Detection", ln=True, align='C')
    pdf.set_font("Arial", size=14)
    pdf.ln(10)
    pdf.cell(0, 10, "User Manual", ln=True, align='C')
    pdf.ln(15)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Step-by-Step Guide", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.ln(5)

    steps = [
        "1. Open the FitNurture app.",
        "2. Enter the required details (e.g., student name, age, etc.).",
        "3. Choose your input method: Upload Image or Use Camera.",
        "4. If uploading, select a clear, full-body image.",
        "5. If using the camera, follow the camera guidelines below.",
        "6. Submit the image for posture analysis.",
        "7. Review the detected posture and any recommendations.",
        "8. Download the PDF report if needed."
    ]
    for step in steps:
        pdf.multi_cell(0, 8, step)
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Camera Guidelines", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.ln(5)
    camera_guidelines = [
        "- Ensure your head is near the top center of the frame.",
        "- Stand straight, facing the camera.",
        "- Make sure your full body is visible in the frame.",
        "- Align your head between the horizontal guidelines.",
        "- Keep your body centered between the vertical guidelines.",
        "- Stand 6-8 feet away from the camera for best results.",
        "- Use good lighting to improve detection accuracy.",
        "- Avoid wearing loose or baggy clothing."
    ]
    for guide in camera_guidelines:
        pdf.multi_cell(0, 8, guide)
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Tips", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.ln(5)
    tips = [
        "- Review the captured image before submitting.",
        "- If the posture is not detected correctly, retake the photo.",
        "- For best results, use a plain background."
    ]
    for tip in tips:
        pdf.multi_cell(0, 8, tip)

    # Save to a temporary file and offer download in Streamlit
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        pdf.output(tmp_file.name)
        st.success("User manual generated!")
        with open(tmp_file.name, "rb") as f:
            st.download_button(
                label="Download User Manual (PDF)",
                data=f,
                file_name="FitNurture_User_Manual.pdf",
                mime="application/pdf"
            )

# --- App Config ---
@st.cache_resource
def load_pose_model():
    """Cache the MediaPipe pose model to prevent reloading"""
    return mp.solutions.pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1  # Use a lower complexity model
    )

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_static = load_pose_model()

# Initialize session state
for key in ['records', 'current_entry', 'landmark_image', 'abnormalities']:
    if key not in st.session_state:
        st.session_state[key] = [] if key == 'records' else {}

# --- Input Form and Image Processing ---
container = st.container()
with container:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        child_name = st.text_input("Whats the Child's Name? (This is a mandatory field)")
        st.markdown("**Note:** If you're using a mobile device, the camera input is more reliable than file uploads.")
        
        # Add abnormality selection section
        st.markdown("### Select Abnormalities to Detect")
        
        # Initialize session state for abnormality selections if not exists
        if 'selected_abnormalities' not in st.session_state:
            st.session_state.selected_abnormalities = {
                "Kyphosis": True,
                "Lordosis": True,
                "Tech Neck": True,
                "Scoliosis": True,
                "Flat Feet": True,
                "Gait Abnormalities": True,
                "Knock Knees": True,
                "Bow Legs": True
            }
        
        # Select All checkbox
        select_all = st.checkbox("Select All", value=all(st.session_state.selected_abnormalities.values()))
        
        st.markdown("---")
        
        # Individual abnormality checkboxes
        cols = st.columns(2)
        abnormality_list = list(st.session_state.selected_abnormalities.keys())
        half = len(abnormality_list) // 2
        
        # Update all checkboxes based on "Select All" state
        if select_all:
            st.session_state.selected_abnormalities = {k: True for k in st.session_state.selected_abnormalities}
        else:
            # If "Select All" is unchecked, uncheck all abnormalities
            if all(st.session_state.selected_abnormalities.values()):  # Only uncheck all if they were all checked
                st.session_state.selected_abnormalities = {k: False for k in st.session_state.selected_abnormalities}
        
        # First column of checkboxes
        with cols[0]:
            for abnormality in abnormality_list[:half]:
                st.session_state.selected_abnormalities[abnormality] = st.checkbox(
                    abnormality,
                    value=st.session_state.selected_abnormalities[abnormality]
                )
        
        # Second column of checkboxes
        with cols[1]:
            for abnormality in abnormality_list[half:]:
                st.session_state.selected_abnormalities[abnormality] = st.checkbox(
                    abnormality,
                    value=st.session_state.selected_abnormalities[abnormality]
                )
        
        st.markdown("---")
        input_mode = st.radio("Choose Input Mode", ["Upload Image", "Use Camera (Recommended for Mobile)"])

        image_data = None

        # Clear the other input type when switching modes
        if input_mode == "Upload Image":
            if "camera" in st.session_state:
                del st.session_state["camera"]
                clear_image_memory()
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
            if uploaded_file:
                image_data = Image.open(uploaded_file)
                image_data = optimize_image(image_data)  # Optimize uploaded image
        else:  # Camera mode
            if "upload" in st.session_state:
                del st.session_state["upload"]
                clear_image_memory()
            camera_data = st.camera_input("Take a picture using device")
            if camera_data:
                file_bytes = np.asarray(bytearray(camera_data.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_data = Image.fromarray(frame_rgb)
                    image_data = optimize_image(image_data)  # Optimize camera image

# Process image if available and name is provided
if image_data and child_name:
    img_np = np.array(image_data)
    if img_np.shape[-1] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    elif len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    # Create a placeholder for messages
    message_placeholder = st.empty()
    
    try:
        results = pose_static.process(img_np)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        results = None
    
    # Check if pose detection was successful
    if not results or not results.pose_landmarks:
        message_placeholder.error("⚠️ No person detected in the image. Please ensure that:")
        st.markdown("""
        - The full body is visible in the image
        - The person is standing straight
        - The lighting is adequate
        - The image is clear and not blurry
        """)
        # Clear any previous results
        clear_image_memory()
    else:
        # Clear any previous error message
        message_placeholder.empty()
        
        lm = results.pose_landmarks.landmark
        img_with_landmarks = img_np.copy()
        mp_drawing.draw_landmarks(
            img_with_landmarks, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
        
        # Store optimized landmark image
        st.session_state.landmark_image = optimize_image(img_with_landmarks)

        def is_landmark_visible(landmark):
            return (0.01 < landmark.x < 0.99 and 
                   0.01 < landmark.y < 0.99 and 
                   landmark.visibility > 0.5)

        # Calculate neck angle using ear, nose, and shoulder points for better accuracy
        neck_angle = calculate_angle(
            [lm[mp_pose.PoseLandmark.RIGHT_EAR.value].x, lm[mp_pose.PoseLandmark.RIGHT_EAR.value].y],
            [lm[mp_pose.PoseLandmark.NOSE.value].x, lm[mp_pose.PoseLandmark.NOSE.value].y],
            [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        )

        # Calculate forward head position
        ear_shoulder_distance = abs(lm[mp_pose.PoseLandmark.RIGHT_EAR.value].x - lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x)

        # Initialize metrics with None values
        metrics = {
            "shoulder_z": None,
            "hip_z": None,
            "knee_z": None,
            "neck_angle": neck_angle,
            "ear_shoulder_distance": ear_shoulder_distance,
            "shoulder_y_diff": None,
            "foot_z_diff": None,
            "ankle_x_diff": None,
            "knee_x_diff": None
        }

        # Only calculate metrics if relevant landmarks are visible
        if (is_landmark_visible(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]) and 
            is_landmark_visible(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])):
            metrics["shoulder_z"] = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z
            metrics["shoulder_y_diff"] = abs(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y - 
                                          lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)

        if is_landmark_visible(lm[mp_pose.PoseLandmark.LEFT_HIP.value]):
            metrics["hip_z"] = lm[mp_pose.PoseLandmark.LEFT_HIP.value].z

        if is_landmark_visible(lm[mp_pose.PoseLandmark.LEFT_KNEE.value]):
            metrics["knee_z"] = lm[mp_pose.PoseLandmark.LEFT_KNEE.value].z

        if (is_landmark_visible(lm[mp_pose.PoseLandmark.LEFT_HEEL.value]) and 
            is_landmark_visible(lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value])):
            metrics["foot_z_diff"] = abs(lm[mp_pose.PoseLandmark.LEFT_HEEL.value].z - 
                                      lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z)

        if (is_landmark_visible(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]) and 
            is_landmark_visible(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value])):
            metrics["ankle_x_diff"] = abs(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x - 
                                       lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x)

        if (is_landmark_visible(lm[mp_pose.PoseLandmark.LEFT_KNEE.value]) and 
            is_landmark_visible(lm[mp_pose.PoseLandmark.RIGHT_KNEE.value])):
            metrics["knee_x_diff"] = abs(lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x - 
                                      lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x)

        # Initialize abnormalities with only selected conditions
        abnormalities = {k: False for k in st.session_state.selected_abnormalities if st.session_state.selected_abnormalities[k]}

        # Only process selected abnormalities
        if "Kyphosis" in abnormalities:
            if metrics["shoulder_z"] is not None and metrics["hip_z"] is not None:
                abnormalities["Kyphosis"] = metrics["shoulder_z"] - metrics["hip_z"] > 0.15

        if "Lordosis" in abnormalities:
            if metrics["hip_z"] is not None and metrics["knee_z"] is not None:
                abnormalities["Lordosis"] = metrics["hip_z"] - metrics["knee_z"] > 0.1

        if "Tech Neck" in abnormalities:
            abnormalities["Tech Neck"] = (neck_angle > 45 and ear_shoulder_distance > 0.15)

        if "Scoliosis" in abnormalities:
            if metrics["shoulder_y_diff"] is not None:
                abnormalities["Scoliosis"] = metrics["shoulder_y_diff"] > 0.05

        if "Flat Feet" in abnormalities:
            if metrics["foot_z_diff"] is not None:
                abnormalities["Flat Feet"] = metrics["foot_z_diff"] < 0.05

        if "Gait Abnormalities" in abnormalities:
            if metrics["ankle_x_diff"] is not None:
                abnormalities["Gait Abnormalities"] = metrics["ankle_x_diff"] > 0.25

        if "Knock Knees" in abnormalities:
            if metrics["knee_x_diff"] is not None and metrics["ankle_x_diff"] is not None:
                abnormalities["Knock Knees"] = metrics["knee_x_diff"] < metrics["ankle_x_diff"] * 0.7

        if "Bow Legs" in abnormalities:
            if metrics["knee_x_diff"] is not None and metrics["ankle_x_diff"] is not None:
                abnormalities["Bow Legs"] = metrics["ankle_x_diff"] < metrics["knee_x_diff"] * 0.7

        entry = {
            "Student Name": child_name,
            "Student ID": f"FN-{random.randint(1000,9999)}",
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **abnormalities,
            **metrics
        }

        st.session_state.current_entry = entry
        st.session_state.abnormalities = abnormalities

        # Add warning for partial visibility
        visible_parts = [k for k, v in metrics.items() if v is not None]
        if len(visible_parts) < len(metrics):
            st.warning("⚠️ Some body parts are not fully visible in the image. Only partial posture analysis is possible.")

# Add Save Button
if st.session_state.get("current_entry") and st.session_state.get("landmark_image") is not None:
    st.success("Analysis Complete")
    st.image(st.session_state.landmark_image, caption="Landmarked Image", use_container_width=True)
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
    # Implement pagination for large datasets
    records_per_page = 10
    total_pages = len(st.session_state.records) // records_per_page + 1
    current_page = st.selectbox("Select Page", range(1, total_pages + 1)) - 1
    
    start_idx = current_page * records_per_page
    end_idx = start_idx + records_per_page
    
    search_term = st.text_input("🔍 Search by Student Name or ID", key="search")
    df = pd.DataFrame(st.session_state.records[start_idx:end_idx])
    if search_term:
        df = df[df['Student Name'].str.contains(search_term, case=False) | 
                df['Student ID'].str.contains(search_term, case=False)]
    st.dataframe(df, use_container_width=True)
    
    # Optimize CSV download
    @st.cache_data
    def convert_to_csv(df):
        return df.to_csv(index=False).encode("utf-8")
    
    csv = convert_to_csv(df)
    st.download_button("📥 Download CSV", data=csv, file_name="posture_records.csv", mime="text/csv")
else:
    st.info("No records to display yet.")

# Clean up resources when the script ends
clear_image_memory()

# Center the User Manual Download Button above the footer
button_col1, button_col2, button_col3 = st.columns([2, 1, 2])
with button_col2:
    manual_path = os.path.join("assets", "FitNurture_User_Manual.pdf")
    if os.path.exists(manual_path):
        with open(manual_path, "rb") as f:
            st.download_button(
                label="Download User Manual (PDF)",
                data=f,
                file_name="FitNurture_User_Manual.pdf",
                mime="application/pdf"
            )
    else:
        st.warning("User manual PDF not found in assets folder.")

# Add copyright footer
st.markdown("""
    <div class="copyright-footer">
        © Copyright 2025 FutureNurture | <a href="http://www.futurenurture.in" target="_blank">www.futurenurture.in</a>
    </div>
""", unsafe_allow_html=True)
