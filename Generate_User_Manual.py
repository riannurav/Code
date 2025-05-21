import os
from fpdf import FPDF
from PIL import Image # Added to get image dimensions for better logo scaling

# --- Configuration ---
# Adjust these if your logo has a different name or path structure
POSSIBLE_LOGO_NAMES = ["logo.png", "logo.jpg", "logo.JPG", "logo.PNG"]
ASSETS_FOLDER = "assets" 
OUTPUT_FILENAME = "FitNurture_User_Manual.pdf"
# Corrected CONTACT_INFO_PLACEHOLDER (removed trailing ']')
CONTACT_INFO_PLACEHOLDER = "www.futurenurture.in | info@futurenurture.in"

# --- PDF Class ---
class PDFManual(FPDF):
    def header(self):
        pass # Logo will be added manually after the first add_page()

    def footer(self):
        self.set_y(-15) # Position 1.5 cm from bottom
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def _write_styled_text(self, text_line, height):
        """Helper to write text with **bold** markdown.
        Input text_line is a Python 3 Unicode string."""
        parts = text_line.split('**')
        for i, part in enumerate(parts):
            if not part:  
                continue
            
            current_style = self.font_style if self.font_style else ''
            if i % 2 == 1: # Bold part
                if 'B' not in current_style:
                    self.set_font('', current_style + 'B')
            else: # Normal part
                if 'B' in current_style:
                    self.set_font('', current_style.replace('B', ''))
            
            part_to_write = part.replace("[Insert Contact Email or Support Channel Information Here - e.g., support@futurenurture.in]", CONTACT_INFO_PLACEHOLDER)
            
            # FPDF's write method will attempt to render the Unicode string.
            # Characters not supported by the current font (e.g., Arial) will be replaced or omitted by FPDF.
            self.write(height, part_to_write)

    def add_title(self, level, title_text):
        self.ln(6) 
        if level == 1:
            self.set_font('Arial', 'B', 20)
            self.multi_cell(0, 10, title_text, 0, 'C')
            self.ln(4)
        elif level == 2:
            self.set_font('Arial', 'B', 16)
            self.multi_cell(0, 8, title_text, 0, 'L')
            self.ln(3)
        elif level == 3:
            self.set_font('Arial', 'B', 12)
            self.multi_cell(0, 7, title_text, 0, 'L')
            self.ln(2)
        
    def add_paragraph(self, text_block):
        self.set_font('Arial', '', 10)
        line_height = 5
        cleaned_text_block = text_block.replace("[Insert Contact Email or Support Channel Information Here - e.g., support@futurenurture.in]", CONTACT_INFO_PLACEHOLDER)
        # Replace **bold** markers with empty string for simple paragraphs, or implement proper parsing
        cleaned_text_block = cleaned_text_block.replace("**", "") 
        self.multi_cell(0, line_height, cleaned_text_block, 0, 'L')
        self.ln(line_height / 2) 

    def add_list_item(self, text_line):
        self.set_font('Arial', '', 10)
        line_height = 5
        
        self.set_x(self.l_margin) # Ensure list item starts at the left margin
        
        bullet_text = "  -  "
        bullet_cell_width = 10  # Fixed width for the bullet part
        self.cell(bullet_cell_width, line_height, bullet_text, 0, 0) # ln=0 to stay on the same line

        clean_line = text_line.lstrip('*- ').strip()
        clean_line = clean_line.replace("[Insert Contact Email or Support Channel Information Here - e.g., support@futurenurture.in]", CONTACT_INFO_PLACEHOLDER)
        # Replace **bold** markers for list items as well for now
        clean_line = clean_line.replace("**", "")

        if not clean_line: # If the line is empty after stripping
            self.ln(line_height) # Just move to next line
            return

        # Calculate available width for the text part dynamically
        text_x_start = self.get_x() # X position after the bullet cell
        available_text_width = self.w - self.r_margin - text_x_start
        
        # Defensive check for available width
        if available_text_width < 1: 
            print(f"DEBUG: List item text: '{clean_line[:50]}...'")
            print(f"DEBUG: Calculated available_text_width for list item is too small or negative: {available_text_width}. Skipping multi_cell for this item.")
            self.ln(line_height) # Move to next line to avoid error
            return
            
        # Debug print before calling multi_cell
        # print(f"DEBUG: List item text: '{clean_line[:50]}...', Available width: {available_text_width}")
        self.multi_cell(available_text_width, line_height, clean_line, 0, 'L', ln=1)
        
    def add_horizontal_rule(self):
        self.ln(2)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)

# --- Markdown Content ---
# (MARKDOWN_MANUAL_CONTENT remains the same as in the immersive artifact)
# Emojis replaced with text alternatives for better compatibility with default FPDF fonts
MARKDOWN_MANUAL_CONTENT = """
# FitNurture Posture Detection App - User Manual

## 1. Introduction

Welcome to the FitNurture Posture Detection App! This application is designed to help identify potential postural abnormalities in students using image analysis. By capturing a clear image, the app can provide an initial assessment, generate a report, and store data for future reference and analysis, including potential use in developing more accurate machine learning models.

This manual will guide you through using the app effectively, from capturing the right kind of image to understanding the results and managing data.

## 2. Getting Started

### What You Need:
* A device with a camera (smartphone, tablet, or computer with a webcam).
* Good lighting conditions.
* A clear, uncluttered background.
* The student whose posture is to be analyzed.

### Accessing the App:
Simply navigate to the web address where the FitNurture app is hosted (e.g., your Streamlit Cloud URL).

## 3. Capturing an Accurate Posture Image

The accuracy of the posture analysis heavily depends on the quality of the image you provide. Please follow these guidelines carefully:

### 3.1. Importance of a Good Image
A clear, well-positioned image allows the app's AI to accurately identify key body landmarks, which are crucial for calculating posture metrics. Poor images can lead to inaccurate results or failure to detect a person.

### 3.2. Camera Placement and Subject Positioning

**a. Distance from Subject:**
* Position the camera far enough away so that the **entire body** of the student, from head to feet, is clearly visible within the frame.
* There should be some space above the head and below the feet.

**b. Camera Angle and Height:**
* The camera should be placed at approximately the **student's mid-torso (chest/stomach) height**.
* Avoid angling the camera upwards or downwards. It should be as level as possible, pointing directly at the student.

**c. Full Body Visibility:**
* **Crucial:** The entire body must be in the shot. This includes the top of the head, both arms (even if by the sides), the torso, both legs, and the feet.
* Ensure no body parts are cut off by the edge of the frame.

**d. Subject's Posture for the Photo:**
* The student should stand **straight and tall**, in a relaxed but upright natural stance.
* They should be facing **directly sideways** to the camera for a profile view. The app is designed to analyze a sagittal plane (side view) posture. (If a frontal view is required for specific analyses in the future, that would need different instructions).
* Feet should be shoulder-width apart, or a comfortable natural stance, pointing forward.
* Arms should hang naturally at their sides.
* Ask the student to look straight ahead, not up, down, or at the camera (unless the camera is directly in front of them at eye level for a frontal shot, which is not the primary mode for this analysis).

**e. Clothing:**
* Wear form-fitting clothing if possible. Bulky or loose clothing can obscure body landmarks and affect accuracy.
* Avoid clothing with busy patterns that might confuse the landmark detection. Solid, contrasting colors (relative to the background) are best.

**f. Background:**
* Use a **plain, uncluttered background**. A clear wall is ideal.
* Ensure there are no objects or other people in the background that could interfere with person detection.
* The color of the background should contrast with the student's clothing.

**g. Lighting:**
* Ensure the student is **well and evenly lit**.
* Avoid shadows falling on the student.
* Natural daylight is often best, but good indoor lighting can also work. Avoid direct, harsh light that creates strong shadows or overexposure.
* Do not have a bright light source (like a window) directly behind the student, as this will make them appear as a silhouette.

### 3.3. Using the App's Camera Input
1.  Once on the app page, select the "Use Camera" input mode.
2.  If prompted by your browser, **allow the app to access your camera**.
3.  Position the student according to the guidelines above.
4.  Use the camera preview in the app to frame the student correctly.
5.  Click the "Take a picture" button.

### 3.4. Uploading an Existing Image
If you have a pre-existing image that meets all the guidelines:
1.  Select the "Upload Image" input mode.
2.  Click the "Browse files" button (or drag and drop an image).
3.  Select the image file (JPG, PNG, JPEG) from your device.

## 4. Performing the Analysis

### 4.1. Entering Child's Name
* Before taking a picture or uploading an image, you **must** enter the child's name in the "Child's Name" field. This is a mandatory field for associating the analysis with the correct individual.
* **Crucial for Data Accuracy:** Ensure you **change the name for each new student** being analyzed. If you analyze multiple students in one session, always update this field before capturing or uploading the image for the next student. Failure to do so will associate the new analysis with the previously entered name.

### 4.2. Selecting Abnormalities to Detect
* Below the name input, you'll find a section titled "Select Abnormalities to Detect."
* By default, all listed abnormalities are selected for detection.
* You can uncheck any specific abnormalities you do not want the app to analyze for this particular session.
* There's a "Select All" checkbox to quickly toggle all options on or off.

### 4.3. Understanding the Results
* After a successful image capture and processing, the app will display:
    * An image with detected body landmarks overlaid.
    * A list of the selected abnormalities and whether they are considered "Present" or "Not Present" based on the app's analysis.
* **Important:** The results are based on an automated analysis and are for informational purposes only. They are not a substitute for professional medical advice.

## 5. Saving and Managing Results

### 5.1. Saving Results Locally
* After an analysis is complete, if you are satisfied with the result, you can save it.
* Click the "**(Save Icon) Save Result Locally**" button.
* This will store the analysis data (student details, detected abnormalities, and calculated metrics) within your current browser session. This data is not yet uploaded to the cloud.

### 5.2. Viewing Collected Records
* Scroll down to the "**(Chart Icon) View Locally Saved Records**" section.
* Here, you'll see a table of all the records you've saved locally during your current session.
* If there are many records, you can use the "Select Page" dropdown to navigate through them.
* You can also use the "Search by Student Name or ID" field to find specific records.

### 5.3. Downloading Local Records (CSV)
* In the "View Locally Saved Records" section, you'll find a "**(Download Icon) Download All Local Records (CSV)**" button.
* Clicking this will download a CSV (Comma Separated Values) file containing all the records currently saved locally. This file can be opened with spreadsheet software like Microsoft Excel, Google Sheets, etc.

## 6. Uploading Data to the Cloud (Azure SQL)

### 6.1. Purpose of Cloud Upload
* Uploading data to the cloud (Azure SQL database) allows for:
    * Permanent storage of analysis records beyond your current browser session.
    * Centralized data access for authorized personnel.
    * Accumulation of a dataset that can be used for further analysis, research, and the long-term goal of training machine learning models to improve detection accuracy.

### 6.2. How to Upload
1.  Ensure you have one or more records saved locally (see section 5.1).
2.  Scroll down to the "**(Cloud Icon) Cloud Storage (Azure SQL)**" section.
3.  Click the "**(Upload Icon) Upload All Saved Local Records to Azure SQL**" button.
4.  The app will attempt to connect to the database and upload all records currently stored in the "View Locally Saved Records" table.
5.  You will see a success message if the upload is complete, or an error message if issues occur.

### 6.3. What Happens to the Data
* When uploaded, the data for each student (including their ID, name, timestamp, detected abnormalities, and posture metrics) is stored in the secure Azure SQL database.
* If a record with the same "Student ID" already exists in the database, the existing record will be updated with the new information. Otherwise, a new record will be created.

## 7. Generating a PDF Report

After a successful analysis:
1.  Click the "**(Document Icon) Generate PDF Report**" button.
2.  The app will create a PDF document containing:
    * The FitNurture report title and website.
    * The company logo.
    * Student's Name, ID, and the analysis Timestamp.
    * The landmarked image.
    * A summary of detected postural issues.
    * General recommendations for any detected conditions.
    * A disclaimer.
3.  Once generated, a "**(Download Icon) Download Report PDF**" button will appear. Click this to save the PDF file to your device.

## 8. Troubleshooting

### Common Issues & Solutions:

* **"No person detected" or Poor Landmark Detection:**
    * **Cause:** Image quality is likely the issue.
    * **Solution:** Review section 3.2 ("Camera Placement and Subject Positioning") carefully. Ensure:
        * Full body is visible (head to feet).
        * Good, even lighting (no strong shadows or backlighting).
        * Plain, uncluttered background.
        * Subject is standing straight and sideways to the camera.
        * Appropriate clothing (not too loose).
        * Camera is stable and image is not blurry.
        * Try taking the picture again, adjusting distance or angle slightly.

* **PDF Generation Issues (e.g., Blank PDF, Errors):**
    * **Cause:** Could be due to issues with the data being processed or temporary app/browser glitches.
    * **Solution:**
        * Ensure a successful analysis was completed and results are displayed on the screen.
        * Try refreshing the app page and performing the analysis again.
        * If the problem persists, note any error messages displayed and contact support.

* **Cloud Upload Errors (e.g., "Database Connection Error"):**
    * **Cause:** May be due to internet connectivity issues, temporary server problems, or incorrect app configuration (usually an admin concern).
    * **Solution:**
        * Check your internet connection.
        * Try again after a few minutes.
        * If the issue persists, it might require attention from the app administrator to check database credentials and server status.

* **Camera Not Working (especially on Mobile):**
    * **Cause:** Browser might not have permission to access the camera.
    * **Solution:**
        * When prompted by your browser, ensure you "Allow" camera access for the app's website.
        * Check your browser's settings for camera permissions for the site.
        * Try a different browser if the issue continues.
        * Ensure no other app is currently using the camera.

## 9. Contact / Support

For any issues not covered in this manual or for further assistance, please contact:
[Insert Contact Email or Support Channel Information Here - e.g., support@futurenurture.in]

## 10. Disclaimer

This automated analysis is for informational purposes only and not a substitute for professional medical advice. Consult a healthcare provider for health concerns.

---
*Thank you for using the FitNurture Posture Detection App!*
"""

# --- Main PDF Generation Function ---
def generate_manual_pdf(markdown_text, output_filename=OUTPUT_FILENAME):
    pdf = PDFManual(orientation='P', unit='mm', format='A4')
    
    # NOTE on Unicode:
    # The FPDF library (original PyFPDF) has limited support for Unicode characters with its core fonts (like Arial).
    # For full Unicode character rendering (e.g., emojis, etc.), you would typically need to:
    # 1. Use a Unicode-supporting font (e.g., DejaVuSans.ttf).
    # 2. Add this font to FPDF using `pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)`.
    # 3. Set this font using `pdf.set_font("DejaVu", "", 10)`.
    # This script uses default fonts, so some special Unicode characters from the Markdown
    # might not render correctly in the PDF. Python 3 strings are inherently Unicode,
    # so the script handles them correctly up to the point of passing them to FPDF.

    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15) 
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.set_top_margin(15)

    # Add Logo at the very top
    logo_found_path = None
    for name in POSSIBLE_LOGO_NAMES:
        path_to_check = os.path.join(ASSETS_FOLDER, name)
        if os.path.exists(path_to_check):
            logo_found_path = path_to_check
            break
    
    if logo_found_path:
        try:
            img = Image.open(logo_found_path)
            img_w_px, img_h_px = img.size
            aspect_ratio = img_h_px / img_w_px if img_w_px > 0 else 1
            logo_w_mm = 30 
            logo_h_mm = logo_w_mm * aspect_ratio
            
            page_width_mm = 210 
            logo_x_pos = (page_width_mm - logo_w_mm) / 2
            
            pdf.image(logo_found_path, x=logo_x_pos, y=10, w=logo_w_mm, h=logo_h_mm)
            pdf.ln(logo_h_mm + 5) 
        except Exception as e:
            print(f"Warning: Could not process logo image {logo_found_path}: {e}")
            pdf.set_font('Arial', 'I', 8)
            pdf.cell(0,10, "[Error processing logo]",0,1,'C')
            pdf.ln(10)
    else:
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0,10, "[Logo not found in assets folder]",0,1,'C')
        pdf.ln(10)


    lines = markdown_text.strip().split('\n')
    
    paragraph_buffer = [] 

    for line in lines:
        stripped_line = line.strip()

        if stripped_line.startswith('### '):
            if paragraph_buffer: pdf.add_paragraph(" ".join(paragraph_buffer)); paragraph_buffer = []
            pdf.add_title(3, stripped_line[4:])
        elif stripped_line.startswith('## '):
            if paragraph_buffer: pdf.add_paragraph(" ".join(paragraph_buffer)); paragraph_buffer = []
            pdf.add_title(2, stripped_line[3:])
        elif stripped_line.startswith('# '):
            if paragraph_buffer: pdf.add_paragraph(" ".join(paragraph_buffer)); paragraph_buffer = []
            pdf.add_title(1, stripped_line[2:])
        elif stripped_line.startswith('* ') or stripped_line.startswith('- '):
            if paragraph_buffer: pdf.add_paragraph(" ".join(paragraph_buffer)); paragraph_buffer = []
            pdf.add_list_item(stripped_line)
        elif stripped_line.startswith('<!--'): 
            continue
        elif stripped_line == "---":
            if paragraph_buffer: pdf.add_paragraph(" ".join(paragraph_buffer)); paragraph_buffer = []
            pdf.add_horizontal_rule()
        elif stripped_line: 
            paragraph_buffer.append(stripped_line)
        else: 
            if paragraph_buffer:
                pdf.add_paragraph(" ".join(paragraph_buffer))
                paragraph_buffer = []
            pdf.ln(3) 

    if paragraph_buffer:
        pdf.add_paragraph(" ".join(paragraph_buffer))

    try:
        pdf.output(output_filename, "F")
        print(f"User manual '{output_filename}' generated successfully.")
    except Exception as e:
        print(f"Error saving PDF: {e}")
        print("Make sure you have write permissions in the current directory or specify a full path for the output file.")

# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.exists(ASSETS_FOLDER):
        os.makedirs(ASSETS_FOLDER)
        print(f"Created '{ASSETS_FOLDER}' directory. Place your logo (e.g., logo.png) there for it to be included.")

    generate_manual_pdf(MARKDOWN_MANUAL_CONTENT)
