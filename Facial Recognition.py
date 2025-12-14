#Facial Recognition
import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import pickle
from datetime import datetime
import csv
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# Load pre-trained face encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1920, 1080)}))
picam2.start()

# Email configuration - Update these with your email credentials
SMTP_SERVER = "smtp.gmail.com"  # For Gmail. For Outlook: smtp-mail.outlook.com, For Yahoo: smtp.mail.yahoo.com
SMTP_PORT = 587  # Use 587 for TLS, 465 for SSL
SENDER_EMAIL = "rutvikdeshpande11@gmail.com"  # Your email address
SENDER_PASSWORD = "vpch toji olin pfsc"  # Your email password or App Password (for Gmail, use App Password)
RECIPIENT_EMAIL = "preetamd@gmail.com"  # Recipient email address

# Initialize our variables
cv_scaler = 4 # this has to be a whole number
photos_folder = "detected_photos"  # Folder to save photos of detected persons
attendance_file = "attendance.csv"  # CSV file to store attendance records

# Create photos folder if it doesn't exist
if not os.path.exists(photos_folder):
    os.makedirs(photos_folder)

face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0
printed_names = set()  # Track names that have been printed
attendance_data = {}  # Track attendance with timestamps: {name: [datetime1, datetime2, ...]}
detected_images = {}  # Track images of detected persons: {name: image_path}

def process_frame(frame):
    global face_locations, face_encodings, face_names, printed_names
    
    # Resize the frame using cv_scaler to increase performance (less pixels processed, less time spent)
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    
    # Convert the image from BGR to RGB colour space, the facial recognition library uses RGB, OpenCV uses BGR
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')
    
    face_names = []
    for i, face_encoding in enumerate(face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        # Print the name once when first detected (skip "Unknown")
        if name != "Unknown" and name not in printed_names:
            print(f"Person detected: {name}")
            printed_names.add(name)
            
            # Record attendance with timestamp
            current_time = datetime.now()
            if name not in attendance_data:
                attendance_data[name] = []
            attendance_data[name].append(current_time)
            
            # Save to CSV file
            save_attendance(name, current_time)
            
            # Capture and save image of the detected person
            if i < len(face_locations):
                # Get face location in original frame coordinates
                (top, right, bottom, left) = face_locations[i]
                top_orig = top * cv_scaler
                right_orig = right * cv_scaler
                bottom_orig = bottom * cv_scaler
                left_orig = left * cv_scaler
                
                # Add some padding around the face
                padding = 50
                top_orig = max(0, top_orig - padding)
                left_orig = max(0, left_orig - padding)
                bottom_orig = min(frame.shape[0], bottom_orig + padding)
                right_orig = min(frame.shape[1], right_orig + padding)
                
                # Extract face region from original frame
                face_image = frame[top_orig:bottom_orig, left_orig:right_orig]
                
                # Save the image
                timestamp_str = current_time.strftime('%Y%m%d_%H%M%S')
                image_filename = f"{name}_{timestamp_str}.jpg"
                image_path = os.path.join(photos_folder, image_filename)
                cv2.imwrite(image_path, face_image)
                
                # Store image path for email attachment
                detected_images[name] = image_path
                print(f"[INFO] Saved photo: {image_path}")
        
        face_names.append(name)
    
    return frame

def draw_results(frame):
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)
        
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left -3, top - 35), (right+3, top), (244, 42, 3), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)

    return frame

def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps

def save_attendance(name, timestamp):
    """Save attendance record to CSV file"""
    file_exists = os.path.isfile(attendance_file)
    with open(attendance_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Name', 'Date', 'Time', 'DateTime'])
        writer.writerow([name, timestamp.strftime('%Y-%m-%d'), 
                        timestamp.strftime('%H:%M:%S'), 
                        timestamp.strftime('%Y-%m-%d %H:%M:%S')])

def send_attendance_email():
    """Send attendance report via email using SMTP"""
    # Get all unique known names from the training data
    all_known_names = sorted(list(set(known_face_names)))
    
    # Get current timestamp
    current_timestamp = datetime.now()
    timestamp_str = current_timestamp.strftime('%Y-%m-%d %H:%M:%S')
    
    # Build email content
    email_subject = f"Attendance Report - {current_timestamp.strftime('%Y-%m-%d')}"
    email_body = f"Attendance Report for {timestamp_str}\n\n"
    email_body += "=" * 60 + "\n\n"
    
    # Read attendance data from CSV for today
    today = current_timestamp.strftime('%Y-%m-%d')
    today_records = {}
    
    if os.path.isfile(attendance_file):
        with open(attendance_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Date'] == today:
                    name = row['Name']
                    if name not in today_records:
                        today_records[name] = []
                    today_records[name].append(row['Time'])
    
    # Build attendance list with all names
    email_body += f"Today's Attendance ({today}):\n"
    email_body += "-" * 60 + "\n\n"
    
    present_count = 0
    absent_count = 0
    
    # List all known names with their status
    # Only mark as present if their photo was taken
    for name in all_known_names:
        if name in today_records and name in detected_images:
            # Person is present (detected AND photo captured)
            times = today_records[name]
            first_seen = times[0]
            last_seen = times[-1]
            count = len(times)
            email_body += f"✓ {name}: PRESENT\n"
            email_body += f"  First seen: {first_seen}\n"
            email_body += f"  Last seen: {last_seen}\n"
            email_body += f"  Total detections: {count}\n\n"
            present_count += 1
        else:
            # Person is absent (either not detected or photo not taken)
            email_body += f"✗ {name}: ABSENT\n\n"
            absent_count += 1
    
    # Summary
    email_body += "-" * 60 + "\n"
    email_body += f"Summary:\n"
    email_body += f"  Present: {present_count}\n"
    email_body += f"  Absent: {absent_count}\n"
    email_body += f"  Total: {len(all_known_names)}\n"
    
    email_body += "\n" + "=" * 60 + "\n"
    email_body += "Photos of detected persons are attached to this email.\n"
    email_body += "This is an automated message from the Raspberry Pi Attendance System.\n"
    
    # Create email message
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = email_subject
    msg.attach(MIMEText(email_body, 'plain'))
    
    # Attach images of detected persons
    for name, image_path in detected_images.items():
        if os.path.isfile(image_path):
            try:
                with open(image_path, 'rb') as f:
                    img_data = f.read()
                    image = MIMEImage(img_data)
                    image.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
                    msg.attach(image)
                    print(f"[INFO] Attached photo for {name}: {os.path.basename(image_path)}")
            except Exception as e:
                print(f"[WARNING] Failed to attach image for {name}: {str(e)}")
    
    # Send email using SMTP
    try:
        # Connect to SMTP server
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()  # Enable encryption
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        
        # Send email
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, text)
        server.quit()
        
        print(f"[INFO] Attendance email sent successfully to {RECIPIENT_EMAIL}")
    except smtplib.SMTPAuthenticationError:
        print(f"[ERROR] Authentication failed. Please check your email and password.")
        print(f"[INFO] For Gmail, you need to use an App Password, not your regular password.")
        print(f"[INFO] Generate one at: https://myaccount.google.com/apppasswords")
    except smtplib.SMTPException as e:
        print(f"[ERROR] SMTP error occurred: {str(e)}")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {str(e)}")


while True:
    # Capture a frame from camera
    frame = picam2.capture_array()
    
    # Process the frame with the function
    processed_frame = process_frame(frame)
    
    # Get the text and boxes to be drawn based on the processed frame
    display_frame = draw_results(processed_frame)
    
    # Calculate and update FPS
    current_fps = calculate_fps()
    
    # Attach FPS counter to the text and boxes
    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (display_frame.shape[1] - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display everything over the video feed.
    cv2.imshow('Video', display_frame)
    
    # Break the loop and stop the script if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# By breaking the loop we run this code here which closes everything
cv2.destroyAllWindows()
picam2.stop()

# Send attendance email when script exits
print("[INFO] Program exited. Sending attendance report...")
send_attendance_email()
print("[INFO] Script ended.")
