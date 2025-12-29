#Facial Recognition
import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import pickle
from datetime import datetime
import subprocess
import csv
import os

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

# Initialize our variables
cv_scaler = 4 # this has to be a whole number

face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0
printed_names = set()  # Track names that have been printed
attendance_data = {}  # Track attendance with timestamps: {name: [datetime1, datetime2, ...]}
attendance_file = "attendance.csv"  # CSV file to store attendance records

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
    for face_encoding in face_encodings:
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
    """Send attendance report via email using system mail command"""
    # Read attendance from CSV file
    if not os.path.isfile(attendance_file):
        print("[INFO] No attendance records found. Email not sent.")
        return
    
    # Build email content
    email_subject = f"Attendance Report - {datetime.now().strftime('%Y-%m-%d')}"
    email_body = f"Attendance Report for {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    email_body += "=" * 50 + "\n\n"
    
    # Read attendance data from CSV
    today = datetime.now().strftime('%Y-%m-%d')
    today_records = []
    
    with open(attendance_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Date'] == today:
                today_records.append(row)
    
    if today_records:
        email_body += f"Today's Attendance ({today}):\n"
        email_body += "-" * 50 + "\n"
        
        # Group by name
        name_records = {}
        for record in today_records:
            name = record['Name']
            if name not in name_records:
                name_records[name] = []
            name_records[name].append(record['Time'])
        
        for name in sorted(name_records.keys()):
            times = name_records[name]
            first_seen = times[0]
            last_seen = times[-1]
            count = len(times)
            email_body += f"{name}:\n"
            email_body += f"  First seen: {first_seen}\n"
            email_body += f"  Last seen: {last_seen}\n"
            email_body += f"  Total detections: {count}\n\n"
    else:
        email_body += "No attendance records for today.\n"
    
    email_body += "\n" + "=" * 50 + "\n"
    email_body += "This is an automated message from the Raspberry Pi Attendance System.\n"
    
    # Get recipient email from environment variable or use default
    recipient_email = os.environ.get('ATTENDANCE_EMAIL', 'email')  # Change default as needed
    
    # Send email using system mail command (requires mailutils or sendmail)
    try:
        # Create email content with headers
        email_content = f"""Subject: {email_subject}
To: {recipient_email}
From: raspberrypi@local
Content-Type: text/plain

{email_body}
"""
        
        # Send email using sendmail command
        process = subprocess.Popen(['sendmail', recipient_email], 
                                  stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
        process.communicate(input=email_content.encode('utf-8'))
        
        if process.returncode == 0:
            print(f"[INFO] Attendance email sent successfully to {recipient_email}")
        else:
            print(f"[WARNING] Failed to send email. Make sure mailutils is installed: sudo apt-get install mailutils")
    except FileNotFoundError:
        print(f"[WARNING] sendmail not found. Install mailutils: sudo apt-get install mailutils")
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
