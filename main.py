import os
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone
from datetime import datetime
import pytz

# --- Supabase Imports ---
from supabase import create_client, Client
# ------------------------

# --- Supabase Configuration ---
SUPABASE_URL = "https://iemssxxuliutroobfesv.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImllbXNzeHh1bGl1dHJvb2JmZXN2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI3NTA4NzksImV4cCI6MjA2ODMyNjg3OX0.yWsYo8AIhLCAWpwIhgbO9f1BJXV8-zdaiJ9LsRPEDYQ"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
SUPABASE_STORAGE_BUCKET = "attendance"
# --------------------------------

cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread('Resources/background.png')

# Importing the mode images into a list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

# Add debug prints for mode images
print("Mode images loaded:")
for i, img_mode in enumerate(imgModeList):
    print(f"  imgModeList[{i}] shape: {img_mode.shape if img_mode is not None else 'None (failed to load)'} - File: {modePathList[i] if i < len(modePathList) else 'N/A'}")


# Load the encoding file
print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print("Encode File Loaded")

modeType = 0
counter = 0
student_id = -1
imgStudent = []
studentInfo = {}

# Define UTC timezone once globally for efficiency
utc_timezone = pytz.utc

# Define re-attendance interval
ATTENDANCE_INTERVAL_SECONDS = 30 # Set this to your desired cooldown in seconds (e.g., 30 for 30 seconds)
DISPLAY_DETAILS_FRAMES = 5 # How many frames to show student details before marking decision

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBackground[162:162 + 480, 55:55 + 640] = img
    # This line draws the mode image. It should always be drawn based on the current modeType.
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    if faceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                student_id = studentIds[matchIndex]

                if counter == 0:
                    cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                    cv2.imshow("Face Attendance", imgBackground)
                    cv2.waitKey(1)
                    counter = 1
                    modeType = 1 # Set mode to 1 (fetching data and then displaying details)

        if counter != 0:
            # print(f"DEBUG: Current Counter: {counter}, Mode Type: {modeType}")

            # Step 1: Fetch data and image ONCE when counter is 1
            if counter == 1:
                # --- Get the Data from Supabase Database ---
                try:
                    # print(f"DEBUG: Attempting to fetch data for {student_id}")
                    response = supabase.table('students').select('*').eq('id', student_id).single().execute()
                    studentInfo = response.data
                    # print(f"DEBUG: Successfully fetched studentInfo: {studentInfo}")
                except Exception as e:
                    print(f"\n!!!!!!!!! ERROR FETCHING STUDENT DATA !!!!!!!!!")
                    print(f"DEBUG: Student ID being queried: {student_id}")
                    print(f"DEBUG: Error details: {e}")
                    studentInfo = {}
                    modeType = 0
                    counter = 0
                    cvzone.putTextRect(imgBackground, "Error fetching data", (275, 400), colorR=(0,0,255))
                    cv2.imshow("Face Attendance", imgBackground)
                    cv2.waitKey(1)
                    continue

                # --- Get the Image from Supabase Storage ---
                try:
                    res = supabase.storage.from_(SUPABASE_STORAGE_BUCKET).download(f'images/{student_id}.png')
                    array = np.frombuffer(res, np.uint8)
                    imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
                    if imgStudent is None: # Check if imdecode failed
                        raise ValueError("Failed to decode image from bytes.")
                except Exception as e:
                    print(f"Error fetching or decoding student image: {e}")
                    imgStudent = []
                    cvzone.putTextRect(imgBackground, "Error loading image", (275, 400), colorR=(0,0,255))
                    cv2.imshow("Face Attendance", imgBackground)
                    cv2.waitKey(1)
                    modeType = 0
                    counter = 0
                    continue

            # Step 2: Display Student Details (modeType 1)
            if modeType == 1:
                if counter <= DISPLAY_DETAILS_FRAMES and studentInfo: # Show student info for X frames
                    cv2.putText(imgBackground, str(studentInfo.get('total_attendance', 'N/A')), (861, 125),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentInfo.get('major', 'N/A')), (1006, 550),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(student_id), (1006, 493),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentInfo.get('standing', 'N/A')), (910, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(imgBackground, str(studentInfo.get('year', 'N/A')), (1025, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(imgBackground, str(studentInfo.get('starting_year', 'N/A')), (1125, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                    student_name = studentInfo.get('name', 'N/A')
                    (w, h), _ = cv2.getTextSize(student_name, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    offset = (414 - w) // 2
                    cv2.putText(imgBackground, student_name, (808 + offset, 445),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                    if imgStudent is not None and len(imgStudent) > 0:
                        imgBackground[175:175 + 216, 909:909 + 216] = imgStudent
                    else:
                        cv2.putText(imgBackground, "No Image", (909, 283),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)
                    counter += 1 # Increment counter while displaying student info
                else: # After displaying details for enough frames, proceed to attendance logic
                    # --- ATTENDANCE LOGIC AND UPDATE ---
                    last_attendance_str = studentInfo.get('last_attendance_time')

                    # Parse timestamp (with fix for microseconds and timezone colon)
                    if last_attendance_str:
                        try:
                            # Using fromisoformat() for Python 3.7+ is the cleanest
                            # datatimeObject = datetime.fromisoformat(last_attendance_str)
                            # Or if below 3.7, use string manipulation
                            if last_attendance_str.endswith('+00:00'):
                                cleaned_time_str = last_attendance_str.replace('+00:00', '+0000')
                            elif last_attendance_str.endswith('-00:00'):
                                cleaned_time_str = last_attendance_str.replace('-00:00', '-0000')
                            else:
                                cleaned_time_str = last_attendance_str # Fallback for other offsets

                            # Ensure it's parsed with microseconds
                            datatimeObject = datetime.strptime(cleaned_time_str, "%Y-%m-%dT%H:%M:%S.%f%z")

                            current_utc_time = datetime.now(utc_timezone)
                            # Ensure both datetime objects are timezone-aware and in UTC for comparison
                            if datatimeObject.tzinfo is None:
                                datatimeObject = utc_timezone.localize(datatimeObject)
                            else:
                                datatimeObject = datatimeObject.astimezone(utc_timezone)

                            secondsElapsed = (current_utc_time - datatimeObject).total_seconds()
                        except ValueError as e:
                            print(f"Warning: Could not parse last_attendance_time '{last_attendance_str}': {e}. Treating as first attendance.")
                            secondsElapsed = 999999
                    else:
                        secondsElapsed = 999999

                    # print(f"DEBUG: Seconds Elapsed: {secondsElapsed:.2f}, current modeType: {modeType}")

                    if secondsElapsed > ATTENDANCE_INTERVAL_SECONDS:
                        current_attendance = studentInfo.get('total_attendance', 0)
                        new_attendance = current_attendance + 1
                        current_time_for_db = datetime.now(utc_timezone).isoformat()

                        try:
                            supabase.table('students').update({
                                'total_attendance': new_attendance,
                                'last_attendance_time': current_time_for_db
                            }).eq('id', student_id).execute()

                            studentInfo['total_attendance'] = new_attendance
                            studentInfo['last_attendance_time'] = current_time_for_db
                            # print(f"DEBUG: Attendance updated for {student_id} at {current_time_for_db}")
                            modeType = 2 # Set to "Marked Successfully" mode
                            counter = 0 # Reset counter to immediately show the new mode
                            cv2.waitKey(1)
                        except Exception as e:
                            print(f"Error updating attendance: {e}")
                            modeType = 0
                            counter = 0
                            cvzone.putTextRect(imgBackground, "Update Failed!", (275, 400), colorR=(0,0,255))
                            cv2.imshow("Face Attendance", imgBackground)
                            cv2.waitKey(1)
                            continue
                    else:
                        # If secondsElapsed is NOT > ATTENDANCE_INTERVAL_SECONDS (too soon to mark again)
                        # print(f"DEBUG: Too soon to mark. Was modeType {modeType}, setting to 3.")
                        modeType = 3 # This is the "Already Marked" or "Too Soon" mode
                        counter = 0 # Reset counter to immediately start displaying mode 3

            # Step 3: Handle "Marked Successfully" (modeType 2) and "Already Marked" (modeType 3) durations
            elif modeType == 2: # "Marked Successfully" mode
                counter += 1
                if counter >= 1: # Show "Marked Successfully" for about 1 frames then reset
                    counter = 0
                    modeType = 0
                    studentInfo = {}
                    imgStudent = []
                    # print(f"DEBUG: Resetting from mode 2. New mode: {modeType}")

            elif modeType == 3: # "Already Marked" / "Too Soon" mode
                counter += 1
                if counter >= 2: # Show "Already Marked" for about 2 frames then reset
                    counter = 0
                    modeType = 0
                    studentInfo = {}
                    imgStudent = []
                    # print(f"DEBUG: Resetting from mode 3. New mode: {modeType}")

    else: # No face detected
        if modeType != 0: # Only reset if not already in idle mode
            # print("DEBUG: No face detected. Resetting mode to 0.")
            modeType = 0
            counter = 0
            studentInfo = {}
            imgStudent = [] # Clear student image when no face
    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)

