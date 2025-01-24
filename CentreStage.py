import cv2
import mediapipe as mp
import threading

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def process_frame(frame, face_detection):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    return results

def detect_motion(prev_frame, curr_frame, threshold=25):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return True, (x, y, w, h)
    return False, None

def smooth_transition(curr_crop, prev_crop, alpha=0.2):
    return int(prev_crop + alpha * (curr_crop - prev_crop))

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face_detection:
    frame_results = None
    frame_lock = threading.Lock()
    prev_frame = None
    prev_x1, prev_y1, prev_x2, prev_y2 = 0, 0, 0, 0

    def detection_thread():
        global frame_results
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            with frame_lock:
                frame_results = process_frame(frame, face_detection)

    detection_thread_instance = threading.Thread(target=detection_thread, daemon=True)
    detection_thread_instance.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        with frame_lock:
            results = frame_results

        ih, iw, _ = frame.shape

        if results and results.detections:
            x_min, y_min = iw, ih
            x_max, y_max = 0, 0

            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)

            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            face_width = x_max - x_min
            face_height = y_max - y_min
            crop_x1 = max(center_x - int(face_width * 1.25), 0)
            crop_y1 = max(center_y - int(1.5 * face_height), 0)
            crop_x2 = min(center_x + int(face_width * 1.25), iw)
            crop_y2 = min(center_y + int(1.25 * face_height), ih)
            crop_x1 = smooth_transition(crop_x1, prev_x1)
            crop_y1 = smooth_transition(crop_y1, prev_y1)
            crop_x2 = smooth_transition(crop_x2, prev_x2)
            crop_y2 = smooth_transition(crop_y2, prev_y2)
            prev_x1, prev_y1, prev_x2, prev_y2 = crop_x1, crop_y1, crop_x2, crop_y2
            frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            frame = cv2.resize(frame, (iw, ih))

        elif prev_frame is not None:
            motion_detected, motion_bbox = detect_motion(prev_frame, frame)
            if motion_detected and motion_bbox:
                x, y, w, h = motion_bbox
                crop_x1 = max(x - int(1.25 * w), 0)
                crop_y1 = max(y - int(1.25 * h), 0)
                crop_x2 = min(x + int(1.75 * w), iw)
                crop_y2 = min(y + int(1.75 * h), ih)
                crop_x1 = smooth_transition(crop_x1, prev_x1)
                crop_y1 = smooth_transition(crop_y1, prev_y1)
                crop_x2 = smooth_transition(crop_x2, prev_x2)
                crop_y2 = smooth_transition(crop_y2, prev_y2)
                prev_x1, prev_y1, prev_x2, prev_y2 = crop_x1, crop_y1, crop_x2, crop_y2
                frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                frame = cv2.resize(frame, (iw, ih))

        prev_frame = frame.copy()
        cv2.imshow('Center Stage - Improved', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
