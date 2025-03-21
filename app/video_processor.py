import cv2

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        cv2.imwrite(f"frame_{frame_count}.jpg", frame)

    cap.release()