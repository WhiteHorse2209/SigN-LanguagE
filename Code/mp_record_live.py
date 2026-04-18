import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

CSV_FILE = 'landmarks_data_2hands.csv'

def ensure_csv_header():
    """Create the CSV with header if it doesn't exist."""
    if not os.path.exists(CSV_FILE):
        header = []
        for i in range(21):
            header.extend([f'h1_x{i}', f'h1_y{i}', f'h1_z{i}'])
        for i in range(21):
            header.extend([f'h2_x{i}', f'h2_y{i}', f'h2_z{i}'])
        header.append('label')
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        print(f"Created new dataset file: {CSV_FILE}")
    else:
        print(f"Appending to existing dataset: {CSV_FILE}")

def record_gesture(label, target_frames=200):
    """Open webcam and record 'target_frames' frames for the given label."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not found!")
        return 0

    collected = 0
    recording = False
    buffer = []

    print(f"\n--- Recording: '{label}' ---")
    print("Position your hand(s) in the frame.")
    print("Press 'r' to START recording, 'q' to finish early.")
    print(f"Target: {target_frames} frames\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            landmarks_row = []
            for hand_lm in results.multi_hand_landmarks[:2]:
                mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
                for lm in hand_lm.landmark:
                    landmarks_row.extend([lm.x, lm.y, lm.z])

            # Pad to 2-hand size
            while len(landmarks_row) < (21 * 3 * 2):
                landmarks_row.extend([0.0, 0.0, 0.0])

            if recording:
                landmarks_row.append(label)
                buffer.append(landmarks_row)
                collected += 1

        # Status overlay
        status = f"RECORDING: {collected}/{target_frames}" if recording else "Press 'r' to record"
        color = (0, 0, 255) if recording else (0, 200, 0)
        cv2.rectangle(frame, (0, 0), (w, 55), (0, 0, 0), -1)
        cv2.putText(frame, f"Sign: {label}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 100), 2)
        cv2.putText(frame, status, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Live Recording - Sign Language", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r') and not recording:
            recording = True
            print("Recording started!")
        elif key == ord('q'):
            break

        if collected >= target_frames:
            print(f"Collected {target_frames} frames. Done!")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save collected data
    if buffer:
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(buffer)
        print(f"Saved {len(buffer)} frames for '{label}' to {CSV_FILE}")

    return len(buffer)

def main():
    ensure_csv_header()

    print("=" * 40)
    print("  LIVE SIGN LANGUAGE DATA RECORDER")
    print("=" * 40)
    print("This tool records YOUR hand signs and adds them to the dataset.")
    print("After recording, re-run mp_trainer.py to update the model.\n")

    while True:
        label = input("Enter the sign name to record (or 'done' to finish): ").strip()
        if label.lower() == 'done':
            break
        if not label:
            continue

        try:
            count = int(input(f"How many frames to record for '{label}'? (default 200): ").strip() or 200)
        except ValueError:
            count = 200

        recorded = record_gesture(label, count)

        again = input(f"\nRecord another sign? (y/n): ").strip().lower()
        if again != 'y':
            break

    print("\n[Done] Recording session finished!")
    print("Now run:  py -3.9 mp_trainer.py  to retrain the model with your new signs.")

if __name__ == "__main__":
    main()
