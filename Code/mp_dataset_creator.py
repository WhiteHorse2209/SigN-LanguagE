import cv2
import mediapipe as mp
import os
import csv
import numpy as np

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def extract_landmarks(video_path, label):
    cap = cv2.VideoCapture(video_path)
    data = []
    
    print(f"Processing {video_path}...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            landmarks_row = []
            
            # Loop through up to 2 hands detected
            for hand_landmarks in results.multi_hand_landmarks[:2]:
                for lm in hand_landmarks.landmark:
                    landmarks_row.extend([lm.x, lm.y, lm.z])
            
            # If only 1 hand was detected, we must pad the rest with 0s 
            # so the model still receives data for 2 hands (21 joints * 3 coords = 63 values per hand)
            while len(landmarks_row) < (21 * 3 * 2):
                landmarks_row.extend([0.0, 0.0, 0.0])
                
            landmarks_row.append(label)
            data.append(landmarks_row)
    
    cap.release()
    return data

def main():
    # Recursively find all video files
    video_extensions = ('.mp4', '.mov', '.avi', '.MP4', '.MOV')
    all_data = []
    header = []
    
    # Header for Hand 1
    for i in range(21):
        header.extend([f'h1_x{i}', f'h1_y{i}', f'h1_z{i}'])
    
    # Header for Hand 2
    for i in range(21):
        header.extend([f'h2_x{i}', f'h2_y{i}', f'h2_z{i}'])
        
    header.append('label')

    print("Searching for videos in subfolders...")
    
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith(video_extensions):
                video_path = os.path.join(root, file)
                label = os.path.basename(root)
                if label == '.' or label == '':
                    label = os.path.splitext(file)[0]
                
                print(f"Adding video: {video_path} with label: {label}")
                video_data = extract_landmarks(video_path, label)
                all_data.extend(video_data)

    if not all_data:
        print("No landmarks could be extracted from any videos!")
        return

    # Save to CSV
    with open('landmarks_data_2hands.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_data)
        
    print(f"\n[Success] Extracted data from {len(all_data)} frames across all videos.")
    print("Landmarks saved to 'landmarks_data_2hands.csv'")

if __name__ == "__main__":
    main()
