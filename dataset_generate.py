import cv2
import os

def extract_all_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print("Error: Could not open the video file.")
        return
    
    frame_count = 0
    success = True
    
    while success:
        success, frame = video.read()
        
        if success:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
    
    video.release()
    print(f"Extraction complete! {frame_count} frames saved in '{output_folder}'.")
video_path = ".\Video\play_num_3.mp4" 
output_folder = ".\\all_frames_3"
extract_all_frames(video_path, output_folder)
