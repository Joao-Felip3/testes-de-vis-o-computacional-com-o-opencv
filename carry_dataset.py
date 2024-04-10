import cv2
import os

def extract_frames_from_webcam(output_folder, num_frames):
    #recebe o caminho da pasta onde os frames serão guardados e o nuúmero de frames para capturar 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open webcam
    cap = cv2.VideoCapture(0)  
    # usa a webcam para fazer as capturas
    frame_count = 0

    # Loop
    while frame_count < num_frames:
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to capture frame")
            break

        #salva o frame como uma imagem
        frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)

        frame_count += 1

        # Display the frame (optional)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Total frames captured: {frame_count}")


output_folder = 'folder path'  


extract_frames_from_webcam(output_folder, num_frames=1000)
