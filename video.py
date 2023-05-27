import cv2

face_cascade = cv2.CascadeClassifier('FaceDetectionTraining.xml')

# Open the video capture
video_capture = cv2.VideoCapture('video.mp4')  

while True:
    
    ret, frame = video_capture.read()

    desired_width = 640
    desired_height = 680

    # Resize the frame
    resized_frame = cv2.resize(frame, (desired_width, desired_height))

    # Convert the frame to grayscale
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Faces found
        for (x, y, w, h) in faces:
            # Draw rectangles around the detected faces
            cv2.rectangle(resized_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Write "Human Face" below the border
            cv2.putText(resized_frame, "Human Face", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
   
   # Display the frame with detected faces
    cv2.imshow('Face Recognition', resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
video_capture.release()
cv2.destroyAllWindows()
