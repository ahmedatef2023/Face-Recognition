import cv2

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier('FaceDetectionTraining.xml')

# Load the image
image = cv2.imread('a2.bmp')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform face detection
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

if len(faces) > 0:
    # Faces found
    for (x, y, w, h) in faces:
        # Draw rectangles around the detected faces
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Write "Human Face" below the border
        cv2.putText(image, "Human Face", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
else:
    # No faces found
    cv2.putText(image, "Faces not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Display the image with detected faces
cv2.imshow('Face Recognition', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
