import cv2
import os

# Path to the positive and negative samples directories
positive_samples_path = 'input/positiveSamples'
negative_samples_path = 'input/negativeSamples'

# Create a list of paths to positive sample images
positive_images = [os.path.join(positive_samples_path, image) for image in os.listdir(positive_samples_path)]

# Create a list of paths to negative sample images
negative_images = [os.path.join(negative_samples_path, image) for image in os.listdir(negative_samples_path)]

# Path to store the generated positive samples
positive_samples_output_path = 'output/positiveSamples'

# Create the output directory if it doesn't exist
if not os.path.exists(positive_samples_output_path):
    os.makedirs(positive_samples_output_path)

# Generate positive samples
num_samples = 1000000  # Number of positive samples to generate
sample_width = 24  # Width of the generated positive samples
sample_height = 24  # Height of the generated positive samples

for image_path in positive_images:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Run OpenCV's haar cascade to detect faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (sample_width, sample_height))
        
        # Generate a unique filename for each positive sample
        output_filename = os.path.join(positive_samples_output_path, f'pos_{num_samples}.jpg')
        
        # Save the positive sample image
        cv2.imwrite(output_filename, face_resized)
        
        num_samples += 1

# Create a positive samples description file
positive_samples_description_file = 'output/positiveSamples/description.txt'

with open(positive_samples_description_file, 'w') as file:
    for image_path in os.listdir(positive_samples_output_path):
        file.write(f'{os.path.join(positive_samples_output_path, image_path)} 1 0 0 24 24\n')
