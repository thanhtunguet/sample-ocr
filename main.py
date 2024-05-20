import cv2
import pytesseract
from PIL import Image
import numpy as np

# Path to the tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Update this path based on your installation

def preprocess_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Optionally, apply additional processing like noise reduction
    # blur = cv2.GaussianBlur(thresh, (1, 1), 0)
    
    return thresh

def extract_text_from_image(image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Use pytesseract to extract text
    text = pytesseract.image_to_string(processed_image)
    
    return text

if __name__ == "__main__":
    # Path to the image
    image_path = 'test-03.jpg'  # Update with your image path

    # Extract text
    text = extract_text_from_image(image_path)
    print("Extracted Text:\n", text)
