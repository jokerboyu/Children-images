import cv2
import numpy as np

def preprocess_image(image):
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred

def segment_bands(image):
    # Perform adaptive thresholding to separate bands from background
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert the thresholded image
    thresh = cv2.bitwise_not(thresh)
    
    return thresh

def extract_bands(image):
    # Find contours in the image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate over detected contours
    bands = []
    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Append the y-coordinate to the list of bands
        bands.append(y)
    
    return bands

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Segment bands
    segmented_image = segment_bands(preprocessed_image)
    
    # Extract y-coordinates of bands
    bands = extract_bands(segmented_image)
    
    return bands

def calculate_matching_score(bands1, bands2):
    # Count the number of y-axes in the first image that equal y-axes in the second image
    num_matching_y_axes = sum(1 for y1 in bands1 if y1 in bands2)
    
    # Calculate the matching score
    matching_score = num_matching_y_axes / len(bands1)
    
    return matching_score

# List of paths to the input images
image_paths = ["C:\\Users\\EM\\Downloads\\dna images\\training image\\child1.png", "C:\\Users\\EM\\Downloads\\dna images\\training image\\father1.png"]


# Process each image and extract y-coordinates of bands
bands_list = []
for image_path in image_paths:
    bands_list.append(process_image(image_path))

# Calculate matching score
matching_score = calculate_matching_score(bands_list[0], bands_list[1])

# Display the matching score
print("Matching Score:", matching_score)
