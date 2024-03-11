import cv2
import os

def match_dna_pattern(parent_pattern, image_path, threshold=0.5):
    # Read the image
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize template to match the expected size
    h, w = image_gray.shape
    parent_pattern_resized = cv2.resize(parent_pattern, (w, h))

    # Perform pattern matching using template matching
    result = cv2.matchTemplate(image_gray, parent_pattern_resized, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # Check if the match exceeds the threshold
    if max_val >= threshold:
        return True, max_val, max_loc
    else:
        return False, max_val, None

def visualize_matching(image, parent_pattern, match_loc):
    h, w = parent_pattern.shape[:2]

    # Draw a rectangle around the matched region
    top_left = match_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Image with Pattern Matching", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def match_images_in_folder(parent_pattern, folder_path, threshold=0.5):
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and any(file_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
            # Match the patterns in each image
            is_match, match_value, match_loc = match_dna_pattern(parent_pattern, file_path)

            # Display the results
            print(f"Checking {filename}...")
            if is_match:
                print(f"The pattern is found in {filename} with a match value of {match_value:.2f}.")
                visualize_matching(cv2.imread(file_path), parent_pattern, match_loc)
            else:
                print(f"No pattern found in {filename}. Match value: {match_value:.2f}")

def main():
    # Example DNA pattern (convert your pattern image to a grayscale numpy array)
    parent_pattern = cv2.imread("C:\\Users\\EM\\Downloads\\dna images\\training image\\mth.png", cv2.IMREAD_GRAYSCALE)

    # Example folder path
    folder_path = "C:\\Users\\EM\\Downloads\\dna images\\training image"

    # Match the patterns in images in the folder
    match_images_in_folder(parent_pattern, folder_path)

if _name_ == "_main_":
    main()
