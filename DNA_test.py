import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def match_dna_pattern(parent_pattern, image_path, threshold=0.5):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return False, None, None

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equa_liz = cv2.equalizeHist(image_gray)

    result = cv2.matchTemplate(equa_liz, parent_pattern, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        return True, max_val, max_loc
    else:
        return False, max_val, None

def visualize_matching(image, parent_pattern, match_loc):
    h, w = parent_pattern.shape[:2]
    top_left = match_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    cv2.imshow("Image with Pattern Matching", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Example DNA pattern (convert your pattern image to a grayscale numpy array)
    parent_pattern = cv2.imread(r"C:\Users\joe\Documents\Joe\training image\p27.png", cv2.IMREAD_GRAYSCALE)

    # Example dataset (list of image paths)
    dataset = [
        r"C:\Users\joe\Documents\Joe\training image\p1.png",
        r"C:\Users\joe\Documents\Joe\training image\p2.png",
        # Add more image paths as needed
    ]

    # Split the dataset into training and test sets
    train_images, test_images = train_test_split(dataset, test_size=0.2, random_state=42)

    # Match the patterns in the test set images
    for image_path in test_images:
        is_match, match_value, match_loc = match_dna_pattern(parent_pattern, image_path)

        # Display the results
        if is_match:
            print("Your DNA was found with a match value of {:.2f}%.".format(match_value * 100))
            print("Congratulations!")
            visualize_matching(cv2.imread(image_path), parent_pattern, match_loc)
        else:
            print("No match was found in:", image_path)

if __name__ == "__main__":
    main()
