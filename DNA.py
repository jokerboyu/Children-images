import cv2
import numpy as np



def match_dna_pattern(parent_pattern, image_path, threshold=0.5):
  
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equa_liz=cv2.equalizeHist(image_gray)
 
    result = cv2.matchTemplate(image_gray, parent_pattern, cv2.TM_CCOEFF_NORMED)
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

    # Example image path
    image_path = r"C:\Users\joe\Documents\Joe\training image\p33.png"

    # Match the patterns in the ima
    is_match, match_value, match_loc = match_dna_pattern(parent_pattern, image_path)

    perca=match_value*100

    # Display the results
    if is_match:
        print("Your DNA was found with a match value of {:.2f}%.".format(perca))
        print("Congratulations")
        visualize_matching(cv2.imread(image_path), parent_pattern, match_loc)
    else:
        print("No match was found.".format(match_value))



if __name__ == "__main__":
    main()

