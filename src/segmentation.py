import cv2

def segmentation(image):
    """
    Performs tree segmentation by binarizing the image.

    Parameters:
    image(numpy.ndarray): Pre-processed image.

    Returns:
    segmentated(numpy.ndarray): Binarized image with trees segmented.
    """
        
    # Apply thresholding to segment
    _, segmentated = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    
    # Function to remove shadow from large tree (sample2) and small points, counted as trees
    segmentated = remove_shadow(segmentated)
    
    cv2.imwrite(f"./segmentation_results/segmented.tif", segmentated)
    
    return segmentated

def remove_shadow(image):
    """
    Remove small points and big shadow(sample2).

    Parameters:
    image(numpy.ndarray): Segmentated image.

    Returns:
    image(numpy.ndarray): Image without big shadow and small trees.
    """
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    big_shadow = 1000  
    small_points = 50

    # Iterate over the contours
    for contour in contours:
        
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        
        # Filter contours based on size
        if area > big_shadow or area < small_points:

            cv2.drawContours(image, [contour], 0, 0, -1)
            
    
    return image