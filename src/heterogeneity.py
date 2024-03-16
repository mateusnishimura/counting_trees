import cv2
import numpy as np

def get_heterogeneity(polygons):

    areas = [cv2.contourArea(contour) for contour in polygons]

    # Find the maximum area value
    max_area = max(areas)

    # Normalize the areas
    normalized_areas = [area / max_area for area in areas]

    # Get the standard deviation
    heterogeneity = np.std(normalized_areas)

    print(f"Heterogeneidade: {heterogeneity}")