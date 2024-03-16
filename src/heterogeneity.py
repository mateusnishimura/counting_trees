import cv2
import numpy as np

def get_heterogeneity(polygons):
    """
    Calculates the heterogeneity of areas represented by a list of contours.

    Parameters:
    polygons (list): A list of contours representing areas.

    Returns:
    float: The heterogeneity value.
    """
    areas = [cv2.contourArea(contour) for contour in polygons]

    # Find the maximum area value
    max_area = max(areas)

    # Normalize the areas
    normalized_areas = [area / max_area for area in areas]

    # Get the standard deviation
    heterogeneity = np.std(normalized_areas)

    print(f"Heterogeneidade: {heterogeneity}")