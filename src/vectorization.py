import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon
from PIL import Image
import cv2

def create_geometry(image):
    
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Creates list to store polygons
    polygons = []

    # Iterate over all found contours
    for contour in contours:
        # Check if the contour has enough points
        if len(contour) >= 4:
            # Transform the contour into a list of points
            points = [point[0] for point in contour]

            # Create shapely polygon object
            polygon = Polygon(points)
            
            # Add polygon to the list
            polygons.append(polygon)
    
    # Create a GeoDataFrame with the polygons
    gdf = gpd.GeoDataFrame(geometry=polygons)

    gdf.to_file('shapefile.geojson', driver='GeoJSON')
    
    return gdf, contours

def generate_image(original_image, image):
    # Load the GeoJSON file with vector geometries
    gdf, contours = create_geometry(image)

    # Plot image
    _, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(original_image)

    # Plot the vector geometries on the image
    for geom in gdf['geometry']:
        if geom.geom_type == 'Polygon':
            # Convert coordinates into a format accepted by patches.Polygon
            xy = [(x, y) for x, y in zip(*geom.exterior.xy)]
            patch = patches.Polygon(xy, edgecolor='none', facecolor='r', alpha=0.5) 
            ax.add_patch(patch)
    
    plt.axis('off')        
    plt.savefig('./result/vectorized.tif', format='tiff', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
    
    count_trees()
    
    return contours

def count_trees():
    gdf = gpd.read_file("./shapefile.geojson")
    print(f"Árvores encontradas: {len(gdf)}")
    return len(gdf)
    