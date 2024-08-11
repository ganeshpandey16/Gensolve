import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cairosvg
import io

def convert_svg_to_png(svg_path):
    try:
        with open(svg_path, "rb") as svg_file:
            svg_data = svg_file.read()
        png_data = cairosvg.svg2png(bytestring=svg_data)
        return Image.open(io.BytesIO(png_data))
    except Exception as e:
        print(f"Error converting SVG to PNG: {e}")
        return None

def identify_and_complete_shapes(image_path):
    # Convert SVG to PNG
    image = convert_svg_to_png(image_path)
    if image is None:
        return

    image_cv = np.array(image)
    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    
    # Thresholding to get binary image
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # Detect contours
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    outer_mask = np.zeros_like(binary_image)
    inner_mask = np.zeros_like(binary_image)

    # Process contours and fit ellipses
    for i, contour in enumerate(contours):
        if len(contour) >= 5:  # cv2.fitEllipse requires at least 5 points
            try:
                # Fit an ellipse to the contour
                ellipse = cv2.fitEllipse(contour)
                
                # Determine if the contour is an outer or inner ellipse
                if hierarchy[0][i][3] == -1:  # -1 indicates no parent (outer ellipse)
                    cv2.ellipse(outer_mask, ellipse, 255, thickness=-1)  # Fill the outer ellipse
                else:
                    cv2.ellipse(inner_mask, ellipse, 255, thickness=-1)  # Fill the inner ellipse

            except cv2.error as e:
                print(f"Error fitting ellipse: {e}")

    # Find missing parts
    missing_outer_parts = cv2.bitwise_xor(binary_image, outer_mask)
    missing_inner_parts = cv2.bitwise_xor(binary_image, inner_mask)

    # Find contours of the missing parts
    outer_missing_contours, _ = cv2.findContours(missing_outer_parts, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inner_missing_contours, _ = cv2.findContours(missing_inner_parts, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the missing parts on the original image
    for contour in outer_missing_contours:
        cv2.drawContours(image_cv, [contour], -1, (0, 255, 0), thickness=1)  # Green color for missing outer parts

    for contour in inner_missing_contours:
        cv2.drawContours(image_cv, [contour], -1, (255, 0, 0), thickness=1)  # Red color for missing inner parts

    # Draw completed ellipses on the original image
    for contour in contours:
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                if hierarchy[0][i][3] == -1:
                    cv2.ellipse(image_cv, ellipse, (0, 255, 0), 2)  # Outer ellipses in green
                else:
                    cv2.ellipse(image_cv, ellipse, (255, 0, 0), 2)  # Inner ellipses in red
            except cv2.error as e:
                print(f"Error fitting ellipse: {e}")

    # Show the image with completed shapes
    plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def main():
    svg_path = "C:\\Users\\ishit\\OneDrive\\Desktop\\problems\\occlusion2.svg"
    identify_and_complete_shapes(svg_path)

if _name_ == "_main_":
    main()
