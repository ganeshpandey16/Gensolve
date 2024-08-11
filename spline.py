import numpy as np
import cv2
from scipy.interpolate import splprep, splev

# Function to calculate a point on a quadratic Bezier curve
def bezier_point(t, p0, p1, p2):
    return (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2   

# Function to draw a quadratic Bezier curve 
def draw_bezier_curve(img, p0, p1, p2, color, thickness=2):
    prev_point = p0
    for t in np.linspace(0, 1, 100):
        point = bezier_point(t, p0, p1, p2)
        cv2.line(img, tuple(prev_point.astype(int)), tuple(point.astype(int)), color, thickness)
        prev_point = point

# Function to draw spline curves
def draw_spline_curve(img, points, color, thickness=2):
    if len(points) < 10:
        return
    
    # Prepare points for spline interpolation
    points = np.array(points)
    tck, _ = splprep(points.T, s=0)
    u_fine = np.linspace(0, 1, 1000)
    x_fine, y_fine = splev(u_fine, tck)
    
    # Draw the spline curve
    for i in range(len(x_fine) - 1):
        cv2.line(img, (int(x_fine[i]), int(y_fine[i])), 
                      (int(x_fine[i+1]), int(y_fine[i+1])), color, thickness)

# Function to draw symmetry lines
def draw_symmetry_lines(img, contour, shape):
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return

    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
    
    # Draw lines based on the shape symmetry
    if shape in ['Circle', 'Ellipse', 'Square', 'Regular Polygon']:
        # Vertical symmetry line
        cv2.line(img, (cX, 0), (cX, img.shape[0]), (0, 255, 255), 1)
        # Horizontal symmetry line
        cv2.line(img, (0, cY), (img.shape[1], cY), (0, 255, 255), 1)
        
        if shape == 'Regular Polygon':
            # Draw diagonal lines for regular polygons
            cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 255), 1)
            cv2.line(img, (0, img.shape[0]), (img.shape[1], 0), (0, 255, 255), 1)

# Read the image
img = cv2.imread('tc3.png')

# Convert the image to grayscale
imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
_, thrash = cv2.threshold(imgGrey, 240, 255, cv2.THRESH_BINARY)

# Apply morphological operations to reduce noise
kernel = np.ones((5, 5), np.uint8)
imgMorph = cv2.morphologyEx(thrash, cv2.MORPH_CLOSE, kernel)

# Detect contours
contours, _ = cv2.findContours(imgMorph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Get image dimensions
height, width = img.shape[:2]

# Process each contour
for contour in contours:
    # Filter out contours that are too close to the border or too large
    area = cv2.contourArea(contour)
    if area < 100 or area > (height * width * 0.8):
        continue
    
    # Approximate the contour
    epsilon = 0.005 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Draw spline curves
    draw_spline_curve(img, approx[:, 0, :], (0, 0, 255), 2)
    
    # Detect shapes and draw symmetry lines
    if len(approx) == 2:
        print("Straight Line")
        # cv2.putText(img, "Straight Line", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
    elif len(approx) == 3:
        print("Triangle")
        # cv2.putText(img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    elif len(approx) == 4:
        x1, y1, w, h = cv2.boundingRect(approx)
        aspectRatio = float(w) / h
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
            print("Square")
            # cv2.putText(img, "Square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            draw_symmetry_lines(img, contour, 'Square')
        else:
            print("Rectangle")
            # cv2.putText(img, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            draw_symmetry_lines(img, contour, 'Rectangle')
    elif len(approx) >= 5 and len(approx) <= 9:
        print("Regular Polygon")
        # cv2.putText(img, "Regular Polygon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        draw_symmetry_lines(img, contour, 'Regular Polygon')
    elif len(approx) == 10:
        print("Star")
        # cv2.putText(img, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        draw_symmetry_lines(img, contour, 'Regular Polygon')
    else:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (x_center, y_center), (major_axis, minor_axis), angle = ellipse
            aspect_ratio = major_axis / minor_axis
            cv2.ellipse(img, ellipse, (0, 255, 255), 2)
            if 0.95 <= aspect_ratio <= 1.05:
                print("Circle")
                # cv2.putText(img, "Circle", (int(x_center), int(y_center)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
                draw_symmetry_lines(img, contour, 'Circle')
            elif abs(aspect_ratio - 1) > 0.1:
                print("Ellipse")
                # cv2.putText(img, "Ellipse", (int(x_center), int(y_center)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
                draw_symmetry_lines(img, contour, 'Ellipse')

cv2.imshow("Shapes and Symmetry Lines", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
