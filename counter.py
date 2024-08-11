import numpy as np
import cv2

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
img = cv2.imread('tc1.png')

# Convert the image to grayscale
imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
_, thrash = cv2.threshold(imgGrey, 240, 255, cv2.THRESH_BINARY)

# Detect contours
contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Get image dimensions
height, width = img.shape[:2]

# Process each contour
for contour in contours:
    # Filter out contours that are too close to the border or too large
    area = cv2.contourArea(contour)
    if area < 100 or area > (height * width * 0.8):
        continue
    
    # Approximate the contour
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Draw contours using Bezier curves
    for i in range(len(approx)):
        p0 = approx[i][0]
        p1 = (approx[i][0] + approx[(i+1) % len(approx)][0]) / 2
        p2 = approx[(i+1) % len(approx)][0]
        draw_bezier_curve(img, p0, p1, p2, (0, 0, 255), 2)

    # Detect shapes and draw symmetry lines
    if len(approx) == 2:
        print("Straight Line")
    elif len(approx) == 3:
        print("Triangle")
    elif len(approx) == 4:
        x1, y1, w, h = cv2.boundingRect(approx)
        aspectRatio = float(w) / h
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
            print("Square")
            draw_symmetry_lines(img, contour, 'Square')
        else:
            print("Rectangle")
            draw_symmetry_lines(img, contour, 'Rectangle')
    elif len(approx) >= 5 and len(approx) <= 9:
        print("Regular Polygon")
        draw_symmetry_lines(img, contour, 'Regular Polygon')
    elif len(approx) == 10:
        print("Star")
        draw_symmetry_lines(img, contour, 'Regular Polygon')  # Treat star as a regular polygon for symmetry
    else:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (x_center, y_center), (major_axis, minor_axis), angle = ellipse
            aspect_ratio = major_axis / minor_axis
            cv2.ellipse(img, ellipse, (0, 255, 255), 2)
            if 0.95 <= aspect_ratio <= 1.05:
                print("Circle")
                draw_symmetry_lines(img, contour, 'Circle')
            elif abs(aspect_ratio - 1) > 0.1:
                print("Ellipse")
                draw_symmetry_lines(img, contour, 'Ellipse')

cv2.imshow("Shapes and Symmetry Lines", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
