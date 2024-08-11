from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/output/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

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

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # Process the image
            img = cv2.imread(file_path)
            imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thrash = cv2.threshold(imgGrey, 240, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            height, width = img.shape[:2]
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100 or area > (height * width * 0.8):
                    continue
                
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                for i in range(len(approx)):
                    p0 = approx[i][0]
                    p1 = (approx[i][0] + approx[(i+1) % len(approx)][0]) / 2
                    p2 = approx[(i+1) % len(approx)][0]
                    draw_bezier_curve(img, p0, p1, p2, (0, 0, 255), 2)

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
                    draw_symmetry_lines(img, contour, 'Regular Polygon')
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

            output_file_path = os.path.join(app.config['OUTPUT_FOLDER'], file.filename)
            cv2.imwrite(output_file_path, img)
            
            return render_template("display.html", input_image=file.filename, output_image=file.filename)

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)
