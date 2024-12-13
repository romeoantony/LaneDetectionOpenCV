import cv2 as cv
import numpy as np
import os
import shutil

def roi(img, vertices):
    mask = np.zeros_like(img)
    cv.fillPoly(mask, vertices, 255) 
    masked_image = cv.bitwise_and(img, mask)
    return masked_image

def draw_hough_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(blank_image, (x1, y1), (x2, y2), (190, 150, 0), thickness=5)
    img = cv.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def detect(real_image):
    real_image = cv.resize(real_image, (640, 480))

    # Convert BGR to HSV
    hsv = cv.cvtColor(real_image, cv.COLOR_BGR2HSV)
    
    # Define range of yellow and white color in HSV
    lower_yellow = np.array([12, 100, 100])
    upper_yellow = np.array([70, 255, 255])
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([255, 30, 255])
    
    # Threshold the HSV image to get only yellow and white colors
    mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv.inRange(hsv, lower_white, upper_white)
    mask_lane = cv.bitwise_or(mask_yellow, mask_white)
    
    masked1 = cv.bitwise_or(mask_yellow, mask_white)
    masked = cv.bitwise_or(masked1, mask_lane)
    
    # Define region of interest
    height = real_image.shape[0]
    width = real_image.shape[1]
    
    roi_vertices = np.array([[(0, height), (3*width/10, 6*height/10),
                              (7*width/10, 6*height/10), (width, height)]], dtype=np.int32)
    cropped = roi(masked, roi_vertices)
    
    blur = cv.GaussianBlur(cropped, (5, 5), 0)
    
    # Create a horizontal kernel
    kernel = np.ones((1, 20), np.uint8)

    # Dilate the image
    dilated_image = cv.dilate(blur, kernel, iterations=1)

    # Erode the dilated image
    eroded_image = cv.erode(dilated_image, kernel, iterations=1)
    
    cropped = eroded_image
    
    edges = cv.Canny(cropped, 50, 150)

    lines = cv.HoughLinesP(edges, rho=5, theta=np.pi/180, threshold=100,
                            lines=np.array([]), minLineLength=30, maxLineGap=100)

    if lines is not None:
        image_with_lines = draw_hough_lines(real_image, lines)
    else:
        image_with_lines = real_image
        
    return image_with_lines

def main():
    video_path = 'input1.mp4'
    video = cv.VideoCapture(video_path)
    length = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    input_frames_path = 'input_frames/'
    output_frames_path = 'output_frames/'

    shutil.rmtree(input_frames_path, ignore_errors=True)
    shutil.rmtree(output_frames_path, ignore_errors=True)
    os.makedirs(input_frames_path, exist_ok=True)
    os.makedirs(output_frames_path, exist_ok=True)

    count = 0
    while True:
        success, image = video.read()
        if not success:
            break
        cv.imwrite((f'{input_frames_path}{count}.png'), image)
        
        output_image = detect(image)
        cv.imwrite((f'{output_frames_path}{count}.png'), output_image)
        
        count += 1
        progress = 100 * count // length
        print(f'\rProgress: [{"="*progress}{" "*(100-progress)}] {progress}%', end='', flush=True)

    video.release()

    img = [cv.imread(f'output_frames/{i}.png') for i in range(length) if os.path.isfile(f'output_frames/{i}.png')]
    if len(img) < 1:
        exit()
    

    height, width, layers = img[0].shape

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video = cv.VideoWriter('output_video.mp4', fourcc, 20, (width, height))

    for j in range(length):
        video.write(img[j])

    cv.destroyAllWindows()
    video.release()
    print('Done')

if __name__ == "__main__":
    main()
