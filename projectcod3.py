import numpy as np
import cv2

def region_selection(image):
    """
    Determine and cut the region of interest in the input image.
    Parameters:
        image: we pass here the output from canny where we have 
        identified edges in the frame
    """
    mask = np.zeros_like(image) 
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def hough_transform(image):
    """
    Determine and cut the region of interest in the input image.
    Parameter:
        image: grayscale image which should be an output from the edge detector
    """
    rho = 1
    theta = np.pi / 180
    threshold = 20
    minLineLength = 20
    maxLineGap = 500
    return cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                            minLineLength=minLineLength, maxLineGap=maxLineGap)
    
def average_slope_intercept(lines):
    """
    Find the slope and intercept of the left and right lanes of each image.
    Parameters:
        lines: output from Hough Transform
    """
    left_lines = [] 
    left_weights = [] 
    right_lines = [] 
    right_weights = [] 
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane

def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of each line into pixel points.
    Parameters:
        y1: y-value of the line's starting point.
        y2: y-value of the line's end point.
        line: The slope and intercept of the line.
    """
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    """
    Create full length lines from pixel points.
    Parameters:
        image: The input test image.
        lines: The output lines from Hough Transform.
    """
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12, patch_color=[0, 255, 0]):
    """
    Draw lines and green patches onto the input image.
    Parameters:
        image: The input test image (video frame in our case).
        lines: The output lines from Hough Transform.
        color (Default = red): Line color.
        thickness (Default = 12): Line thickness.
        patch_color (Default = green): Color of the patches to fill the area between the lines.
    """
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    
    patch_image = np.zeros_like(image)
    polygon = np.array([line[0] for line in lines if line is not None] +
                       [line[1] for line in reversed(lines) if line is not None])
    cv2.fillPoly(patch_image, [polygon], patch_color)
    
    result = cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)
    result = cv2.addWeighted(result, 1.0, patch_image, 0.5, 0.0)  # Adjust the alpha value to control the transparency of the patch
    
    return result
prev_left_slope = None
prev_right_slope = None

def frame_processor(image):
    """
    Process the input frame to detect lane lines and draw green patches within them.
    Also adds a lane departure warning message if the vehicle deviates from the lane
    or attempts to change lanes.
    Parameters:
        image: image of a road where one wants to detect lane lines
        (we will be passing frames of video to this function)
    """
    global prev_left_slope, prev_right_slope
    
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
    low_t = 50
    high_t = 150
    edges = cv2.Canny(blur, low_t, high_t)
    region = region_selection(edges)
    hough = hough_transform(region)
    lines = lane_lines(image, hough)
    result = draw_lane_lines(image, lines)
    
    # Lane departure warning message
    left_line, right_line = lines
    if left_line is not None and right_line is not None:
        _, left_intercept = left_line[0]
        _, right_intercept = right_line[1]
        lane_width = right_intercept - left_intercept
        frame_width = image.shape[1]
        lane_center = (right_intercept + left_intercept) / 2
        frame_center = frame_width / 2
        deviation = frame_center - lane_center
        deviation_threshold = 50  # Define a deviation threshold
        
        if abs(deviation) > deviation_threshold:
            warning_msg = ""
            cv2.putText(result, warning_msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Lane change warning message
    global prev_left_slope, prev_right_slope
    if left_line is not None and right_line is not None:
        left_slope = (left_line[1][1] - left_line[0][1]) / (left_line[1][0] - left_line[0][0])
        right_slope = (right_line[1][1] - right_line[0][1]) / (right_line[1][0] - right_line[0][0])
        
        if prev_left_slope is not None and prev_right_slope is not None:
            slope_threshold = 0.1  # Define a slope threshold for lane change detection
            if abs(left_slope - prev_left_slope) > slope_threshold or abs(right_slope - prev_right_slope) > slope_threshold:
                lane_change_warning_msg = "Lane Change Detected! Be cautious."
                cv2.putText(result, lane_change_warning_msg, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        prev_left_slope = left_slope
        prev_right_slope = right_slope
    
    return result


    

def main():
    video_path = '/Users/nishantmishra/Desktop/car lan detactioin/test2.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Couldn't open video file.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = frame_processor(frame)
        cv2.imshow('Lane Detection', processed_frame)
        out.write(processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

