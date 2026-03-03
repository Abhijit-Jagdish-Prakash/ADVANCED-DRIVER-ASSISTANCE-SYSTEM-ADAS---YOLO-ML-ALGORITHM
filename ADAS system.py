import cv2
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
WIDTH, HEIGHT = 1280, 720
LANE_ALPHA = 0.15 
COLLISION_DIST_THRESHOLD = 12.0 # Meters

class LaneState:
    def __init__(self):
        self.last_left = [300, HEIGHT, 580, 450]
        self.last_right = [1000, HEIGHT, 700, 450]

    def smooth_step(self, new_coords, is_left=True):
        old_coords = self.last_left if is_left else self.last_right
        smoothed = [int(LANE_ALPHA * n + (1 - LANE_ALPHA) * o) for n, o in zip(new_coords, old_coords)]
        if is_left: self.last_left = smoothed
        else: self.last_right = smoothed
        return smoothed

lane_memory = LaneState()

def get_lane_bounds(y, line):
    x1, y1, x2, y2 = line
    if y2 == y1: return x1
    return x1 + (y - y1) * (x2 - x1) / (y2 - y1)

def region_of_interest(img):
    mask = np.zeros_like(img)
    vertices = np.array([[(50, HEIGHT), (WIDTH//2 - 100, 450), (WIDTH//2 + 100, 450), (WIDTH-50, HEIGHT)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

def preprocess_for_weather(img):
    """Enhances visibility for fog and rain conditions."""
    # 1. Convert to HLS to isolate yellow/white lanes better than Grayscale
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:,:,1]
    
    # 2. Apply CLAHE to the L (Lightness) channel to 'see' through fog
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_l = clahe.apply(l_channel)
    
    # 3. Bilateral Filter: Smooths rain noise while keeping lane edges sharp
    blurred = cv2.bilateralFilter(enhanced_l, 9, 75, 75)
    
    # 4. Adaptive Canny
    v = np.median(blurred)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    canny = cv2.Canny(blurred, lower, upper)
    
    return canny

def draw_lane_overlay(img, left_line, right_line, is_danger):
    overlay = img.copy()
    pts = np.array([[(left_line[0], left_line[1]), (left_line[2], left_line[3]),
                     (right_line[2], right_line[3]), (right_line[0], right_line[1])]], dtype=np.int32)
    color = (0, 0, 255) if is_danger else (0, 255, 0)
    cv2.fillPoly(overlay, pts, color)
    return cv2.addWeighted(overlay, 0.25, img, 0.75, 0)

def pipeline(image):
    # Process image with weather-resistant enhancements
    canny = preprocess_for_weather(image)
    cropped = region_of_interest(canny)
    
    # Use higher threshold for Hough lines to avoid rain-streak 'lines'
    lines = cv2.HoughLinesP(cropped, 2, np.pi/180, 100, minLineLength=50, maxLineGap=150)
    
    left_pts, right_pts = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            if -2.0 < slope < -0.5: left_pts.extend([(x1, y1), (x2, y2)])
            elif 0.5 < slope < 2.0: right_pts.extend([(x1, y1), (x2, y2)])

    y_min, y_max = 450, HEIGHT
    # Wrap in try-except or check length to prevent errors when visibility is near zero
    if len(left_pts) >= 2:
        xs, ys = zip(*left_pts)
        p = np.poly1d(np.polyfit(ys, xs, 1))
        lane_memory.smooth_step([p(y_max), y_max, p(y_min), y_min], True)
        
    if len(right_pts) >= 2:
        xs, ys = zip(*right_pts)
        p = np.poly1d(np.polyfit(ys, xs, 1))
        lane_memory.smooth_step([p(y_max), y_max, p(y_min), y_min], False)

    return lane_memory.last_left, lane_memory.last_right

def process_video():
    model = YOLO('yolov8n.pt') 
    cap = cv2.VideoCapture('video/car.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        
        l_line, r_line = pipeline(frame)
        
        # YOLOv8 is already quite robust in rain, but we can lower NMS slightly if needed
        results = model.predict(frame, conf=0.35, classes=[2, 3, 5, 7], verbose=False)
        
        collision_detected = False
        vehicle_boxes = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            car_x_base, car_y_base = (x1 + x2) / 2, y2 
            
            lane_left_limit = get_lane_bounds(car_y_base, l_line)
            lane_right_limit = get_lane_bounds(car_y_base, r_line)
            
            dist = (1.8 * 800) / max((x2 - x1), 1)
            is_in_my_lane = (lane_left_limit - 15) < car_x_base < (lane_right_limit + 15)
            
            is_danger = dist < COLLISION_DIST_THRESHOLD and is_in_my_lane
            if is_danger: collision_detected = True
            vehicle_boxes.append(((x1, y1, x2, y2), dist, is_danger))

        frame = draw_lane_overlay(frame, l_line, r_line, collision_detected)

        for (x1, y1, x2, y2), dist, is_danger in vehicle_boxes:
            color = (0, 0, 255) if is_danger else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{dist:.1f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if collision_detected:
            cv2.rectangle(frame, (0, 0), (WIDTH, 70), (0, 0, 255), -1)
            cv2.putText(frame, "!!! WARNING: VEHICLE IN LANE !!!", (WIDTH//4 + 50, 45), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)

        cv2.imshow('Weather-Resistant ADAS', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()

