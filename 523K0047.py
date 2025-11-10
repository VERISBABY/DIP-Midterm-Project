# Task 1
import cv2
import numpy as np
import math, time
import matplotlib.pyplot as plt
from typing import List, Tuple
from dataclasses import dataclass

def remove_small_components(binary_img: np.ndarray, min_size: int = 30) -> np.ndarray:
    if binary_img is None:
        return binary_img
    _, bin_thresh = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_thresh, connectivity=8)
    
    output = np.zeros_like(bin_thresh)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            output[labels == i] = 255
    return output

def preprocess_frame(frame):
    # Use bilateral filter for better edge preservation
    blurred = cv2.bilateralFilter(frame, 9, 75, 75)

    # Convert to HSV for color segmentation
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    return hsv, blurred

def color_segmentation(hsv_frame):
    # Multi-stage color segmentation with advanced noise reduction

    # Multiple color ranges for robustness
    red_ranges = [
        ([0, 150, 80], [8, 255, 255]),     # Bright red
        ([172, 150, 80], [180, 255, 255]), # Dark red
    ]
    
    blue_ranges = [
        ([100, 150, 50], [130, 255, 255]), # Standard blue
    ]
    
    # Create masks using multiple ranges
    mask_red = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    mask_blue = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    
    for lower, upper in red_ranges:
        mask_red |= cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
    
    for lower, upper in blue_ranges:
        mask_blue |= cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
    
    # Multi-stage morphological processing
    kernels = {
        'small': np.ones((2, 2), np.uint8),
        'medium': np.ones((3, 3), np.uint8),
        'large': np.ones((5, 5), np.uint8)
    }
    
    # Advanced red mask processing
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernels['medium'], iterations=2)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernels['small'], iterations=2)
    mask_red = remove_small_components(mask_red, min_size=400)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernels['medium'], iterations=2)
    
    # Advanced blue mask processing
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernels['medium'], iterations=1)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernels['medium'], iterations=2)
    mask_blue = cv2.medianBlur(mask_blue, 3)
    mask_blue = remove_small_components(mask_blue, min_size=200)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernels['small'], iterations=1)
    
    # Final combined mask with additional filtering
    combined_mask = mask_red | mask_blue
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernels['medium'], iterations=1)
    combined_mask = remove_small_components(combined_mask, min_size=400)
    
    return combined_mask, mask_red, mask_blue

def detect_shapes_on_mask(combined_mask):

    # Apply morphological operations to improve shape detection
    kernel = np.ones((3, 3), np.uint8)
    processed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Remove small noise components
    processed_mask = remove_small_components(processed_mask, min_size=500)
    
    # Find contours
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, processed_mask

def analyze_contour_curvature(contour):
    # Analyze contour curvature to detect wavy edges. Returns True if edges are relatively straight, False if wavy

    if len(contour) < 20:
        return True  # Not enough points to analyze
        
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return True
        
    # Approximate contour to reduce noise
    epsilon = 0.01 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) < 4:
        return True
        
    # Calculate curvature for each point
    curvatures = []
    window_size = 5
    
    for i in range(len(contour)):
        if i < window_size or i >= len(contour) - window_size:
            continue
            
        # Get points in window
        prev_point = contour[i - window_size][0]
        curr_point = contour[i][0]
        next_point = contour[i + window_size][0]
        
        # Calculate vectors
        v1 = curr_point - prev_point
        v2 = next_point - curr_point
        
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            # Calculate angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            
            # High curvature when angle deviates significantly from 180° (straight line)
            curvature = abs(180 - angle)
            curvatures.append(curvature)
    
    if not curvatures:
        return True
        
    avg_curvature = np.mean(curvatures)
    # print(f"Average curvature: {avg_curvature}")

    # If average curvature is too high, it's likely wavy
    return avg_curvature < 150  # Threshold for wavy edges

def analyze_shape_properties(contour):

    perimeter = cv2.arcLength(contour, True)
    if perimeter < 1:
        return "unknown", 0, 0, 0, 0
    
    area = cv2.contourArea(contour)
    
    # Multiple geometric properties
    circularity = (4 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Check for wavy edges and right angles
    has_straight_edges = analyze_contour_curvature(contour)
    
    # Multi-level polygon approximation for better shape detection
    shapes = []
    confidences = []
    
    for epsilon_factor in [0.01, 0.02, 0.03]:
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)
        
        # Bounding rectangle properties
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Convex hull properties
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Rectangle properties
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0
        
        # Shape classification with weighted criteria
        shape = "unknown"
        confidence = 0
        
        # Circular signs (high circularity and solidity)
        if circularity > 0.75 and solidity > 0.85 and vertices >= 8:
            shape = "circle"
            confidence = min(circularity * solidity, 1.0)
        
        # Triangular signs
        triangle_score = 0
        if vertices == 3:
            triangle_score += 0.4
        if solidity > 0.8:
            triangle_score += solidity * 0.4
        if 0.5 <= aspect_ratio <= 2.0:
            triangle_score += 0.2
            
        if triangle_score >= 0.7 and shape == "unknown":
            shape = "triangle"
            confidence = triangle_score

        # Rectangular signs, and has_straight_edges
        if vertices == 4 and solidity > 0.85 and 0.6 <= aspect_ratio <= 2.2 and has_straight_edges:
            shape = "rectangle"
            confidence = solidity
        
        # Octagonal signs
        octagon_score = 0
        if 7 <= vertices <= 9:
            octagon_score += 0.4
        if solidity > 0.8:
            octagon_score += solidity * 0.3
        if 0.8 <= aspect_ratio <= 1.2:
            octagon_score += 0.3
            
        if octagon_score >= 0.7 and shape == "unknown" and area >= 1500:
            shape = "octagon"
            confidence = octagon_score
        
        if shape != "unknown":
            shapes.append(shape)
            confidences.append(confidence)
    
    # Return the shape with highest confidence across different approximations
    if shapes:
        best_idx = np.argmax(confidences)
        return shapes[best_idx], confidences[best_idx], len(cv2.approxPolyDP(contour, 0.02 * perimeter, True)), circularity, solidity
    
    return "unknown", 0, 0, circularity, solidity

def detect_red_border_signs(frame, hsv_frame):

    lower_red1 = np.array([0, 120, 80])    # Lower saturation for border detection
    upper_red1 = np.array([10, 255, 255])
    
    lower_red2 = np.array([160, 80, 60])   # Lower saturation for wider coverage
    upper_red2 = np.array([180, 255, 255])
    
    mask_red = (cv2.inRange(hsv_frame, lower_red1, upper_red1) | 
                cv2.inRange(hsv_frame, lower_red2, upper_red2))
    
    # Enhanced morphological operations for border detection
    kernel_small = np.ones((2, 2), np.uint8)
    kernel_medium = np.ones((3, 3), np.uint8)
    
    # Multi-stage processing for better border detection
    # Remove noise while preserving thin borders
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel_medium, iterations=1)
    # Connect broken border segments
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel_small, iterations=2)
    mask_red = remove_small_components(mask_red, min_size=300)
    # Final smoothing
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel_medium, iterations=1)

    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_borders = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        # Adjusted area range for border signs (can be smaller since it's just borders)
        if area < 1000 or area > 20000:
            continue
            
        # Analyze shape for red border signs
        shape, confidence, _, circularity, _ = analyze_shape_properties(contour)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # More lenient aspect ratio for borders
        if aspect_ratio > 2.5 or aspect_ratio < 0.8:
            continue
            
        # Additional check: verify this is likely a border by analyzing the region
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            continue
            
        # Check if this might be a border by analyzing color distribution
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Count red pixels in the ROI
        red_pixels_roi = cv2.countNonZero(cv2.inRange(hsv_roi, lower_red1, upper_red1) |
                                          cv2.inRange(hsv_roi, lower_red2, upper_red2))
        
        total_pixels_roi = w * h
        if total_pixels_roi == 0:
            continue
            
        red_ratio_roi = red_pixels_roi / total_pixels_roi
        
        # For borders, i expect moderate red ratio (not filling entire area)
        if red_ratio_roi < 0.1 or red_ratio_roi >= 0.8:
            continue
            
        # Special handling for circular borders (prohibition signs)
        if shape == "circle" and confidence >= 0.7:
            # Check if it's likely a circular border by analyzing internal content
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Create a mask for the inner region
            center_x, center_y = w // 2, h // 2
            radius = min(w, h) // 3
            inner_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(inner_mask, (center_x, center_y), radius, 255, -1)
            
            # Check inner region for non-red content
            inner_red_pixels = cv2.countNonZero(
                cv2.bitwise_and(
                    cv2.inRange(hsv_roi, lower_red1, upper_red1) |
                    cv2.inRange(hsv_roi, lower_red2, upper_red2),
                    inner_mask
                )
            )
            
            inner_total_pixels = cv2.countNonZero(inner_mask)
            if inner_total_pixels > 0:
                inner_red_ratio = inner_red_pixels / inner_total_pixels
                # If inner region has low red, it's likely a border
                if inner_red_ratio < 0.3:
                    detected_borders.append((x, y, w, h, shape, confidence))
                    continue
        
        # Accept if it has reasonable shape for border detection
        # Lower confidence threshold for borders since they might be partial
        if shape != "unknown" and confidence >= 0.7:
            detected_borders.append((x, y, w, h, shape, confidence))
        # Fallback for border-like shapes with good circularity
        elif circularity >= 0.7 and area >= 1000:
            detected_borders.append((x, y, w, h, "circle", circularity))
    
    return detected_borders

def detect_traffic_signs(frame):

    # Preprocessing
    hsv, _ = preprocess_frame(frame)
    
    # Enhanced color segmentation
    combined_mask, _, _ = color_segmentation(hsv)
    
    # Improved shape detection on combined mask
    contours, processed_mask = detect_shapes_on_mask(combined_mask)
    
    # Also detect red border signs with shape verification
    red_border_signs = detect_red_border_signs(frame, hsv)

    detected_signs = []
    shape_info = []
    
    print(f"Found {len(contours)} potential regions from color mask")
    print(f"Found {len(red_border_signs)} potential red border signs")
    
    # Process contours from combined mask
    for _, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # Reasonable area filtering
        if area < 1000 or area > 20000:
            continue
            
        # Advanced shape analysis
        shape, confidence, vertices, circularity, solidity = analyze_shape_properties(contour)
        
        # Skip if not a valid traffic sign shape or low confidence
        if shape == "unknown" or confidence < 0.7:  # Slightly lower threshold
            continue
            
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Additional aspect ratio check
        if aspect_ratio > 2.5 or aspect_ratio < 0.4:
            continue
            
        # Color verification in ROI
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            continue
            
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Check colors present in this region with improved blue detection
        red_pixels = cv2.countNonZero(cv2.inRange(hsv_roi, np.array([0, 120, 70]), np.array([10, 255, 255])) |
                                      cv2.inRange(hsv_roi, np.array([170, 120, 70]), np.array([180, 255, 255])))
        blue_pixels = cv2.countNonZero(cv2.inRange(hsv_roi, np.array([100, 150, 60]), np.array([130, 255, 200])))
        
        total_pixels = w * h
        if total_pixels == 0:
            continue
            
        red_ratio = red_pixels / total_pixels
        blue_ratio = blue_pixels / total_pixels
        
        # Determine dominant color
        max_color_ratio = max(red_ratio, blue_ratio)
        dominant_color = "red" if red_ratio > blue_ratio else "blue"
        
        # Require significant color presence
        color_threshold = 0.25
        if max_color_ratio < color_threshold:
            continue
            
        # Validate color-shape combination
        if not validate_color_shape_combination(shape, dominant_color):
            continue
            
        # Final verification with balanced criteria
        if is_likely_traffic_sign(contour, shape):
            detected_signs.append((x, y, w, h))
            shape_info.append((shape, dominant_color, confidence, vertices))
            print(f"Accepted: {shape} {dominant_color} (conf: {confidence:.2f}, vertices: {vertices}, area: {area:.0f}, color_ratio: {max_color_ratio:.2f})")
        else:
            print(f"Rejected by final check: {shape} {dominant_color} (conf: {confidence:.2f})")
    
    # Process red border signs with shape verification
    for x, y, w, h, shape, confidence in red_border_signs:
        # Check for overlap with existing detections
        overlap = False
        for sx, sy, sw, sh in detected_signs:
            if (abs(x - sx) < min(w, sw) and abs(y - sy) < min(h, sh)):
                overlap = True
                break
                
        if not overlap:
            detected_signs.append((x, y, w, h))
            shape_info.append((shape, "red", confidence, 4))
            print(f"Added verified red border sign: {shape} at ({x}, {y}, {w}, {h})")

    return detected_signs, processed_mask, shape_info

def validate_color_shape_combination(shape, color):
    # Validate if the color-shape combination is typical for traffic signs

    # Common traffic sign color-shape combinations
    valid_combinations = {
        "circle": ["red", "blue"],
        "triangle": ["red"],
        "rectangle": ["blue"],
        "octagon": ["red"],
        "border": ["red"]
    }
    
    return shape in valid_combinations and color in valid_combinations[shape]

def is_likely_traffic_sign(contour, shape):
    """
    Balanced verification for traffic sign characteristics
    """
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    
    if perimeter == 0:
        return False
        
    # Compactness check
    compactness = (4 * math.pi * area) / (perimeter * perimeter)
    
    # Minimum bounding rectangle check
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box_area = cv2.contourArea(box.astype(np.int32))
    
    if box_area == 0:
        return False
        
    # Area ratio
    area_ratio = area / box_area
    
    # Balanced criteria for different shapes
    if shape == "circle":
        return compactness > 0.5 and area_ratio > 0.5  # Slightly more lenient
    elif shape == "triangle":
        return compactness > 0.3 and area_ratio > 0.4  # More lenient
    elif shape == "rectangle":
        return compactness > 0.4 and area_ratio > 0.55  # Balanced criteria
    elif shape == "octagon":
        return compactness > 0.6 and area_ratio > 0.6 and area >= 1500
    elif shape == "border":
        return compactness > 0.35 and area_ratio > 0.35
    
    return False

def extract_and_plot_frame_processing(input_video, frame_number=130):
    # Extract a specific frame and show detailed processing analysis

    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print("Error: Could not open input video")
        return
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    if not ret:
        print(f"Error: Could not read frame {frame_number}")
        cap.release()
        return
    
    # Process the frame
    detected_signs, combined_mask, shape_info = detect_traffic_signs(frame)
    
    # Create visualization
    frame_with_detections = frame.copy()
    frame_with_contours = frame.copy()
    
    # Draw all contours found in the mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame_with_contours, contours, -1, (255, 0, 0), 2)
    
    # Draw only the detected signs with color coding
    color_map = {"red": (0, 0, 255), "blue": (255, 0, 0)}
    
    for i, (x, y, w, h) in enumerate(detected_signs):
        shape, color, confidence, vertices = shape_info[i]
        box_color = color_map.get(color, (0, 255, 0))
        
        # Draw bounding box
        cv2.rectangle(frame_with_detections, (x, y), (x + w, y + h), box_color, 2)
        cv2.rectangle(frame_with_contours, (x, y), (x + w, y + h), box_color, 2)
        
        # Add label
        label = f'{shape} {color}'
        cv2.putText(frame_with_detections, label, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        cv2.putText(frame_with_contours, label, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
    
    # Convert BGR to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_with_detections_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)
    frame_with_contours_rgb = cv2.cvtColor(frame_with_contours, cv2.COLOR_BGR2RGB)
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Balanced Traffic Sign Detection - Frame {frame_number}', fontsize=16, fontweight='bold')
    
    # Original frame
    axes[0, 0].imshow(frame_rgb)
    axes[0, 0].set_title('Original Frame')
    axes[0, 0].axis('off')
    
    # Color mask
    axes[0, 1].imshow(combined_mask, cmap='gray')
    axes[0, 1].set_title('Enhanced Color Segmentation Mask')
    axes[0, 1].axis('off')
    
    # Frame with all contours and detections
    axes[1, 0].imshow(frame_with_contours_rgb)
    axes[1, 0].set_title('All Contours (Blue) + Valid Detections (Colored)')
    axes[1, 0].axis('off')
    
    # Final detection result
    axes[1, 1].imshow(frame_with_detections_rgb)
    axes[1, 1].set_title(f'Final Detection: {len(detected_signs)} traffic signs')
    axes[1, 1].axis('off')
    
    # Add detection summary
    detection_text = f"Detection Summary:\n- Total regions: {len(contours)}\n- Valid traffic signs: {len(detected_signs)}"
    for i, (shape, color, confidence, vertices) in enumerate(shape_info):
        detection_text += f"\n- Sign {i+1}: {shape} {color} (conf: {confidence:.2f})"
    
    plt.figtext(0.02, 0.02, detection_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.15)
    plt.show()
    
    cap.release()
    return frame_number, len(detected_signs)

def process_video(input_path, output_path):
    # Process the input video and create output video with traffic sign detection

    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print("Error: Could not open input video")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {input_path}")
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    frame_count = 0
    detection_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Detect traffic signs in the current frame
        traffic_signs, _, shape_info = detect_traffic_signs(frame)
        
        # Draw rectangles around detected traffic signs with color coding
        # color_map = {"red": (0, 0, 255), "blue": (255, 0, 0)}
        
        for i, (x, y, w, h) in enumerate(traffic_signs):
            # shape, color, confidence, vertices = shape_info[i]
            # box_color = color_map.get(color, (0, 255, 0))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            detection_count += 1
        
        # Insert Student ID
        cv2.rectangle(frame, (10, 10), (250, 50), (0, 0, 0), -1)
        cv2.putText(frame, '523K0047', (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add frame counter
        # cv2.putText(frame, f'Frame: {frame_count}', (width - 150, 30), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write the processed frame
        out.write(frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames, Detections: {detection_count}")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Processing completed! Output saved as: {output_path}")
    print(f"Total frames processed: {frame_count}")
    print(f"Total detections: {detection_count}")

def main_task1():
    
    input_video = "task1.mp4"
    output_video = "523K0047.avi"
    
    try:
        # First, extract and plot frame processing analysis
        # print("Extracting frame for processing analysis...")
        # khung: 2500, 210, 100, 1800, 1180, 1190
        # frame_num, detections = extract_and_plot_frame_processing(input_video, frame_number=1180)
        # print(f"Frame analysis completed for frame {frame_num}. Detected {detections} traffic signs.")
        
        # Then process the entire video
        print("\nStarting video processing...")
        process_video(input_video, output_video)
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

# Task 2

@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    area: float
    contour: np.ndarray = None
    
    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def coordinates(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

class ProfessionalDigitDetector:    
    def __init__(self, student_id: str = '523K0047'):
        self.student_id = student_id
        
        self.params = {
            'min_area': 60,           # baseline minimum area to filter tiny noise
            'max_area': 2500,         
            'min_aspect_ratio': 0.2,
            'max_aspect_ratio': 1.2,
            'min_width': 8,
            'min_height': 15,
            'gaussian_kernel': (3, 3),
            'adaptive_block_size': 31,
            'adaptive_c': 6,
            # fallback min connected component size
            'fallback_min_cc': 40
        }
        
    def load_image(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return img

    # xóa đường kẻ ngang
    def remove_lines(self, img: np.ndarray) -> np.ndarray:
        result = img.copy()
        
        # Threshold the image to get binary image
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Define kernels for line detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        # Detect horizontal lines
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Detect vertical lines  
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Combine horizontal and vertical lines
        lines_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
        
        # Dilate the lines mask to ensure complete removal
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        lines_mask = cv2.dilate(lines_mask, dilate_kernel, iterations=1)
        
        # Use inpainting to remove lines while preserving the background
        result = cv2.inpaint(result, lines_mask, 3, cv2.INPAINT_TELEA)
        
        return result
    
    def advanced_preprocessing(self, img: np.ndarray) -> np.ndarray:

        img = self.remove_lines(img)

        denoised = cv2.GaussianBlur(img, (3,3), 0)
        denoised = cv2.GaussianBlur(img, (9,9), 0)

        # Apply CLAHE for contrast enhancement (helps with uneven lighting)
        clahe = cv2.createCLAHE(clipLimit=19.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Adaptive thresholding for uneven lighting
        binary_adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, self.params['adaptive_block_size'], self.params['adaptive_c']
        )
        
        # Otsu threshold
        _, binary_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        combined_binary = cv2.bitwise_and(binary_adaptive, binary_otsu)
        
        return combined_binary

    def smart_morphological_processing(self, binary: np.ndarray):
        processed = binary.copy()

        # Estimate an average stroke width using distance transform
        # foreground must be non-zero (digits are white) for distanceTransform
        fg = (processed > 0).astype(np.uint8) * 255
        # small safety if no foreground
        if np.count_nonzero(fg) == 0:
            return processed

        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        processed = self.remove_small_components(processed, min_size=50)
        processed = cv2.erode(processed, kernel1, iterations=1)
        processed = self.remove_small_components(processed, min_size=150)
        processed = cv2.erode(processed, kernel2, iterations=1)
        processed = self.remove_small_components(processed, min_size=150)
        processed = cv2.dilate(processed, kernel2, iterations=1)

        processed = self.remove_small_components(processed, min_size=270)

        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        processed = cv2.dilate(processed, kernel3, iterations=1)

        return processed

    def remove_small_components(self, binary_img: np.ndarray, min_size: int = 30) -> np.ndarray:
        # Remove small connected components (noise)
        if binary_img is None:
            return binary_img
        # Ensure binary in 0/255
        _, bin_thresh = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_thresh, connectivity=8)
        output = np.zeros_like(bin_thresh)
        for i in range(1, num_labels):  # skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_size:
                output[labels == i] = 255
        return output

    def contour_analysis(self, processed_img: np.ndarray) -> List[BoundingBox]:
        # Remove small noise components first
        cleaned_img = self.remove_small_components(processed_img, min_size=85)
        
        contours, _ = cv2.findContours(cleaned_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        bounding_boxes = []
        img_height, img_width = processed_img.shape
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Area-based filtering
            if not (self.params['min_area'] <= area <= self.params['max_area']):
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # Size-based filtering
            if w < self.params['min_width'] or h < self.params['min_height']:
                continue
                
            # Aspect ratio validation for digits
            aspect_ratio = w / h if h > 0 else 0
            if not (self.params['min_aspect_ratio'] <= aspect_ratio <= self.params['max_aspect_ratio']):
                continue
            
            # Solidity check (area vs convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            if solidity < 0.2:  # Too fragmented
                continue
                
            # Extent check (area vs bounding box area)
            rect_area = w * h
            extent = area / rect_area if rect_area > 0 else 0
            
            if extent < 0.10:  # Too sparse
                continue
                
            # Position check - avoid very small components too close to image borders
            border_margin = 3
            if (x < border_margin or y < border_margin or 
                x + w > img_width - border_margin or 
                y + h > img_height - border_margin):
                # allow if reasonably large
                if area < 120:
                    continue
                
            bounding_boxes.append(BoundingBox(x, y, w, h, area, contour))
        
        # Sort by reading order (left to right, top to bottom)
        bounding_boxes.sort(key=lambda b: (b.y // 30, b.x))
        
        return bounding_boxes
    
    def create_frame(self, image: np.ndarray, 
                                bounding_boxes: List[BoundingBox]) -> np.ndarray:
        
        # Frame dimensions
        top_bar_height = 80
        bottom_bar_height = 60
        side_margin = 40
        frame_color = (240, 240, 240)  # Light gray
        
        # Convert to color if grayscale
        if len(image.shape) == 2:
            display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            display_img = image.copy()
        
        # Create frame with extended canvas
        img_height, img_width = display_img.shape[:2]
        frame_height = img_height + top_bar_height + bottom_bar_height
        frame_width = img_width + 2 * side_margin
        
        # Create frame
        frame = np.full((frame_height, frame_width, 3), frame_color, dtype=np.uint8)
        
        # Place original image in center
        y_offset = top_bar_height
        x_offset = side_margin
        frame[y_offset:y_offset+img_height, x_offset:x_offset+img_width] = display_img
        
        # Draw bounding boxes on the image in the frame
        for i, bbox in enumerate(bounding_boxes, 1):
            # Adjust coordinates for frame offset
            x1 = bbox.x + x_offset
            y1 = bbox.y + y_offset
            x2 = x1 + bbox.width
            y2 = y1 + bbox.height
            
            # Color selection: green for typical valid detections
            color = (0, 180, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Stamp student ID on the top bar (right aligned)
        id_text = f"ID: {self.student_id}"
        text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = frame_width - side_margin - text_size[0]
        text_y = int(top_bar_height * 0.6)
        cv2.putText(frame, id_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,50,50), 2, cv2.LINE_AA)
        
        return frame
    
    def create_detailed_visualization(self, original: np.ndarray,
                                    binary: np.ndarray,
                                    processed: np.ndarray,
                                    final_frame: np.ndarray,
                                    bounding_boxes: List[BoundingBox]):
        
        fig = plt.figure(figsize=(16, 12))
        
        # Create subplots
        gs = plt.GridSpec(3, 4, figure=fig)
        
        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original, cmap='gray')
        ax1.set_title("1. Original Image", fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Binary image
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(binary, cmap='gray')
        ax2.set_title("2. Binary Image\n(Adaptive + Otsu)", fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Processed image
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(processed, cmap='gray')
        ax3.set_title("3. After Morphological\nProcessing", fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # Final result (main)
        ax4 = fig.add_subplot(gs[1:, :2])
        ax4.imshow(cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB))
        ax4.set_title(f"Final Result: {len(bounding_boxes)} Digits Detected", 
                     fontsize=14, fontweight='bold', pad=10)
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def process(self, input_path: str, output_path: str):        
        try:
            original_img = self.load_image(input_path)
            
            binary_img = self.advanced_preprocessing(original_img)
            
            # morphological processing
            processed_img = self.smart_morphological_processing(binary_img)
            # Enhanced contour analysis
            bounding_boxes = self.contour_analysis(processed_img)
            
            # Create frame
            final_frame = self.create_frame(original_img, bounding_boxes)
            
            # Save result
            cv2.imwrite(output_path, final_frame)
            
            # visualization main stages
            # self.create_detailed_visualization(
            #     original_img, binary_img, processed_img, final_frame, bounding_boxes
            # )
            
            return {
                'original': original_img,
                'binary': binary_img,
                'processed': processed_img,
                'final_frame': final_frame,
                'bounding_boxes': bounding_boxes,
                'detection_count': len(bounding_boxes),
                'student_id': self.student_id
            }
            
        except Exception as e:
            print(f"Error during processing: {e}")
            raise

def main():
    INPUT_IMAGE = 'input.png'
    OUTPUT_IMAGE = '523K0047.png'
    STUDENT_ID = '523K0047'
    
    try:
        detector = ProfessionalDigitDetector(student_id=STUDENT_ID)
        
        # Process image
        results = detector.process(INPUT_IMAGE, OUTPUT_IMAGE)
        
        print(f"Detected {results['detection_count']} digit bounding boxes")
        
    except FileNotFoundError as e:
        print(f"File error: {e}")
    except Exception as e:
        print(f"Processing error: {e}")

if __name__ == "__main__":
    main_task1()
    time.sleep(1)
    main()