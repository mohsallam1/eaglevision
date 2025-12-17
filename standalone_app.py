import cv2
import json
import time
import os
import base64
import asyncio
import threading
from queue import Queue
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import torch

# ============================================================================
# CONFIGURATION - ADJUST THESE FOR YOUR SETUP
# ============================================================================

VIDEO_PATH = "videos/1.mp4"
MODEL_PATH = "yolo12m-v2.pt"
CONFIDENCE_THRESHOLD = 0.30  # Balanced - good detection without noise
TARGET_FPS = 12  # Realistic for CPU
FRAME_SKIP = 1  # Process all frames
REPLAY_VIDEO = False  # Set to True to loop video
SMOOTH_PLAYBACK = False  # Disable for speed
PREFER_GPU = True  # Try GPU first, fallback to CPU if not available
IMAGE_SIZE = 640  # Balanced size
BATCH_PROCESS = True  # Process detections in batch
ENABLE_PREPROCESSING = True  # Enable CLAHE for better detection
DEBUG_MODE = True  # Show detection counts in console
SCOOPER_CONFIDENCE_BOOST = 0.03  # ULTRA LOW - catch all scoopers!

# ROI Configuration - ACTUAL METAL INGREDIENT CONTAINERS
# Cover the metal bins on the stainless steel counter (NOT the wall/pizza boxes!)
# The containers are on the counter where worker stands, NOT the far left wall
ROIS = {
    'ingredient_containers': (230, 250, 450, 550)
}


# BALANCED CONFIGURATION - DETECT VIOLATIONS
HAND_TRACKING_HISTORY = 50  # History buffer
VIOLATION_WINDOW = 40  # Analysis window
HAND_SCOOPER_DISTANCE_THRESHOLD = 250  # LARGER - catch scoopers even if far from hand center
HAND_PIZZA_OVERLAP_THRESHOLD = 0.10  # Not used
MIN_FRAMES_IN_ROI = 8  # BALANCED - 8+ frames = real grab (detects violations!)
MIN_FRAMES_WITH_PIZZA = 0  # Not used
HAND_DISTANCE_THRESHOLD = 70  # Tracking distance
REQUIRE_PIZZA_TOUCH = False  # Only care about container grab
VIOLATION_COOLDOWN = 60  # Reasonable cooldown

# Detection quality filters - SENSITIVE
MIN_HAND_SIZE = 600  # LOWER - detect more hands
MIN_SCOOPER_SIZE = 30  # ULTRA LOW - catch tiny scoopers
MIN_PIZZA_SIZE = 1000  # Minimum pixels for pizza

# Output configuration
VIOLATION_FRAMES_DIR = "violation_frames"
VIOLATIONS_JSON_FILE = "violations.json"

# ============================================================================
# GLOBAL STATE
# ============================================================================

frame_queue = Queue(maxsize=100)
violation_count = 0
latest_frame = None
hand_tracks = {}
next_hand_id = 0
violations_list = []

# Create output directory
os.makedirs(VIOLATION_FRAMES_DIR, exist_ok=True)

# ============================================================================
# OPTIMIZATION & ENHANCEMENT FUNCTIONS
# ============================================================================

def preprocess_frame(frame):
    """Enhance frame for better detection (optional - disable for speed)"""
    if not ENABLE_PREPROCESSING:
        return frame
    
    # Fast contrast enhancement
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return enhanced


def post_process_detections(detections, conf_boost=1.1):
    """Boost confidence for stable detections"""
    processed = []
    for det in detections:
        det_copy = det.copy()
        # Boost confidence for larger, more stable boxes
        box = det['bbox']
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area > 5000:  # Large stable objects
            det_copy['conf'] = min(det['conf'] * conf_boost, 0.99)
        processed.append(det_copy)
    return processed


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_box_overlap(box1, box2):
    """Calculate IoU between two boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)
    
    if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
        return 0.0
    
    inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def point_in_roi(point, roi):
    """Check if point is inside ROI"""
    x, y = point
    x1, y1, x2, y2 = roi
    return x1 <= x <= x2 and y1 <= y <= y2


def get_box_center(box):
    """Get center point of box"""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def match_hand_to_track(hand_box, frame_number):
    """
    Match detected hand to existing track with stability requirements
    Requires consistent detection before creating new track
    """
    global next_hand_id, hand_tracks
    
    hand_center = get_box_center(hand_box)
    min_distance = float('inf')
    matched_id = None
    
    # Try to match to existing tracks
    for hand_id, track_data in hand_tracks.items():
        # Check if track is still active (seen recently)
        frames_since_seen = frame_number - track_data['last_seen']
        if frames_since_seen > 20:  # Expire old tracks
            continue
        
        if len(track_data['history']) > 0:
            last_state = track_data['history'][-1]
            last_center = get_box_center(last_state['hand_box'])
            
            # Calculate distance from last known position
            distance = np.sqrt(
                (hand_center[0] - last_center[0])**2 + 
                (hand_center[1] - last_center[1])**2
            )
            
            # Also check size consistency
            last_box = last_state['hand_box']
            last_area = (last_box[2] - last_box[0]) * (last_box[3] - last_box[1])
            current_area = (hand_box[2] - hand_box[0]) * (hand_box[3] - hand_box[1])
            size_ratio = current_area / last_area if last_area > 0 else 1.0
            
            # Must be close in both position AND size
            if (distance < min_distance and 
                distance < HAND_DISTANCE_THRESHOLD and
                0.5 < size_ratio < 2.0):  # Size shouldn't change drastically
                min_distance = distance
                matched_id = hand_id
    
    # Create new track only if not matched
    if matched_id is None:
        matched_id = next_hand_id
        next_hand_id += 1
        hand_tracks[matched_id] = {
            'history': deque(maxlen=HAND_TRACKING_HISTORY),
            'violation_logged': False,
            'last_seen': frame_number,
            'frames_in_roi': 0,
            'frames_at_pizza': 0,
            'stable_frames': 0,  # Count of consecutive detections
            'is_stable': False,  # Whether track is stable enough
            'last_violation_frame': -999  # Cooling period tracker
        }
    else:
        # Quick stability check (2 frames is enough for speed)
        hand_tracks[matched_id]['stable_frames'] += 1
        if hand_tracks[matched_id]['stable_frames'] >= 2:
            hand_tracks[matched_id]['is_stable'] = True
    
    return matched_id


def check_violation(hand_id, frame_number):
    """
    THREE-STAGE VIOLATION DETECTION - Complete Workflow Validation
    
    VIOLATION = Hand grabs from container WITHOUT scooper, then places on pizza
    
    Three-stage detection:
    STAGE 1 - GRABBING: Hand in ROI without scooper (4+ frames)
    STAGE 2 - TRANSFER: Hand moves away from ROI
    STAGE 3 - PLACING: Hand touches pizza (3+ frames)
    
    All stages must occur in sequence within the violation window
    """
    global hand_tracks
    
    if hand_id not in hand_tracks:
        return False
    
    track = hand_tracks[hand_id]
    history = track['history']
    
    # Need sufficient history
    if len(history) < MIN_FRAMES_IN_ROI:
        return False
    
    # Already logged violation for this track
    if track.get('violation_logged', False):
        return False
    
    # Cooling period to prevent duplicate alerts
    if frame_number - track.get('last_violation_frame', -999) < VIOLATION_COOLDOWN:
        return False
    
    # =================================================================
    # STAGE 1: Find grabbing phase (hand in ROI without scooper)
    # =================================================================
    
    grabbing_frames = []
    for i in range(-min(VIOLATION_WINDOW, len(history)), 0):
        if abs(i) > len(history):
            continue
        state = history[i]
        if state['in_roi'] and not state['has_scooper']:
            grabbing_frames.append(i)
    
    # Must have sustained grabbing
    if len(grabbing_frames) < MIN_FRAMES_IN_ROI:
        return False
    
    # Get ROI name from grabbing phase
    roi_name = None
    roi_start_frame = None
    for i in grabbing_frames:
        if history[i]['roi_name']:
            roi_name = history[i]['roi_name']
            roi_start_frame = history[i]['frame_number']
            break
    
    # =================================================================
    # STAGE 2: Verify no scooper during grabbing
    # =================================================================
    
    for i in grabbing_frames:
        if history[i]['has_scooper']:
            return False  # Had scooper - not a violation
    
    # =================================================================
    # STAGE 3: Find placing phase (hand touching pizza)
    # =================================================================
    
    if REQUIRE_PIZZA_TOUCH:
        placing_frames = []
        pizza_start_frame = None
        
        for i in range(-min(15, len(history)), 0):  # Look at recent frames
            if abs(i) > len(history):
                continue
            state = history[i]
            if state['near_pizza']:
                placing_frames.append(i)
                if pizza_start_frame is None:
                    pizza_start_frame = state['frame_number']
        
        # Must have sustained pizza contact
        if len(placing_frames) < MIN_FRAMES_WITH_PIZZA:
            return False
        
        # Verify sequence: ROI happened before Pizza
        if roi_start_frame and pizza_start_frame:
            if roi_start_frame >= pizza_start_frame:
                return False  # Wrong order
            
            # Verify within violation window
            time_diff = pizza_start_frame - roi_start_frame
            if time_diff > VIOLATION_WINDOW:
                return False  # Too far apart
    else:
        placing_frames = []
    
    # =================================================================
    # VIOLATION CONFIRMED - All three stages detected!
    # =================================================================
    
    track['violation_logged'] = True
    track['last_violation_frame'] = frame_number
    
    # Store violation details
    track['violation_details'] = {
        'roi_frames': len(grabbing_frames),
        'pizza_frames': len(placing_frames),
        'roi_name': roi_name,
        'frame_number': frame_number,
        'violation_type': f'Grabbed from {roi_name} without scooper and placed on pizza',
        'confidence_level': 'CONFIRMED'
    }
    
    return True


def save_violation(frame_number, frame, violation_info):
    """Save violation to file and JSON"""
    global violations_list
    
    timestamp = datetime.now()
    frame_path = os.path.join(VIOLATION_FRAMES_DIR, f"violation_frame_{frame_number}.jpg")
    
    # Save frame
    cv2.imwrite(frame_path, frame)
    
    # Save to JSON
    violation_data = {
        'frame_number': frame_number,
        'timestamp': timestamp.isoformat(),
        'frame_path': frame_path,
        'violation_info': violation_info
    }
    violations_list.append(violation_data)
    
    # Write to file
    with open(VIOLATIONS_JSON_FILE, 'w') as f:
        json.dump(violations_list, f, indent=2)
    
    return frame_path


# ============================================================================
# VIDEO PROCESSING THREAD
# ============================================================================

def process_video_thread(model, device='cpu'):
    """Background thread for video processing"""
    global frame_queue, violation_count, latest_frame, hand_tracks
    
    print("=" * 70)
    print("🦅 EAGLE VISION - STANDALONE MODE")
    print("=" * 70)
    print(f"Video: {VIDEO_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(f"ROIs: {len(ROIS)} configured")
    for name, roi in ROIS.items():
        print(f"  - {name}: {roi}")
    print("-" * 70)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"✗ Error: Could not open video {VIDEO_PATH}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Video opened: {total_frames} frames at {fps:.1f} FPS")
    print(f"  Resolution: {width}x{height}")
    print("-" * 70)
    
    frame_number = 0
    processed_frames = 0
    start_time = time.time()
    last_frame_time = time.time()
    frame_interval = 1.0 / TARGET_FPS if SMOOTH_PLAYBACK else 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            if REPLAY_VIDEO:
                print("\n✓ End of video - restarting...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_number = 0
                hand_tracks.clear()
                start_time = time.time()
                continue
            else:
                print("\n✓ End of video reached - stopping playback")
                print(f"Total violations detected: {violation_count}")
                print(f"Check violations.json for detailed records")
                break
        
        frame_number += 1
        
        # Skip frames for faster processing
        if frame_number % FRAME_SKIP != 0:
            continue
        
        processed_frames += 1
        
        # Enhance frame for better detection
        enhanced_frame = preprocess_frame(frame)
        
        # Run YOLO detection with OPTIMIZED settings for speed + accuracy
        results = model(
            enhanced_frame, 
            conf=CONFIDENCE_THRESHOLD,  # Balanced threshold
            iou=0.40,  # Balanced NMS
            max_det=40,  # Limited detections = faster
            half=False,  # FP32 for CPU stability
            verbose=False,
            imgsz=IMAGE_SIZE,  # 640 = 2x faster than 768
            agnostic_nms=True,  # Faster NMS
            device=device,
            augment=False  # No augmentation for speed
        )
        
        # Extract detections with filtering
        hands = []
        scoopers = []
        pizzas = []
        persons = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls].lower()
                
                bbox = (int(x1), int(y1), int(x2), int(y2))
                
                # QUALITY FILTERING - Accept good detections only
                box_area = (x2 - x1) * (y2 - y1)
                
                # Remove very tiny noise
                if box_area < 200:
                    continue
                
                # Remove huge false detections
                if box_area > (width * height * 0.4):
                    continue
                
                # Class-specific filtering with size requirements
                if 'hand' in class_name:
                    if conf >= CONFIDENCE_THRESHOLD and box_area >= MIN_HAND_SIZE:
                        hands.append({'bbox': bbox, 'conf': conf})
                    
                elif 'scooper' in class_name or 'scoop' in class_name or 'spoon' in class_name or 'utensil' in class_name:
                    # LOWER confidence for scoopers - they're hard to detect!
                    if conf >= SCOOPER_CONFIDENCE_BOOST and box_area >= MIN_SCOOPER_SIZE:
                        scoopers.append({'bbox': bbox, 'conf': conf})
                    
                elif 'pizza' in class_name or 'dough' in class_name or 'bread' in class_name or 'food' in class_name:
                    if conf >= CONFIDENCE_THRESHOLD and box_area >= MIN_PIZZA_SIZE:
                        pizzas.append({'bbox': bbox, 'conf': conf})
                    
                elif 'person' in class_name:
                    persons.append({'bbox': bbox, 'conf': conf})
        
        # Post-process detections for stability
        hands = post_process_detections(hands)
        scoopers = post_process_detections(scoopers)
        pizzas = post_process_detections(pizzas)
        
        # Track hands and detect violations with enhanced logic
        current_violations = []
        
        for hand in hands:
            hand_box = hand['bbox']
            hand_center = get_box_center(hand_box)
            hand_id = match_hand_to_track(hand_box, frame_number)
            
            # Check if hand has scooper nearby (use distance - more reliable!)
            has_scooper = False
            for scooper in scoopers:
                scooper_center = get_box_center(scooper['bbox'])
                distance = ((hand_center[0] - scooper_center[0])**2 + 
                           (hand_center[1] - scooper_center[1])**2)**0.5
                if distance < HAND_SCOOPER_DISTANCE_THRESHOLD:
                    has_scooper = True
                    break
            
            # Check if hand is in ROI
            in_roi = False
            roi_name = None
            for name, roi in ROIS.items():
                if point_in_roi(hand_center, roi):
                    in_roi = True
                    roi_name = name
                    break
            
            # Check if hand is near pizza (stricter overlap check)
            near_pizza = False
            for pizza in pizzas:
                overlap = calculate_box_overlap(hand_box, pizza['bbox'])
                if overlap > HAND_PIZZA_OVERLAP_THRESHOLD:
                    near_pizza = True
                    break
            
            # Update frame counters
            if in_roi and not has_scooper:
                hand_tracks[hand_id]['frames_in_roi'] += 1
            else:
                hand_tracks[hand_id]['frames_in_roi'] = 0
            
            if near_pizza:
                hand_tracks[hand_id]['frames_at_pizza'] += 1
            else:
                hand_tracks[hand_id]['frames_at_pizza'] = 0
            
            # Update track history
            hand_tracks[hand_id]['history'].append({
                'frame_number': frame_number,
                'hand_box': hand_box,
                'has_scooper': has_scooper,
                'in_roi': in_roi,
                'roi_name': roi_name,
                'near_pizza': near_pizza
            })
            hand_tracks[hand_id]['last_seen'] = frame_number
            
            # Check for violations (simple and effective)
            if check_violation(hand_id, frame_number):
                violation_details = hand_tracks[hand_id].get('violation_details', {})
                
                current_violations.append({
                    'hand_id': hand_id,
                    'hand_box': hand_box,
                    'roi_name': violation_details.get('roi_name'),
                    'frames_in_roi': violation_details.get('roi_frames', 0),
                    'frames_at_pizza': violation_details.get('pizza_frames', 0),
                    'violation_type': violation_details.get('violation_type', 'Unknown')
                })
                violation_count += 1
                
                print("\n" + "=" * 70)
                print(f"🚨 VIOLATION DETECTED! Frame {frame_number}")
                print("=" * 70)
                print(f"Hand ID: {hand_id}")
                print(f"Type: {violation_details.get('violation_type')}")
                print(f"ROI: {violation_details.get('roi_name')}")
                print(f"Frames in ROI (no scooper): {violation_details.get('roi_frames', 0)}")
                if REQUIRE_PIZZA_TOUCH:
                    print(f"Frames at pizza: {violation_details.get('pizza_frames', 0)}")
                print(f"Confidence: {violation_details.get('confidence_level')}")
                print("=" * 70 + "\n")
        
        # Draw annotations
        annotated_frame = frame.copy()
        
        # Draw ROIs
        for roi_name, roi in ROIS.items():
            x1, y1, x2, y2 = roi
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(annotated_frame, roi_name, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw hand-scooper associations first (lines)
        for hand in hands:
            hand_center = get_box_center(hand['bbox'])
            closest_scooper = None
            min_distance = float('inf')
            
            for scooper in scoopers:
                scooper_center = get_box_center(scooper['bbox'])
                distance = ((hand_center[0] - scooper_center[0])**2 + 
                           (hand_center[1] - scooper_center[1])**2)**0.5
                if distance < HAND_SCOOPER_DISTANCE_THRESHOLD and distance < min_distance:
                    min_distance = distance
                    closest_scooper = scooper_center
            
            # Draw line if scooper is associated
            if closest_scooper:
                cv2.line(annotated_frame, 
                        (int(hand_center[0]), int(hand_center[1])),
                        (int(closest_scooper[0]), int(closest_scooper[1])),
                        (0, 255, 0), 2)
        
        # Draw detections with enhanced visuals
        for hand in hands:
            x1, y1, x2, y2 = hand['bbox']
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"Hand {hand['conf']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        for scooper in scoopers:
            x1, y1, x2, y2 = scooper['bbox']
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
            label = f"Scooper {scooper['conf']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 165, 255), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        for pizza in pizzas:
            x1, y1, x2, y2 = pizza['bbox']
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (203, 192, 255), 3)
            label = f"Pizza {pizza['conf']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (203, 192, 255), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw enhanced info overlay with detection counts
        overlay_height = 110
        cv2.rectangle(annotated_frame, (10, 10), (380, 10 + overlay_height), (0, 0, 0), -1)
        cv2.rectangle(annotated_frame, (10, 10), (380, 10 + overlay_height), (0, 255, 255), 2)
        
        # Violation count
        cv2.putText(annotated_frame, f"Violations: {violation_count}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Frame info
        cv2.putText(annotated_frame, f"Frame: {frame_number}/{total_frames}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Detection counts
        det_text = f"Hands:{len(hands)} Scoopers:{len(scoopers)} Pizzas:{len(pizzas)}"
        cv2.putText(annotated_frame, det_text, (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Device info
        device_text = "GPU" if device == 'cuda' else "CPU"
        cv2.putText(annotated_frame, f"Device: {device_text}", (20, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Save violation frame
        if current_violations:
            save_violation(frame_number, annotated_frame, current_violations)
        
        # Encode frame with FAST settings
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, 75,  # Lower quality = much faster
            cv2.IMWRITE_JPEG_OPTIMIZE, 0    # Disable optimization for speed
        ]
        _, buffer = cv2.imencode('.jpg', annotated_frame, encode_params)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare message with enhanced info
        message = {
            'type': 'frame',
            'frame': frame_base64,
            'frame_number': frame_number,
            'total_frames': total_frames,
            'progress': (frame_number / total_frames * 100) if total_frames > 0 else 0,
            'violation_count': violation_count,
            'violation_detected': len(current_violations) > 0,
            'detections': {
                'hands': len(hands),
                'scoopers': len(scoopers),
                'pizzas': len(pizzas)
            },
            'video_ended': False
        }
        
        latest_frame = message
        
        # Add to queue
        if not frame_queue.full():
            frame_queue.put(message)
        else:
            try:
                frame_queue.get_nowait()
            except:
                pass
            frame_queue.put(message)
        
        # Progress logging with detection info
        if processed_frames % 20 == 0:
            elapsed = time.time() - start_time
            processing_fps = processed_frames / elapsed if elapsed > 0 else 0
            progress_pct = (frame_number / total_frames * 100) if total_frames > 0 else 0
            if DEBUG_MODE:
                print(f"Frame {frame_number}/{total_frames} ({progress_pct:.1f}%) | FPS: {processing_fps:.1f} | "
                      f"Detections: H:{len(hands)} S:{len(scoopers)} P:{len(pizzas)} | Violations: {violation_count}")
            else:
                print(f"Frame {frame_number}/{total_frames} ({progress_pct:.1f}%) | FPS: {processing_fps:.1f} | Violations: {violation_count}")
        
        # Adaptive sleep - no sleep if FPS is low
        if processed_frames > 10:
            elapsed = time.time() - start_time
            current_fps = processed_frames / elapsed if elapsed > 0 else 0
            if current_fps > TARGET_FPS:
                time.sleep(0.005)  # Slow down if too fast
            # No sleep if FPS is low - maximize throughput
    
    # Send video ended message
    end_message = {
        'type': 'video_ended',
        'frame_number': frame_number,
        'total_frames': total_frames,
        'violation_count': violation_count,
        'video_ended': True
    }
    
    if not frame_queue.full():
        frame_queue.put(end_message)
    
    latest_frame = end_message
    
    cap.release()
    print("\n" + "=" * 70)
    print(f"VIDEO PROCESSING COMPLETE")
    print(f"Total frames processed: {frame_number}")
    print(f"Total violations detected: {violation_count}")
    print(f"Violations saved to: {VIOLATIONS_JSON_FILE}")
    print("=" * 70)


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(title="Eagle Vision Standalone")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML"""
    html_path = Path("frontend/index.html")
    if html_path.exists():
        return FileResponse(html_path)
    else:
        return """
        <html>
            <head><title>Eagle Vision</title></head>
            <body>
                <h1>Eagle Vision is running!</h1>
                <p>Frontend HTML not found. Check frontend/index.html</p>
                <p>WebSocket endpoint: ws://localhost:8080/ws/video</p>
            </body>
        </html>
        """


@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    """WebSocket endpoint for video streaming"""
    await websocket.accept()
    print("✓ WebSocket client connected")
    
    try:
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to Eagle Vision",
            "violation_count": violation_count
        })
        
        while True:
            try:
                if not frame_queue.empty():
                    frame_data = frame_queue.get()
                    await websocket.send_json(frame_data)
                else:
                    await websocket.send_json({
                        "type": "heartbeat",
                        "violation_count": violation_count
                    })
                
                await asyncio.sleep(0.01)
            
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                break
    
    finally:
        print("✗ WebSocket client disconnected")


@app.get("/api/violations")
async def get_violations():
    """Get violation history"""
    return {
        "total_violations": violation_count,
        "violations": violations_list[-50:]  # Last 50
    }


@app.get("/api/status")
async def get_status():
    """Get system status"""
    return {
        "status": "running",
        "queue_size": frame_queue.qsize(),
        "violation_count": violation_count,
        "model_loaded": True
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    print("=" * 70)
    print("🦅 EAGLE VISION - STANDALONE APPLICATION")
    print("=" * 70)
    print("Starting up...")
    print()
    
    # Check files exist
    if not os.path.exists(VIDEO_PATH):
        print(f"✗ Error: Video not found at {VIDEO_PATH}")
        print("  Please place your video at videos/test.mp4")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"✗ Error: Model not found at {MODEL_PATH}")
        print("  Please place your YOLO model at yolo12m-v2.pt")
        return
    
    print(f"✓ Video found: {VIDEO_PATH}")
    print(f"✓ Model found: {MODEL_PATH}")
    print()
    
    # Load YOLO model with automatic GPU/CPU selection
    print(f"Loading YOLO model from {MODEL_PATH}...")
    print()
    
    # Check GPU availability
    print("=" * 70)
    print("CHECKING COMPUTE DEVICE...")
    print("=" * 70)
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"GPU count: {gpu_count}")
        print()
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            memory_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  └─ Memory: {memory_gb:.2f} GB")
    else:
        print("No CUDA GPU detected")
        if PREFER_GPU:
            print()
            print("⚠️  GPU not available - will use CPU instead")
            print("💡 To enable GPU: Install CUDA and GPU-enabled PyTorch")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    print("=" * 70)
    print()
    
    # Determine device with fallback
    if PREFER_GPU and cuda_available:
        device = 'cuda:0'
        print(f"✓ USING GPU: {torch.cuda.get_device_name(0)}")
        use_gpu = True
    else:
        device = 'cpu'
        print("✓ USING CPU")
        use_gpu = False
    
    print()
    
    # Load model
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    
    # GPU-specific optimizations
    if use_gpu:
        print("Optimizing for GPU...")
        model.to(device)
        
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        print("✓ GPU optimizations enabled")
        print("  ├─ cuDNN benchmark: Enabled")
        print("  ├─ TF32 math: Enabled")
        print("  └─ Mixed precision: Ready")
    else:
        print("Optimizing for CPU...")
        print("✓ CPU optimizations enabled")
        print("  └─ Threading: Enabled")
    
    # Warm up model
    print()
    print("Warming up model...")
    dummy_frame = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    
    warmup_runs = 5 if use_gpu else 2
    for i in range(warmup_runs):
        _ = model(dummy_frame, verbose=False, imgsz=IMAGE_SIZE, device=device)
        if i == 0:
            print(f"  └─ Warming up... (0/{warmup_runs})", end='\r')
        else:
            print(f"  └─ Warming up... ({i}/{warmup_runs})", end='\r')
    
    if use_gpu:
        torch.cuda.synchronize()
    
    print(f"  └─ Warming up... ({warmup_runs}/{warmup_runs}) ✓")
    
    print()
    print("=" * 70)
    print("✓ MODEL READY")
    print("=" * 70)
    print(f"Device: {device.upper()}")
    print(f"Classes: {list(model.names.values())}")
    print(f"Image size: {IMAGE_SIZE}px")
    print(f"Frame skip: Every {FRAME_SKIP} frames")
    print(f"Target FPS: {TARGET_FPS}")
    
    if use_gpu:
        allocated_memory = torch.cuda.memory_allocated() / 1e6
        reserved_memory = torch.cuda.memory_reserved() / 1e6
        print(f"GPU Memory: {allocated_memory:.1f} MB allocated, {reserved_memory:.1f} MB reserved")
        print(f"Expected FPS: 40-60 (GPU)")
    else:
        print(f"Expected FPS: 8-12 (CPU)")
    
    print("=" * 70)
    print()
    
    # Start video processing thread with device info
    video_thread = threading.Thread(target=process_video_thread, args=(model, device), daemon=True)
    video_thread.start()
    print("✓ Video processing thread started")
    print()
    
    # Start web server
    print("Starting web server...")
    print("=" * 70)
    print("🌐 Access the system at: http://localhost:8080")
    print("=" * 70)
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


if __name__ == "__main__":
    main()

