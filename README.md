# 🦅 Eagle Vision

**Hygiene Compliance Monitoring System**

Eagle Vision is an AI-powered real-time video monitoring system that detects food safety violations in commercial kitchens. Using advanced computer vision and YOLO object detection, it identifies when workers grab ingredients from containers without using proper utensils (scoopers), ensuring compliance with hygiene standards.

## ✨ Features

- **Real-time Object Detection**: Detects hands, scoopers, pizzas, and other objects using YOLO v12
- **Three-Stage Violation Detection**: 
  - Stage 1: Hand enters ROI (Region of Interest) without scooper
  - Stage 2: Hand moves away from ROI
  - Stage 3: Hand touches pizza/food surface
- **Web-Based Dashboard**: Modern, responsive web interface with real-time video streaming
- **Violation Tracking**: Automatically saves violation frames and detailed JSON records
- **GPU/CPU Support**: Automatic device detection with GPU acceleration when available
- **Configurable ROIs**: Define custom regions of interest for ingredient containers
- **Performance Optimized**: Frame skipping, batch processing, and adaptive FPS control

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster processing)
- YOLO model file (`yolo12m-v2.pt`)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/eaglevison.git
   cd eaglevison
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements_standalone.txt
   ```

4. **Place your video file**
   - Create a `videos` directory
   - Place your video file as `videos/1.mp4` (or update `VIDEO_PATH` in `standalone_app.py`)

5. **Place your YOLO model**
   - Ensure `yolo12m-v2.pt` is in the project root (or update `MODEL_PATH` in `standalone_app.py`)

### Running the Application

```bash
python standalone_app.py
```

The application will:
1. Check for video and model files
2. Detect available compute device (GPU/CPU)
3. Load and warm up the YOLO model
4. Start the web server on `http://localhost:8080`

Open your browser and navigate to `http://localhost:8080` to view the dashboard.

## ⚙️ Configuration

Edit the configuration section in `standalone_app.py` to customize behavior:

```python
# Video and Model Paths
VIDEO_PATH = "videos/1.mp4"
MODEL_PATH = "yolo12m-v2.pt"

# Detection Settings
CONFIDENCE_THRESHOLD = 0.30  # Detection confidence threshold
TARGET_FPS = 12  # Target processing FPS
FRAME_SKIP = 1  # Process every Nth frame (1 = all frames)
REPLAY_VIDEO = False  # Loop video when finished
PREFER_GPU = True  # Use GPU if available

# ROI Configuration (Region of Interest)
ROIS = {
    'ingredient_containers': (230, 250, 450, 550)  # (x1, y1, x2, y2)
}

# Violation Detection Settings
MIN_FRAMES_IN_ROI = 8  # Minimum frames hand must be in ROI
HAND_SCOOPER_DISTANCE_THRESHOLD = 250  # Max distance for hand-scooper association
VIOLATION_COOLDOWN = 60  # Frames between violation detections
```

### ROI Configuration

To configure the Region of Interest (ROI) for ingredient containers:

1. Run the application and view the video stream
2. Note the coordinates where ingredient containers are located
3. Update the `ROIS` dictionary with the bounding box coordinates: `(x1, y1, x2, y2)`
   - `x1, y1`: Top-left corner
   - `x2, y2`: Bottom-right corner

## 📁 Project Structure

```
eaglevison/
├── standalone_app.py          # Main application file
├── requirements_standalone.txt # Python dependencies
├── yolo12m-v2.pt              # YOLO model file (not included)
├── videos/                    # Video files directory
│   └── 1.mp4                  # Input video
├── frontend/                   # Web frontend
│   └── index.html             # Dashboard HTML
├── violation_frames/          # Saved violation frames
│   └── violation_frame_*.jpg  # Captured violation images
├── violations.json            # Violation records (JSON)
└── README.md                  # This file
```

## 🔧 Technology Stack

- **Backend**: FastAPI, Uvicorn
- **Computer Vision**: OpenCV, Ultralytics YOLO
- **Deep Learning**: PyTorch
- **Frontend**: HTML5, JavaScript, WebSocket
- **Image Processing**: NumPy, SciPy

## 📡 API Endpoints

### WebSocket
- `ws://localhost:8080/ws/video` - Real-time video stream with detection data

### REST API
- `GET /` - Web dashboard
- `GET /api/violations` - Get violation history (last 50)
- `GET /api/status` - Get system status

## 🎯 How It Works

1. **Video Processing**: The application reads frames from the input video
2. **Object Detection**: YOLO model detects hands, scoopers, and pizzas in each frame
3. **Hand Tracking**: Hands are tracked across frames using distance and size matching
4. **ROI Monitoring**: System checks if hands enter the configured ROI (ingredient containers)
5. **Scooper Detection**: Verifies if a scooper is associated with the hand
6. **Violation Detection**: Triggers when:
   - Hand is in ROI for minimum frames
   - No scooper is detected near the hand
   - Hand moves to pizza/food surface
7. **Recording**: Violation frames are saved with timestamps and metadata

## 📊 Output

### Violation Frames
- Saved as JPEG images in `violation_frames/` directory
- Filename format: `violation_frame_{frame_number}.jpg`

### Violations JSON
- Detailed records in `violations.json`
- Includes frame number, timestamp, violation type, and bounding boxes

Example violation record:
```json
{
  "frame_number": 210,
  "timestamp": "2025-12-17T23:23:29.660281",
  "frame_path": "violation_frames/violation_frame_210.jpg",
  "violation_info": [{
    "hand_id": 0,
    "roi_name": "ingredient_containers",
    "frames_in_roi": 8,
    "violation_type": "Grabbed from ingredient_containers without scooper and placed on pizza"
  }]
}
```

## 🐛 Troubleshooting

### GPU Not Detected
- Ensure CUDA is installed: `nvidia-smi` should show GPU info
- Install GPU-enabled PyTorch:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

### Video Not Found
- Check that `VIDEO_PATH` points to a valid video file
- Supported formats: MP4, AVI, MOV (via OpenCV)

### Model Not Found
- Ensure `yolo12m-v2.pt` is in the project root
- Or update `MODEL_PATH` to point to your model file

### Low FPS
- Reduce `IMAGE_SIZE` (e.g., 640 → 512)
- Increase `FRAME_SKIP` (e.g., 2 = process every 2nd frame)
- Disable `ENABLE_PREPROCESSING` for faster processing
- Use GPU if available


