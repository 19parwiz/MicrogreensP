# AgroTech Recognition System

AI-powered plant and face recognition system for agricultural applications. Built with YOLO and OpenCV for real-time detection and analysis.

## Features

- **Plant Recognition**: Detect and classify microgreens (Beetroot, Arugula, Mangold, Tarragon, Basil)
- **Face Recognition**: Multi-person detection and identification
- **Growth Tracking**: Monitor plant development over time
- **Multiple Camera Support**: Webcam, IP cameras, video files
- **Real-time Processing**: Live feed analysis with instant results

## Installation

### Prerequisites
- Python 3.8+
- Webcam or IP camera (optional)
- GPU recommended for better performance

### Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/19parwiz/AgroTechP.git
   cd AgroTechP
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download pre-trained models** (if not included)
   ```bash
   # Models are included in the repository
   # Additional models will be downloaded automatically on first run
   ```

## Usage

### Basic Usage
```bash
# Run the main application
python main.py

# Specific detection modes
python main.py --mode plant          # Plant recognition only
python main.py --mode face           # Face recognition only
python main.py --mode enhanced       # Enhanced plant detection
```

### Camera Configuration
```bash
# Use IP camera
python main.py --camera-url "http://192.168.1.100:8080/video"

# Use specific webcam
python main.py --camera-id 1
```

### Enhanced Plant Recognition
```bash
# Run enhanced detection with advanced features
python src/enhanced_plant_recognition.py
```

## Project Structure

```
AgroTech/
├── src/                          # Core modules
│   ├── camera.py                 # Camera handling
│   ├── plant_recognition_module.py
│   ├── enhanced_plant_recognition.py
│   ├── face_recognition_module.py
│   └── growth_tracker.py
├── models/                       # Trained models
│   ├── best.pt                   # YOLO model
│   ├── plant_model.h5           # Plant classifier
│   └── yolo11_microgreens/      # Training results
├── data/                         # Datasets
│   ├── microgreens/             # Plant training data
│   └── faces/                   # Face training data
├── services/                     # Service modules
├── training/                     # Training scripts
├── tools/                        # Utility scripts
├── test_images/                  # Sample images
└── outputs/                      # Results and logs
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory:
```env
CAMERA_URL=http://192.168.1.100:8080/video
DEFAULT_CAMERA_ID=0
MODEL_PATH=models/best.pt
CONFIDENCE_THRESHOLD=0.5
```

### Model Configuration
- **Plant Models**: Located in `models/` directory
- **Training Data**: Organized in `data/microgreens/` and `data/faces/`
- **Custom Models**: Add your trained models to the `models/` directory

## 🎮 Usage Examples

### 1. Plant Detection
```python
from src.enhanced_plant_recognition import EnhancedPlantRecognition

# Initialize detector
detector = EnhancedPlantRecognition()

# Process image
result = detector.detect_image("path/to/plant_image.jpg")
print(f"Detected: {result['class']} with confidence: {result['confidence']}")
```

### 2. Face Recognition
```python
from src.face_recognition_module import FaceRecognition

# Initialize face recognizer
face_rec = FaceRecognition()

# Add new person
face_rec.add_person("John Doe", "path/to/john_photos/")

# Recognize faces
faces = face_rec.recognize_faces(image)
```

### 3. Growth Tracking
```python
from src.growth_tracker import GrowthTracker

# Track plant growth
tracker = GrowthTracker()
tracker.add_measurement("plant_001", size=5.2, health_score=0.85)
```

## 🚀 Docker Support

Build and run with Docker:
```bash
# Build image
docker build -t agrotech .

# Run container
docker run -it --rm -v /dev/video0:/dev/video0 agrotech
```

## 📊 Model Performance

### Microgreen Detection
- **Accuracy**: 94.5% on validation set
- **Classes**: 5 microgreen varieties
- **Model**: YOLOv11n optimized for real-time detection

### Face Recognition
- **Accuracy**: 98.2% on test set
- **Speed**: Real-time processing at 30 FPS
- **Features**: Multi-face detection and tracking

## 🛠️ Development

### Training New Models
```bash
# Train plant recognition model
python training/train_microgreens.py

# Train face recognition model
python training/train.py
```

### Adding New Plant Species
1. Add training images to `data/microgreens/train/NewSpecies/`
2. Update `data/microgreens.yaml` with new class
3. Retrain the model using training scripts

### Testing
```bash
# Run comparison tests
python compare_detections.py
python compare_precision.py

# Test specific modules
python training/test_inference.py
```

## 📈 Performance Optimization

- **GPU Acceleration**: Automatically uses CUDA if available
- **Model Optimization**: TensorRT support for NVIDIA GPUs
- **Memory Management**: Efficient image processing and caching
- **Multi-threading**: Parallel processing for multiple cameras

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **YOLOv11**: For object detection capabilities
- **OpenCV**: For computer vision operations
- **TensorFlow**: For deep learning model support
- **Ultralytics**: For YOLO implementation

## 📞 Contact

- **Author**: Pariwz 
- **GitHub**: [@19parwiz](https://github.com/19parwiz)
- **Project**: [AgroTechP](https://github.com/19parwiz/AgroTechP)

---

⭐ **Star this repository if you found it helpful!**
