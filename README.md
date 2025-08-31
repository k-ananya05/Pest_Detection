# Pest Detection & Eco Pesticide Recommendation System

An AI-powered system that detects pests in crop images and recommends eco-friendly pesticides.

## Features

- **Pest Detection**: Uses a deep learning model to detect and localize pests in images
- **Eco-Friendly Recommendations**: Suggests environmentally friendly pesticides for detected pests
- **Severity Assessment**: Provides treatment recommendations based on infestation severity
- **Web API**: RESTful API for easy integration with other applications
- **Visualization**: Generates annotated images with detection results

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- CUDA-compatible GPU (recommended for faster inference)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pest-detection.git
   cd pest-detection
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training the Model

To train the pest detection model:

```bash
python train.py
```

Training progress will be logged to TensorBoard. You can monitor it by running:

```bash
tensorboard --logdir=runs
```

### 2. Running the API

To start the FastAPI server:

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

### 3. Using the API

#### Detect Pests in an Image

Send a POST request to `/detect` with an image file:

```bash
curl -X 'POST' \
  'http://localhost:8000/detect' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@path/to/your/image.jpg;type=image/jpeg'
```

#### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
pest-detection/
├── app.py                 # FastAPI application
├── config.py              # Configuration settings
├── dataset.py             # Dataset handling
├── model.py               # Model architecture
├── train.py               # Training script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── static/                # Static files (uploads, results)
│   ├── uploads/           # User-uploaded images
│   └── results/           # Processed images with detections
└── saved_models/          # Trained model checkpoints
```

## Pesticide Database

The system includes a built-in database of eco-friendly pesticides. You can customize it by modifying the `pesticide_db.json` file.

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Wadhwani AI for the pest management open dataset
- PyTorch and FastAPI communities for their excellent tools
- All contributors who helped improve this project
