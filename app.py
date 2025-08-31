import os
import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import shutil
from pathlib import Path

from model import create_model
from config import MODEL_CONFIG, PATHS, device

# Initialize FastAPI app
app = FastAPI(title="Pest Detection & Eco Pesticide Recommendation")

# Mount static files directory
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Load the model
model = None

class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float
    class_id: int
    class_name: str

class DetectionResult(BaseModel):
    image_path: str
    detections: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]

class PestDetectionRequest(BaseModel):
    image: UploadFile = File(...)
    confidence_threshold: float = 0.5

class PesticideRecommender:
    def __init__(self, db_path: str = 'pesticide_db.json'):
        self.db = self._load_database(db_path)
    
    def _load_database(self, db_path: str) -> Dict:
        """Load the pesticide database from a JSON file."""
        if os.path.exists(db_path):
            with open(db_path, 'r') as f:
                return json.load(f)
        
        # Default database
        return {
            "default": {
                "eco_friendly": ["Neem Oil", "Insecticidal Soap"],
                "effectiveness": 0.8,
                "environmental_impact": "Low",
                "safety_level": "Safe for beneficial insects"
            },
            "aphid": {
                "eco_friendly": ["Neem Oil", "Insecticidal Soap", "Horticultural Oil"],
                "effectiveness": 0.9,
                "environmental_impact": "Low",
                "safety_level": "Safe for beneficial insects"
            },
            "caterpillar": {
                "eco_friendly": ["Bacillus thuringiensis (Bt)", "Spinosad"],
                "effectiveness": 0.85,
                "environmental_impact": "Low",
                "safety_level": "Safe for beneficial insects when used as directed"
            },
            "mite": {
                "eco_friendly": ["Insecticidal Soap", "Horticultural Oil", "Sulfur"],
                "effectiveness": 0.8,
                "environmental_impact": "Low to Moderate",
                "safety_level": "Use with caution in hot weather"
            }
        }
    
    def get_recommendation(self, pest_type: str, severity: float = 0.5) -> Dict[str, Any]:
        """
        Get pesticide recommendation based on pest type and severity.
        
        Args:
            pest_type: Type of pest (e.g., 'aphid', 'caterpillar')
            severity: Severity of infestation (0.0 to 1.0)
            
        Returns:
            Dictionary containing recommendation details
        """
        pest_info = self.db.get(pest_type.lower(), self.db['default'])
        
        # Adjust recommendation based on severity
        if severity > 0.7:  # High severity
            if "Pyrethrin-based spray" not in pest_info['eco_friendly']:
                pest_info['eco_friendly'].append("Pyrethrin-based spray")
            pest_info['effectiveness'] = min(0.95, pest_info['effectiveness'] * 1.1)
        
        return {
            "pest_type": pest_type,
            "recommended_pesticides": pest_info['eco_friendly'],
            "effectiveness": round(pest_info['effectiveness'], 2),
            "environmental_impact": pest_info['environmental_impact'],
            "safety_level": pest_info.get('safety_level', 'Moderate'),
            "application_notes": self._get_application_notes(pest_type, severity)
        }
    
    def _get_application_notes(self, pest_type: str, severity: float) -> List[str]:
        """Get application notes based on pest type and severity."""
        notes = []
        
        if severity > 0.7:  # High severity
            notes.append("Apply treatment immediately and repeat after 7 days.")
        else:
            notes.append("Apply treatment as needed, monitor pest population.")
        
        if pest_type.lower() == 'aphid':
            notes.append("Focus on the undersides of leaves where aphids often cluster.")
        elif pest_type.lower() == 'caterpillar':
            notes.append("Apply in the evening to protect beneficial insects.")
        
        return notes

# Initialize pesticide recommender
recommender = PesticideRecommender()

def load_model():
    """Load the pest detection model."""
    global model
    if model is None:
        model = create_model(num_classes=MODEL_CONFIG['num_classes'])
        model.load_state_dict(torch.load(PATHS['model_save'] / 'best_model.pth', map_location=device))
        model.eval()
        model.to(device)

@app.on_event("startup")
async def startup_event():
    """Initialize the model when the application starts."""
    load_model()
    print("Model loaded successfully")

def process_image(image_path: str, confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Process an image to detect pests and get recommendations.
    
    Args:
        image_path: Path to the input image
        confidence_threshold: Minimum confidence score for detections
        
    Returns:
        Dictionary containing detection results and recommendations
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = torch.tensor(np.array(image) / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Process predictions
    detections = []
    for i in range(len(predictions[0]['boxes'])):
        box = predictions[0]['boxes'][i].cpu().numpy()
        score = predictions[0]['scores'][i].cpu().numpy()
        label = predictions[0]['labels'][i].cpu().numpy()
        
        if score >= confidence_threshold:
            # Convert box coordinates to list
            box = box.tolist()
            
            # Get class name (in a real app, you'd have a mapping from class ID to name)
            class_name = f"pest_{int(label)}"
            
            detections.append({
                "bbox": box,
                "confidence": float(score),
                "class_id": int(label),
                "class_name": class_name
            })
    
    # Get recommendations for each detected pest
    recommendations = []
    for detection in detections:
        pest_type = detection['class_name']
        severity = detection['confidence']  # Using confidence as a proxy for severity
        
        recommendation = recommender.get_recommendation(pest_type, severity)
        recommendations.append({
            "pest_type": pest_type,
            "recommendation": recommendation,
            "detection_confidence": detection['confidence']
        })
    
    # Save visualization
    output_path = save_visualization(image, detections, str(static_dir / "results"))
    
    return {
        "image_path": output_path,
        "detections": detections,
        "recommendations": recommendations
    }

def save_visualization(image: Image.Image, detections: List[Dict], output_dir: str) -> str:
    """
    Save an image with detection boxes and labels.
    
    Args:
        image: Input image
        detections: List of detection dictionaries
        output_dir: Directory to save the output image
        
    Returns:
        Path to the saved image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    output_path = os.path.join(output_dir, f"result_{len(os.listdir(output_dir)) + 1}.jpg")
    
    # Draw bounding boxes and labels
    draw = ImageDraw.Draw(image)
    
    for detection in detections:
        box = detection['bbox']
        label = detection['class_name']
        confidence = detection['confidence']
        
        # Draw rectangle
        draw.rectangle(box, outline="red", width=3)
        
        # Draw label background
        label_text = f"{label} {confidence:.2f}"
        text_bbox = draw.textbbox((box[0], box[1]), label_text)
        draw.rectangle(text_bbox, fill="red")
        
        # Draw text
        draw.text((box[0], box[1]), label_text, fill="white")
    
    # Save the image
    image.save(output_path)
    
    return output_path

@app.post("/detect", response_model=DetectionResult)
async def detect_pests(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5
):
    """
    Detect pests in an uploaded image and get eco-friendly pesticide recommendations.
    """
    # Check if the file is an image
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save the uploaded file
    upload_dir = static_dir / "uploads"
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the image
    try:
        result = process_image(str(file_path), confidence_threshold)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    """Root endpoint with API documentation."""
    return {
        "message": "Pest Detection & Eco Pesticide Recommendation API",
        "endpoints": {
            "/detect": "POST - Upload an image to detect pests and get recommendations",
            "/docs": "API documentation"
        },
        "status": "running"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
