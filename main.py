from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Load trained vehicle detection model
model = YOLO("best.pt")


@app.get("/")
def home():
    return {
        "message": "Vehicle Detection API is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read image
    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    # Run YOLO prediction
    results = model.predict(
        image,
        imgsz=640,
        conf=0.25
    )

    result = results[0]

    detections = []

    if result.boxes is not None:

        for i in range(len(result.boxes)):

            box = result.boxes.xyxy[i].tolist()
            confidence = float(result.boxes.conf[i])

            # Get actual class predicted by the model
            class_id = int(result.boxes.cls[i])
            class_name = model.names[class_id]

            detections.append({
                "bbox": box,
                "confidence": confidence,
                "class": class_name
            })

    return {
        "vehicle_count": len(detections),
        "detections": detections
    }