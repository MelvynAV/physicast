import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from ultralytics import YOLO
from openai import OpenAI
from dotenv import load_dotenv

# Load credentials from the .env file
load_dotenv()

app = FastAPI()

# Mount the static folder for your UI
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load YOLOv8 for Scene Interpretation
vision_model = YOLO("yolov8n.pt")

# Configure the client to use Groq's high speed API
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.post("/analyze")
async def analyze_space(file: UploadFile = File(...)):
    # Save the file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 1. Vision Phase: Detect objects and extract coordinates
    results = vision_model(temp_path)
    
    detected_objects = []
    for box in results[0].boxes:
        label = vision_model.names[int(box.cls)]
        coords = box.xyxy[0].tolist() 
        detected_objects.append(f"{label} at {coords}")

    # Log vision results for debugging
    print(f"VISION LOG: {len(detected_objects)} objects identified.")
    
    os.remove(temp_path)

    # 2. Reasoning Phase: Build a sophisticated spatial prompt
    prompt = (
        f"Act as a Physical AI Spatial Engine. Data: {', '.join(detected_objects)}. "
        "Task: 1. Create a semantic map. 2. Define Interaction Zones vs Static Zones. "
        "3. Suggest a robotic navigation path. Use professional technical English."
    )

    # 3. Inference Phase: Call Groq (Llama 3.3 70B)
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a professional Spatial Reasoning Engine."},
                {"role": "user", "content": prompt}
            ]
        )
        reasoning = response.choices[0].message.content
        print("LOG: Spatial reasoning generated successfully via Groq.")
    except Exception as e:
        print(f"ERROR: {e}")
        reasoning = "The Reasoning Engine is currently unavailable."

    return {
        "detected_objects": detected_objects,
        "spatial_reasoning": reasoning
    }