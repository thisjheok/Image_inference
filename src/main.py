# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
from .api.inference import predict_image  # 상대 경로로 변경

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    result = predict_image(image)  # PyTorch 추론
    return JSONResponse(content={"result": result})
