from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
from io import BytesIO
import traceback

from app.features import create_features  # <--- import ini penting

app = FastAPI()

# Load model pipeline (harus berisi preprocessor dan model)
model = joblib.load("models/final_model.pkl")

@app.post("/predict_csv")
async def predict_from_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))

        # Buat fitur tambahan
        df = create_features(df)

        # Prediksi
        prediction = model.predict(df)
        return {"predictions": prediction.tolist()}

    except Exception as e:
        print(traceback.format_exc())  # Debug di terminal
        return JSONResponse(status_code=500, content={"message": f"Internal error: {str(e)}"})
