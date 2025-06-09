from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, shutil, pickle, base64
from io import BytesIO
from tensorflow.keras.models import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
model = load_model("Model.keras", compile=False)

with open("scalar.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.post("/predict")
async def predict_loss(
    hours: str = Form(None),
    pslack: str = Form(None),
    bus1: str = Form(None),
    bus2: str = Form(None),
    bus3: str = Form(None),
    bus4: str = Form(None),
    bus5: str = Form(None),
    bus6: str = Form(None),
    bus7: str = Form(None),
    bus8: str = Form(None),
    bus9: str = Form(None),
    bus10: str = Form(None),
    bus11: str = Form(None),
    data_file: UploadFile = File(None),
    test_results: UploadFile = File(None)
):
    # Step 1: Load input data
    if data_file:
        data_path = f"temp_{data_file.filename}"
        with open(data_path, "wb") as f:
            shutil.copyfileobj(data_file.file, f)
        input_df = pd.read_csv(data_path)
        os.remove(data_path)
    else:
        # Validate form fields
        required = [hours, pslack, bus1, bus2, bus3, bus4, bus5, bus6, bus7, bus8, bus9, bus10, bus11]
        if any(v is None or v.strip() == "" for v in required):
            return JSONResponse(content={"error": "Missing required form fields"}, status_code=400)

        input_df = pd.DataFrame([[float(hours), float(pslack), float(bus1), float(bus2), float(bus3),
                                  float(bus4), float(bus5), float(bus6), float(bus7), float(bus8),
                                  float(bus9), float(bus10), float(bus11)]],
                                columns=["Hour", "Pslack", "Bus 1", "Bus 2", "Bus 3", "Bus 4", "Bus 5",
                                         "Bus 6", "Bus 7", "Bus 8", "Bus 9", "Bus 10", "Bus 11"])

    # Step 2: Scale features and predict
    features = input_df.iloc[:, 1:]
    scaled = scaler.transform(features)
    predictions = model.predict(scaled).flatten()
    input_df["Predicted Loss"] = predictions

    # Step 3: Optional comparison plot
    plt.figure(figsize=(12, 6))
    if test_results:
        test_result_path = f"temp_{test_results.filename}"
        with open(test_result_path, "wb") as f:
            shutil.copyfileobj(test_results.file, f)
        actual = pd.read_csv(test_result_path).to_numpy().flatten()
        os.remove(test_result_path)

        plt.plot(input_df["Hour"], actual[:len(predictions)], label="Actual", color="red")
        plt.plot(input_df["Hour"], predictions, label="Predicted", color="blue")
        plt.title("Actual vs Predicted Loss")
    else:
        plt.plot(input_df["Hour"], predictions, label="Predicted", color="blue", marker="o")
        plt.title("Predicted Loss Over Time")

    plt.xlabel("Hour")
    plt.ylabel("Loss (MW)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Convert plot to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    # Prepare response
    response = {
        "total_rows": len(input_df),
        "total_load": float(features.sum(axis=1).sum()),
        "predictions": input_df[["Hour", "Predicted Loss"]].to_dict(orient="records"),
        "plot_image_base64": f"data:image/png;base64,{image_base64}"
    }

    return JSONResponse(content=response)
    