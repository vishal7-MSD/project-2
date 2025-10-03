from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.model_utils import predict_all, FEATURES

app = FastAPI(title="Heart Disease Prediction App")

# Static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "features": FEATURES})

@app.post("/predict")
async def predict(request: Request):
    try:
        input_data = await request.json()
        print("Received input:", input_data)  # ✅ Debugging print

        results = predict_all(input_data)

        # Structure results for frontend
        formatted_results = [
            {
                "model": model_name,
                "prediction": model_result["prediction"],
                "probability": model_result["probability"]
            }
            for model_name, model_result in results.items()
        ]

        return JSONResponse(content={"results": formatted_results})

    except Exception as e:
        print("❌ Error during prediction:", e)
        return JSONResponse(status_code=400, content={"error": str(e)})
