from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from uvicorn import run as app_run
from src.forest.constant import APP_HOST, APP_PORT

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/train")
async def trainRouteClient():
    try:
        pass

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.get("/predict")
async def predictRouteClient():
    try:
        pass

    except Exception as e:
        return Response(f"Error Occurred! {e}")


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)