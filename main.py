from fastapi import FastAPI
from routes.plate_routes import router

app = FastAPI(
    title="License Plate Recognition API",
    version="1.0.0"
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "API is running"
    }