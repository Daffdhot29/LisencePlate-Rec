from fastapi import FastAPI 
from routes.detect_plate import router
app = FastAPI()

app.include_router(router) 