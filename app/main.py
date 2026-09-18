from fastapi import FastAPI
from app.routers.v1 import router as v1_router

app = FastAPI()

app.include_router(v1_router)

@app.get("/")
async def root():
    return {"message": "Hello World"}

