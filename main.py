from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from src.endpoints.router import router as mainendpointsrouter

app= FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],   
)

@app.get("/")
async def root():
    return {"status:": "okay"}

app.include_router(router= mainendpointsrouter, prefix= "/api")

