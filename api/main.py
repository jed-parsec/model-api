from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def health_check():
    return "Health Check: Server healthy and running. This is a test"