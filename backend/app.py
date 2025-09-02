from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Media Authenticity MVP Backend is running!"}
