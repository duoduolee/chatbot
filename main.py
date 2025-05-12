
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import httpx
from tenacity import retry, wait_fixed, stop_after_attempt
import traceback

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
VLLM_API_URL = "http://localhost:8000/v1/chat/completions"
HEADERS = {"Authorization": "Bearer EMPTY"}

timeout = httpx.Timeout(30.0, connect=10.0)

@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
async def get_llm_response(payload):
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(VLLM_API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        return response.json()

@app.get("/", response_class=HTMLResponse)
async def get_chat(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user_input": None,
        "response": None,
        "error": None
    })

@app.post("/chat", response_class=HTMLResponse)
async def post_chat(request: Request, user_input: str = Form(...)):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ],
        "temperature": 0.7
    }

    try:
        result = await get_llm_response(payload)
        reply = result["choices"][0]["message"]["content"]
        error = None
    except Exception as e:
        traceback.print_exc()
        reply = None
        error = f"⚠️ Error: {str(e)}"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "user_input": user_input,
        "response": reply,
        "error": error
    })

