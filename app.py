from fastapi import FastAPI, Request, Depends, HTTPException, Form
import requests
from huggingface_hub import hf_hub_download
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2AuthorizationCodeBearer
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
from authlib.integrations.starlette_client import OAuthError
from starlette.config import Config
from starlette.requests import Request
from authlib.integrations.starlette_client import OAuth
import os
import hashlib
from pymongo.mongo_client import MongoClient
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

app = FastAPI()

# ✅ Ensure SessionMiddleware is configured correctly
# ✅ Add CORS Middleware separately
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Add Session Middleware separately
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "defaultsecret"),  # Ensure SECRET_KEY is set
    session_cookie="session",  
    same_site="lax",  
    https_only=True  
)

# ✅ OAuth configuration
oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    authorize_url="https://accounts.google.com/o/oauth2/auth",
    access_token_url="https://oauth2.googleapis.com/token",
    redirect_uri="https://mrcodder-barttune.hf.space/callback",  # ✅ Fix this
    client_kwargs={"scope": "openid email profile"},
)

templates = Jinja2Templates(directory="templates")
users = {}

client = MongoClient(os.getenv("MONGO_URI"))
db = client["user_db"]
users_collection = db["users"]

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Helper to hash passwords
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# --- Signup Routes ---
@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/signup")
async def signup(request: Request, email: str = Form(...), password: str = Form(...)):
    if users_collection.find_one({"email": email}):
        return JSONResponse(content={"error": "Email already registered"}, status_code=400)
    
    users_collection.insert_one({
        "email": email,
        "password": hash_password(password)
    })
    return RedirectResponse(url="/login", status_code=302)

# --- Login Routes ---
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    user = users_collection.find_one({"email": email})
    if user and user["password"] == hash_password(password):
        request.session["user"] = {"email": email}
        return RedirectResponse(url="/chat", status_code=302)
    
    return JSONResponse(content={"error": "Invalid credentials"}, status_code=401)

@app.get("/login/google")
async def login_google(request: Request):
    try:
        raw_uri = request.url_for("auth_google_callback")  # Get URI before modification
        print(f"Raw Redirect URI: {raw_uri}")  # Debugging

        # Ensure it is a string before replacing
        redirect_uri = str(raw_uri).replace("http://", "https://") 
        return await oauth.google.authorize_redirect(request, redirect_uri)
    except Exception as e:
        print(f"Google OAuth Error: {e}")  # Debugging
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/callback", name="auth_google_callback")
async def auth_google_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
        if not token:
            raise HTTPException(status_code=400, detail="Failed to retrieve token.")

        user_info = token.get("userinfo")
        if not user_info:
            user_info = await oauth.google.parse_id_token(request, token)
        
        if not user_info:
            raise HTTPException(status_code=400, detail="Failed to retrieve user information.")

        # Store user in session
        request.session["user"] = dict(user_info)
        return RedirectResponse(url="/chat")
    
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.get("/chat", response_class=HTMLResponse)
def chat(request: Request):
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse("chat.html", {"request": request, "user": user})

@app.get("/logout")
def logout(request: Request):
    request.session.pop("user", None)
    return RedirectResponse(url="/", status_code=302)

# ✅ List of epochs to preload
EPOCHS = [1, 3, 5, 8, 10, 12, 15, 20]
MODEL_CACHE = {}  # ✅ Dictionary to store preloaded models

    
def load_model(epoch):
    """Load model & tokenizer from local directory."""
    # ✅ Load model & tokenizer from local path
    repo_id = f"MrCodder/BartBaseFineTuningModel{epoch}"
    model = BartForConditionalGeneration.from_pretrained(repo_id)
    tokenizer = BartTokenizer.from_pretrained(repo_id)

    # ✅ Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ✅ Store in cache
    MODEL_CACHE[str(epoch)] = (model, tokenizer)
    print(f"✅ Model for epoch {epoch} loaded successfully!")


@app.on_event("startup")
def initialize_models():
    """Download and load all models at startup."""
    print("🚀 Downloading & Loading all models...")
    for epoch in EPOCHS:
        load_model(epoch)
    print("✅ All models are ready to use!")

@app.post("/generate")
def generate(data: dict):
    """Generate text using the preloaded model."""
    input_text = data.get("context", "")
    epoch = data.get("epoch", 1)
    if not input_text:
        print("❌ No input provided!")
        raise HTTPException(status_code=400, detail="No input provided")
    
    if epoch not in MODEL_CACHE:
        print(f"❌ Model for epoch {epoch} not found!")
        raise HTTPException(status_code=400, detail=f"Model for epoch {epoch} not found.")

    try:
        model, tokenizer = MODEL_CACHE[epoch]
        print("✅ Model & tokenizer loaded!")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = tokenizer(input_text, return_tensors="pt")
        print(f"✅ Tokenized Input: {inputs}")

        inputs = inputs.to(device)  # Move input tensors to device
        print("✅ Moved inputs to device!")

        with torch.no_grad():
            summary_ids = model.generate(
                inputs['input_ids'], max_length=150, num_beams=5, early_stopping=True
            )
        print(f"✅ Generated output: {summary_ids}")

        response = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print(f"✅ Final Response: {response}")

        return {"response": response}

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=7860, 
        log_level="info", 
        reload=True, 
        workers=1, 
        use_colors=True, 
        env_file=".env", 
        trust_env=True  # ✅ Ensure environment variables work properly
    )
