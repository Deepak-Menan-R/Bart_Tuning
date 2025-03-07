from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from requests_oauthlib import OAuth2Session
import os
from dotenv import load_dotenv
import torch
from transformers import BartForConditionalGeneration, BartTokenizer

load_dotenv()

app = Flask(__name__)

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# Flask-Login setup
app.secret_key = os.getenv('SECRET_KEY')
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# OAuth configuration
client_id = os.getenv('GOOGLE_CLIENT_ID')
client_secret = os.getenv('GOOGLE_CLIENT_SECRET')
authorization_base_url = 'https://accounts.google.com/o/oauth2/auth'
token_url = 'https://oauth2.googleapis.com/token'
redirect_uri = 'http://127.0.0.1:5000/callback'
scope = ['profile', 'email']

users = {}

class User(UserMixin):
    def __init__(self, user_id, name, email):
        self.id = user_id
        self.name = name
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

# Load model
model_name = "./bart_finetune15"  
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # Mock check: Replace with actual user authentication
        if email == "user@example.com" and password == "password":
            user = User(email, "Test User", email)
            users[email] = user
            login_user(user)
            return redirect(url_for('chat'))

        return "Invalid credentials", 401

    return render_template('login.html')

@app.route('/login/google')
def login_google():
    google = OAuth2Session(client_id, scope=scope, redirect_uri=redirect_uri)
    authorization_url, state = google.authorization_url(authorization_base_url)
    session['oauth_state'] = state
    return redirect(authorization_url)

@app.route('/callback')
def callback():
    google = OAuth2Session(client_id, state=session['oauth_state'], redirect_uri=redirect_uri)
    token = google.fetch_token(token_url, client_secret=client_secret, authorization_response=request.url)

    google = OAuth2Session(client_id, token=token)
    user_info = google.get('https://www.googleapis.com/oauth2/v1/userinfo').json()

    user = User(user_info['id'], user_info['name'], user_info['email'])
    users[user_info['id']] = user
    login_user(user)

    return redirect(url_for('chat'))

@app.route('/chat')
@login_required
def chat():
    return render_template('chat.html', user=current_user)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    input_text = data.get('context', '')

    if not input_text:
        return jsonify({"error": "No input provided"}), 400

    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        summary_ids = model.generate(inputs['input_ids'], max_length=150, num_beams=5, early_stopping=True)

    response = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
