import os
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import PyPDF2
import time
import jwt
# Import the Vapi Python Library and error class.
from vapi import Vapi
from vapi.core.api_error import ApiError

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

def generate_jwt():
    api_key = os.getenv("VITE_VAPI_API_TOKEN")
    org_id = os.getenv("ORG_ID")
    
    if not api_key or not org_id:
        raise ValueError("Missing API key or ORG_ID in environment variables.")
    
    payload = {
        "orgId": org_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
        "iss": "vapi",
        "aud": "vapi"
    }
    
    token = jwt.encode(payload, api_key)
    return f"Bearer {token}"

# Helper function to get headers with JWT token.
def get_auth_headers():
    return {"Authorization": generate_jwt()}

# Home route: Serve the integrated HTML page.
@app.route("/")
def index():
    public_key = os.getenv("VITE_VAPI_PUBLIC_KEY")
    api_key = os.getenv("VITE_VAPI_API_TOKEN")
    return render_template("index.html", public_key=public_key, api_token=api_key)

# Endpoint to extract text from a PDF file.
@app.route("/extract", methods=["POST"])
def extract_pdf_text():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    try:
        headers = get_auth_headers()
        pdf_reader = PyPDF2.PdfReader(file)
        extracted_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
        return jsonify({"text": extracted_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to start a Vapi call using the server SDK.
@app.route("/start-call", methods=["POST"])
def start_call():
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON payload"}), 400
    context_text = data.get("text", "")
    try:
        headers = get_auth_headers()
        # Use the token directly from the header.
        client = Vapi(token=headers["Authorization"])
        call = client.calls.create(context_text)
        return jsonify({"call_id": call.id})
    except ApiError as e:
        return jsonify({"error": str(e), "status_code": e.status_code}), e.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to fetch call details.
@app.route("/call-details", methods=["GET"])
def get_call_details():
    call_id = request.args.get("call_id")
    if not call_id:
        return jsonify({"error": "Call ID is required"}), 400
    try:
        headers = get_auth_headers()
        url = f"https://api.vapi.ai/call/{call_id}"
        response = requests.get(url, headers=headers)
        return jsonify(response.json()), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
