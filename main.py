import os
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import PyPDF2
import time
import jwt
from groq import Groq
# Import the Vapi Python Library and error class.
from vapi import Vapi
from vapi.core.api_error import ApiError

# NEW: Import MongoClient and ObjectId from pymongo and datetime for timestamps
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# Expect your MongoDB URI to be in the environment variable MONGO_URI
MONGO_URI = os.getenv("MONGO_URI")
# Database name is 'lasso' and collection name is 'history'
mongo_client = MongoClient(MONGO_URI)
db_mongo = mongo_client["lasso"]
history_collection = db_mongo["history"]




def generate_jwt():
    api_key = os.getenv("VITE_VAPI_API_TOKEN") #API_CHANGE
    org_id = os.getenv("ORG_ID") #API_CHANGE
    
    if not api_key or not org_id:
        raise ValueError("Missing API key or ORG_ID in environment variables.")
    
    payload = {
        "orgId": org_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + 3500,
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
    public_key = os.getenv("VITE_VAPI_PUBLIC_KEY") #API_CHANGE
    api_key = os.getenv("VITE_VAPI_API_TOKEN") #API_CHANGE
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

# Endpoint to fetch call details and update the call summary.
@app.route("/call-details", methods=["GET"])
def get_call_details():
    call_id = request.args.get("call_id")
    # record_id = request.args.get("record_id")  # optional parameter from frontend
    if not call_id:
        return jsonify({"error": "Call ID is required"}), 400
    try:
        headers = get_auth_headers()
        url = f"https://api.vapi.ai/call/{call_id}"
        response = requests.get(url, headers=headers)
        data = response.json()
        # If the response contains a summary and a record_id is provided,
        # update the call_summary field.
        if "summary" in data:
    # Find the most recent record in the history collection
            recent_record = history_collection.find_one(sort=[("_id", -1)])
            if recent_record:
                history_collection.update_one(
            {"_id": recent_record["_id"]},  # Filter: update this specific document.
            {"$set": {"call_summary": data["summary"]}}  # Update document.
        )
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/groq-chat", methods=["POST"])
def groq_chat():
    try:
        # Get the prompt and survey_length from the request payload.
        data = request.get_json()
        prompt = data.get("prompt")
        survey_length = data.get("survey_length", "medium")  # Default to medium if not provided.
        if not prompt:
            return jsonify({"error": "Missing prompt in request"}), 400

        # Define instructions based on survey length.
        if survey_length == "short":
            length_instruction = "Please generate a short survey (around 2 minutes long)."
        elif survey_length == "medium":
            length_instruction = "Please generate a medium-length survey (3-5 minutes long)."
        elif survey_length == "long":
            length_instruction = "Please generate a detailed survey that is long (over 5 minutes)."
        else:
            length_instruction = ""

        # Combine the prompt with the survey length instruction.
        full_prompt = f"{prompt}\n\n{length_instruction}"

        # Initialize Groq client using the API key from environment variables.
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))  # API_CHANGE

        # Create chat completion using the provided full prompt.
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": full_prompt}],
            model="llama-3.3-70b-versatile",
        )

        # Extract the generated message.
        response_text = chat_completion.choices[0].message.content

        # Insert a new record into the history collection.
        # The prompt summary is stored in 'prompt_summary' and initially call_summary is empty.
        history_collection.insert_one({
            "prompt": prompt,
            "prompt_summary": response_text,
            "call_summary": "",
            "created_at": datetime.utcnow()
        })

        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Endpoint to fetch history and send it to index.html.
@app.route("/history", methods=["GET"])
def history():
    try:
        # Query the last 10 records from the history collection.
        records = list(history_collection.find().sort("_id", -1).limit(10))
        # Format records: convert ObjectId to string if needed.
        formatted_records = []
        for record in records:
            formatted_records.append({
                "prompt": record.get("prompt", ""),
                "prompt_summary": record.get("prompt_summary", ""),
                "call_summary": record.get("call_summary", "")
            })
        # Render the index template and pass the history records.
        return render_template("history.html", records=formatted_records)
    except Exception as e:
        return f"Error fetching history: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)
