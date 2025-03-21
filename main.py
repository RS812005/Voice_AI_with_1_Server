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

# Import for JSON file handling and datetime
import json
from datetime import datetime
import os.path

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# Define the path for our JSON storage file
HISTORY_FILE = "history.json"

# Helper function to read history from JSON file
def read_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r') as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

# Helper function to write history to JSON file
def write_history(history_data):
    with open(HISTORY_FILE, 'w') as file:
        json.dump(history_data, file, indent=2, default=str)

# Helper function to add a new record to history
def add_history_record(prompt, prompt_summary, call_summary=""):
    history = read_history()
    new_record = {
        "id": len(history) + 1,
        "prompt": prompt,
        "prompt_summary": prompt_summary,
        "call_summary": call_summary,
        "created_at": datetime.utcnow().isoformat()
    }
    history.append(new_record)
    write_history(history)
    return new_record

# Helper function to update a record in history
def update_history_record(record_id, updates):
    history = read_history()
    for record in history:
        if record["id"] == record_id:
            record.update(updates)
            break
    write_history(history)

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

@app.route("/call-details", methods=["GET"])
def get_call_details():
    call_id = request.args.get("call_id")
    if not call_id:
        return jsonify({"error": "Call ID is required"}), 400
    try:
        headers = get_auth_headers()
        url = f"https://api.vapi.ai/call/{call_id}"
        response = requests.get(url, headers=headers)
        data = response.json()
        
        # If the response contains an "analysis" object, update the most recent record
        if "analysis" in data:
            analysis = data["analysis"]
            history = read_history()
            if history:
                most_recent = history[-1]
                # Update the call summary if present in analysis
                if "summary" in analysis:
                    most_recent["call_summary"] = analysis["summary"]
                # Update structured data if present in analysis
                if "structuredData" in analysis:
                    most_recent["structured_data"] = analysis["structuredData"]
                    most_recent["employee_id"] = analysis["structuredData"].get("employee_id")

                write_history(history)
                
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/analysis")
def analysis_route():
    # Load all history records from history.json
    records = read_history()
    
    # Define the parameters we're interested in
    parameters = ["happiness", "mental_health", "job_satisfaction", "enps", "communication"]
    # Initialize an aggregated count dictionary for each parameter with ratings 1 to 5
    aggregated = { param: {str(rating): 0 for rating in range(1, 6)} for param in parameters }
    
    total_users = 0
    for record in records:
        total_users += 1
        structured = record.get("structured_data", {})
        ratings = structured.get("ratings", {})
        for param in parameters:
            if param in ratings:
                rating_value = ratings[param].get("rating")
                # Convert rating to an integer between 1 and 5, if valid
                if isinstance(rating_value, (int, float)) and 1 <= rating_value <= 5:
                    aggregated[param][str(int(rating_value))] += 1

    # Pass the aggregated data and total user count to analysis.html
    return render_template("analysis.html", aggregated=aggregated, total_users=total_users)


import re
import json

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

        # UPDATED: Include intro and outro in the JSON schema instruction.
        json_schema_instruction = (
            "Please generate a structured survey in JSON format using the following schema:\n\n"
            "json:\n"
            "{\n"
            "  \"survey_title\": \"\",\n"
            "  \"intro\": \"\",\n"
            "  \"questions\": [\n"
            "    {\n"
            "      \"id\": 1,\n"
            "      \"question\": \"\",\n"
            "      \"type\": \"multiple_choice\", // Can be 'multiple_choice', 'short_answer', 'paragraph', 'rating'\n"
            "      \"options\": [\"\", \"\", \"\", \"\"], // Only for multiple_choice or rating\n"
            "      \"answer\": \"\"\n"
            "    }\n"
            "  ],\n"
            "  \"outro\": \"\"\n"
            "}\n\n"
            "Include a mix of multiple-choice, short-answer, paragraph, and rating questions. "
            "Also, generate an 'intro' that explains the purpose of the survey, and an 'outro' that thanks the user. "
            "Ensure that the survey_title is informative and derived from the initial prompt. "
            "Return only the JSON content without additional commentary."
        )

        # Combine the prompt, length instruction, and JSON schema instruction.
        full_prompt = f"{prompt}\n\n{length_instruction}\n\n{json_schema_instruction}"

        # Initialize Groq client using the API key from environment variables.
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        # Groq API call to generate the structured survey.
        survey_response_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": full_prompt}],
            model="llama-3.3-70b-versatile",
        )
        survey_response_text = survey_response_completion.choices[0].message.content.strip()

        # Remove Markdown code fences if present.
        survey_response_text = re.sub(r"^```(?:json)?\n", "", survey_response_text)
        survey_response_text = re.sub(r"\n```$", "", survey_response_text)

        # Extract only the JSON block from the response.
        json_match = re.search(r'(\{.*\})', survey_response_text, re.DOTALL)
        if json_match:
            json_content = json_match.group(1)
        else:
            return jsonify({"error": "No valid JSON found in response", "raw_response": survey_response_text}), 500

        # Parse the cleaned survey response text into a JSON object.
        try:
            structured_survey = json.loads(json_content)
        except Exception as parse_error:
            return jsonify({
                "error": f"Failed to parse survey JSON: {parse_error}",
                "raw_response": json_content
            }), 500

        # Optionally, add a new record to your JSON history and capture the new record.
        new_record = add_history_record(prompt, json_content)

        # Return the structured survey response, including intro/outro, along with record id.
        return jsonify({
            "survey": structured_survey,
            "record_id": new_record["id"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500




    
# Endpoint to fetch history and send it to index.html.
@app.route("/history", methods=["GET"])
def history():
    try:
        # Get the last 10 records from our JSON file
        records = read_history()
        # Sort by created_at in descending order and take the last 10
        records.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        records = records[:10]
        
        # Format records for display, including the record id
        formatted_records = []
        for record in records:
            formatted_records.append({
                "id": record.get("id", ""),
                "prompt": record.get("prompt", ""),
                "prompt_summary": record.get("prompt_summary", ""),
                "call_summary": record.get("call_summary", "")
            })
            
        # Render the history template and pass the history records.
        return render_template("history.html", records=formatted_records)
    except Exception as e:
        return f"Error fetching history: {str(e)}", 500


# Endpoint to handle chat queries using the raw transcript context
@app.route("/chat-query", methods=["POST"])
def chat_query():
    data = request.get_json()
    question = data.get("question")
    raw_transcript = data.get("raw_transcript", "")
    if not question:
        return jsonify({"error": "Missing question"}), 400
    try:
        prompt = (
            f"Based on the following transcript:\n\n{raw_transcript}\n\n"
            f"Answer the following question in a friendly, conversational manner:\n{question}"
        )
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        response_text = chat_completion.choices[0].message.content
        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Chat page: load raw transcript from a specific history record based on record_id
@app.route("/chat")
def chat():
    record_id = request.args.get("record_id")
    raw_transcript = ""
    if record_id:
        history = read_history()
        for rec in history:
            if str(rec.get("id")) == record_id:
                # Extract raw transcript from structured_data if it exists
                if rec.get("structured_data", {}).get("raw_transcript"):
                    raw_transcript = rec["structured_data"]["raw_transcript"]
                break
    return render_template("chat.html", raw_transcript=raw_transcript)


# --- New Chat with Survey JSON Helper Functions ---
CHAT_SURVEY_FILE = "chat_with_survey.json"

def read_chat_survey():
    if not os.path.exists(CHAT_SURVEY_FILE):
        return []
    try:
        with open(CHAT_SURVEY_FILE, 'r') as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def write_chat_survey(data):
    with open(CHAT_SURVEY_FILE, 'w') as file:
        json.dump(data, file, indent=2, default=str)

def add_chat_survey_record(survey_prompt, survey_response):
    records = read_chat_survey()
    new_record = {
        "id": len(records) + 1,
        "survey_prompt": survey_prompt,
        "survey_response": survey_response,
        "summary": "",
        "chat_with_survey": [  # start with initial bot response (the survey response)
            {"role": "bot", "message": survey_response}
        ],
        "raw_transcript": "Bot: " + survey_response,
        "created_at": datetime.utcnow().isoformat()
    }
    records.append(new_record)
    write_chat_survey(records)
    return new_record

def update_chat_survey_record(record_id, updated_record):
    records = read_chat_survey()
    for i, rec in enumerate(records):
        if rec.get("id") == record_id:
            records[i] = updated_record
            break
    write_chat_survey(records)

@app.route("/chat_with_survey", methods=["POST"])
def chat_with_survey():
    data = request.get_json()
    question = data.get("question")
    survey_response = data.get("survey_response", "")
    # Get record_id from request or initialize as None if this is a new conversation
    record_id = str(data.get("record_id", ""))
    
    if not question:
        return jsonify({"error": "Missing question"}), 400

    CHAT_FILE = "chat_with_survey.json"
    if os.path.exists(CHAT_FILE):
        try:
            with open(CHAT_FILE, "r") as f:
                chat_history = json.load(f)
        except Exception:
            chat_history = {}
    else:
        chat_history = {}

    # Check if this is a new conversation or an existing one
    is_new_conversation = not record_id or record_id not in chat_history
    
    # For new conversations, initialize the record (including survey_started flag)
    if is_new_conversation:
        chat_history[record_id] = {
            "employee_id": None,  # Will be filled in later
            "survey_response": survey_response,
            "survey_prompt": "",
            "summary": "",
            "chat_with_survey": [],
            "created_at": datetime.utcnow().isoformat(),
            "waiting_for_employee_id": True,  # Waiting for employee ID
            "survey_started": False          # Flag to indicate if survey has begun
        }
    
    conversation = chat_history[record_id]["chat_with_survey"]
    employee_id = chat_history[record_id].get("employee_id")
    waiting_for_employee_id = chat_history[record_id].get("waiting_for_employee_id", False)
    survey_started = chat_history[record_id].get("survey_started", False)

    # Append user's message to the conversation
    conversation.append({
        "sender": "user",
        "message": question,
        "timestamp": datetime.utcnow().isoformat()
    })

    # Initialize Groq client
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    # If we don't have an employee ID yet or are waiting for one, try to extract it.
    if not employee_id or waiting_for_employee_id:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "extract_employee_id",
                    "description": "Extract the employee ID from the user's message. Employee IDs are typically alphanumeric strings.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "employee_id": {
                                "type": "string",
                                "description": "The employee ID mentioned by the user"
                            },
                            "found": {
                                "type": "boolean",
                                "description": "Whether an employee ID was found in the message"
                            }
                        },
                        "required": ["employee_id", "found"]
                    }
                }
            }
        ]
        
        raw_transcript = "\n".join([f"{msg['sender']}: {msg['message']}" for msg in conversation])
        prompt = (
            f"Based on the following conversation, extract the employee ID if mentioned:\n\n{raw_transcript}\n\n"
            "If the user has provided an employee ID (typically an alphanumeric code or number), extract it accurately. "
            "Otherwise, indicate that no employee ID was found with 'found': false."
        )
        
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                tools=tools,
                tool_choice="auto",
            )
            
            response_message = chat_completion.choices[0].message
            
            if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
                tool_call = response_message.tool_calls[0]
                function_args = json.loads(tool_call.function.arguments)
                extracted_employee_id = function_args.get("employee_id", "")
                found = function_args.get("found", False)
                
                if found and extracted_employee_id.strip():
                    chat_history[record_id]["employee_id"] = extracted_employee_id
                    chat_history[record_id]["waiting_for_employee_id"] = False
                    
                    # Generate a dynamic intro based on the survey details.
                    dynamic_intro_prompt = (
                        f"Based on the following survey details:\n\n{survey_response}\n\n"
                        "Generate a brief, friendly introduction that explains what this survey is about in a conversational tone."
                    )
                    intro_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": dynamic_intro_prompt}],
                        model="llama-3.3-70b-versatile",
                    )
                    dynamic_intro = intro_completion.choices[0].message.content.strip()

                    response_text = (
                        f"Thank you! I've recorded your employee ID: {extracted_employee_id}.\n\n"
                        f"{dynamic_intro}\n\n"
                        "Are you ready to begin? (Please reply 'yes' to start.)"
                    )
                else:
                    chat_history[record_id]["waiting_for_employee_id"] = True
                    is_repeat = any(msg["sender"] == "bot" and "provide your employee ID" in msg["message"] for msg in conversation)
                    
                    if is_repeat:
                        response_text = (
                            "I still need your employee ID to continue with the survey. "
                            "Please provide just your employee ID number or code (e.g., 'EMP1234' or '5678')."
                        )
                    else:
                        response_text = (
                            "Before we begin the survey, could you please provide your employee ID? "
                            "This helps us associate your responses with your records."
                        )
            else:
                chat_history[record_id]["waiting_for_employee_id"] = True
                response_text = (
                    "Welcome to our survey! To get started, please provide your employee ID number. "
                    "This helps us keep track of your responses."
                )
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    else:
        # Employee ID is already captured.
        # If the survey has not yet started, expect an affirmative ready confirmation.
        if not survey_started:
            if question.strip().lower() in ["yes", "y", "ready", "ok"]:
                # Mark the survey as started.
                chat_history[record_id]["survey_started"] = True
                survey_started = True
                # Proceed to ask the first survey question.
                survey_prompt = (
                    f"Based on the following survey:\n\n{survey_response}\n\n"
                    "Provide only the first question of the survey. "
                    "Make the question conversational, warm, and empathetic."
                )
                try:
                    first_question_response = client.chat.completions.create(
                        messages=[{"role": "user", "content": survey_prompt}],
                        model="llama-3.3-70b-versatile",
                    )
                    first_question = first_question_response.choices[0].message.content
                    response_text = f"Great! Let's begin:\n\n{first_question}"
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
            else:
                response_text = (
                    "Our survey is designed to gather your insights in a friendly, conversational manner. "
                    "When you're ready to begin, please reply with 'yes'."
                )
        else:
            # Survey already started: proceed with asking the next question.
            raw_transcript = chat_history[record_id]["survey_response"] + "\n" + "\n".join(
                [f"{msg['sender']}: {msg['message']}" for msg in conversation]
            )
            
            prompt = (
    f"Based on the following survey conversation:\n\n{raw_transcript}\n\n"
    "You are provided with a survey that must be followed exactly in order. "
    "Do not skip any questions. Begin with the first unanswered question. "
    "Ask each question in a conversational, warm, and empathetic tone. "
    "Adapt the wording as needed to be engaging and friendly, without prefacing the question with phrases like 'Now that we've talked' or 'Now that we've discussed'. "
    "If options are present, do not list them unless asked for. "
    "If unrelated topics are mentioned, respond: 'I'm here to assist with the survey. Let's stay focused on the questions.' "
    "Gently guide the user back if the conversation drifts."
)

            try:
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                )
                response_text = chat_completion.choices[0].message.content
            except Exception as e:
                return jsonify({"error": str(e)}), 500

    # Append the bot's response to the conversation.
    conversation.append({
        "sender": "bot",
        "message": response_text,
        "timestamp": datetime.utcnow().isoformat()
    })

    # Save the updated chat history.
    with open(CHAT_FILE, "w") as f:
        json.dump(chat_history, f, indent=2, default=str)

    return jsonify({"response": response_text, "record_id": record_id})


# New endpoint to stop the survey and generate a summary.
@app.route("/stop-survey", methods=["POST"])
def stop_survey():
    data = request.get_json()
    record_id = str(data.get("record_id"))
    print("record_id", record_id)
    if not record_id:
        return jsonify({"error": "Missing record id"}), 400

    CHAT_FILE = "chat_with_survey.json"

    # Load existing chat history or initialize an empty dict
    if os.path.exists(CHAT_FILE):
        try:
            with open(CHAT_FILE, "r") as f:
                chat_history = json.load(f)
        except Exception:
            chat_history = {}
    else:
        chat_history = {}

    if record_id not in chat_history:
        return jsonify({"error": "Chat record not found"}), 404

    record = chat_history[record_id]
    # Build the raw transcript using the stored survey_response and chat conversation.
    conversation = record.get("chat_with_survey", [])
    raw_transcript = record.get("survey_response", "") + "\n" + "\n".join(
        [f"{msg['sender']}: {msg['message']}" for msg in conversation]
    )

    summary_prompt = (
        "Summarize the following survey conversation into the JSON schema below:\n\n"
        """{
  "summary": "",
  "ratings": {
    "happiness": { "rating": 5, "comment": "" },
    "mental_health": { "rating": 4, "comment": "" },
    "job_satisfaction": { "rating": 3, "comment": "" },
    "enps": { "rating": 5, "comment": "" },
    "communication": { "rating": 4, "comment": "" }
  },
  "next_steps": "",
  "transcript_summary": {
    "highlights": ["", ""],
    "concerns": ["", ""],
    "feedback": ["", ""]
  },
  "raw_transcript": "",
  "overall": ""
}
"""
        f"\nConversation:\n{raw_transcript}"
    )

    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        summary_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": summary_prompt}],
            model="llama-3.3-70b-versatile",
        )
        summary_text = summary_completion.choices[0].message.content
        record["summary"] = summary_text
        chat_history[record_id] = record
        with open(CHAT_FILE, "w") as f:
            json.dump(chat_history, f, indent=2, default=str)
        return jsonify({"message": "Survey has been stopped."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    



@app.route("/public_survey")
def public_survey():
    return render_template("public_survey.html")


@app.route("/public-survey-voice-ai", methods=["POST"])
def public_survey_voice_ai():
    data = request.get_json()
    recordid = data.get("recordid")
    if not recordid:
        return jsonify({"error": "Missing recordid"}), 400

    # Read history records from the JSON file.
    history = read_history()
    # Search for the record with matching ID.
    record = next((rec for rec in history if str(rec.get("id")) == str(recordid)), None)
    if not record:
        return jsonify({"error": "Record not found"}), 404

    # Return the prompt_summary for that record.
    return jsonify({"prompt_summary": record.get("prompt_summary", "")})


@app.route("/public-survey-chat-survey", methods=["POST"])
def public_survey_chat_survey():
    data = request.get_json()
    recordid = data.get("recordid")
    # chatrecordid = data.get("chatrecordid")
    if not recordid :
        return jsonify({"error": "Missing recordid or chatrecordid"}), 400
    # Read the chat history file
    # Read the chat history file
    chat_history = read_chat_survey()

# Filter keys that are valid integers
    valid_keys = [int(key) for key in chat_history.keys() if key is not None and key.isdigit()]

# Determine new chatrecordid by incrementing the highest existing key
    max_chat_id = max(valid_keys) if valid_keys else 0
    new_chatrecordid = str(max_chat_id + 1)


    # Read history records from the JSON file.
    history = read_history()
    # Search for the record with matching ID.
    record = next((rec for rec in history if str(rec.get("id")) == str(recordid)), None)
    if not record:
        return jsonify({"error": "Record not found"}), 404

    # Return the prompt_summary for that record.
    return jsonify({"prompt_summary": record.get("prompt_summary", ""), "chatrecordid": new_chatrecordid})


if __name__ == "__main__":
    app.run(debug=True)
