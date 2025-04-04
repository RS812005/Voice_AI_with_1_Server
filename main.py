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
import uuid


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
        with open(HISTORY_FILE, "r") as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


# Helper function to write history to JSON file
def write_history(history_data):
    with open(HISTORY_FILE, "w") as file:
        json.dump(history_data, file, indent=2, default=str)


# Helper function to add a new record to history
def add_history_record(prompt, prompt_summary, public_survey_id="", call_summary=""):
    history = read_history()

    # # Find the highest existing ID
    # max_id = 0
    # for record in history:
    #     if record.get("id") and int(record["id"]) > max_id:
    #         max_id = int(record["id"])

    # Create new record with ID one higher than the max
    new_record = {
        "id": str(uuid.uuid4()),  # Generate a unique UUID for the ID,,
        "public_survey_id": public_survey_id,  # Generates a unique public survey id.
        "prompt": prompt or "",
        "prompt_summary": prompt_summary or "",
        "call_summary": call_summary,
        "created_at": datetime.utcnow().isoformat(),
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


# --- New Chat with Survey JSON Helper Functions ---
CHAT_SURVEY_FILE = "chat_with_survey.json"


def read_chat_survey():
    if not os.path.exists(CHAT_SURVEY_FILE):
        return []
    try:
        with open(CHAT_SURVEY_FILE, "r") as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def write_chat_survey(data):
    with open(CHAT_SURVEY_FILE, "w") as file:
        json.dump(data, file, indent=2, default=str)


def add_chat_survey_record(survey_prompt, survey_response, public_survey_id):
    records = read_chat_survey()
    new_record = {
        "id": str(uuid.uuid4()),  # Generate a unique UUID for the ID,
        "public_survey_id": public_survey_id,  # New field added, initialized to an empty string
        "survey_prompt": survey_prompt,
        "survey_response": survey_response,
        "summary": "",
        "chat_with_survey": [  # start with initial bot response (the survey response)
            {"role": "bot", "message": survey_response}
        ],
        "raw_transcript": "Bot: " + survey_response,
        "created_at": datetime.utcnow().isoformat(),
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


def generate_jwt():
    api_key = os.getenv("VITE_VAPI_API_TOKEN")  # API_CHANGE
    org_id = os.getenv("ORG_ID")  # API_CHANGE

    if not api_key or not org_id:
        raise ValueError("Missing API key or ORG_ID in environment variables.")

    payload = {
        "orgId": org_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
        "iss": "vapi",
        "aud": "vapi",
    }

    token = jwt.encode(payload, api_key)
    return f"Bearer {token}"


# Helper function to get headers with JWT token.
def get_auth_headers():
    return {"Content-Type": "application/json", "Authorization": generate_jwt()}


# Home route: Serve the integrated HTML page.
@app.route("/")
def index():
    public_key = os.getenv("VITE_VAPI_PUBLIC_KEY")  # API_CHANGE
    api_key = os.getenv("VITE_VAPI_API_TOKEN")  # API_CHANGE
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
    headers = get_auth_headers()
    print(headers)
    try:
        
        # Use the token directly from the header.
        client = Vapi(token=headers["Authorization"])
        call = client.calls.create(context_text)
        return jsonify({"call_id": call.id})
    except ApiError as e:
        return jsonify({"error": str(e), "status_code": e.status_code}), e.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/call-details", methods=["POST"])
def get_call_details():
    # Get call_id from URL query parameters.
    call_id = request.args.get("call_id")
    # Get record_id from the JSON body.
    data = request.get_json()
    record_id = data.get("record_id") if data else None

    if not call_id:
        return jsonify({"error": "Call ID is required"}), 400
    if not record_id:
        return jsonify({"error": "Record ID is required in the request body"}), 400

    try:
        # headers = get_auth_headers()
        headers = {
            "Authorization": "Bearer " + os.getenv("VITE_VAPI_API_TOKEN"),
            "Content-Type": "application/json",
        }

        url = f"https://api.vapi.ai/call/{call_id}"
        # print(headers)
        response = requests.get(url, headers=headers)
        data = response.json()

        # If the response contains an "analysis" object, update the history record for the provided record_id.
        if "analysis" in data:
            analysis = data["analysis"]
            history = read_history()
            # try:
            #     record_id_int = int(record_id)
            # except ValueError:
            #     return jsonify({"error": "Invalid record ID"}), 400

            # Find the matching record by ID.
            record = next((rec for rec in history if rec.get("id") == record_id), None)
            if record:
                if "summary" in analysis:
                    record["call_summary"] = analysis["summary"]
                if "structuredData" in analysis:
                    record["structured_data"] = analysis["structuredData"]
                    record["employee_id"] = analysis["structuredData"].get(
                        "employee_id"
                    )
                write_history(history)
            else:
                return jsonify({"error": "Record not found in history"}), 404

        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analysis")
def analysis_route():
    # Load all history records from history.json
    records = read_history()

    # Define the parameters we're interested in
    parameters = [
        "happiness",
        "mental_health",
        "job_satisfaction",
        "enps",
        "communication",
    ]
    # Initialize an aggregated count dictionary for each parameter with ratings 1 to 5
    aggregated = {
        param: {str(rating): 0 for rating in range(1, 6)} for param in parameters
    }

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
    return render_template(
        "analysis.html", aggregated=aggregated, total_users=total_users
    )


import re
import json


@app.route("/groq-chat", methods=["POST"])
def groq_chat():
    try:
        # Get the prompt and survey_length from the request payload.
        data = request.get_json()
        prompt = data.get("prompt")
        survey_length = data.get(
            "survey_length", "medium"
        )  # Default to medium if not provided.
        if not prompt:
            return jsonify({"error": "Missing prompt in request"}), 400

        # Define instructions based on survey length.
        if survey_length == "short":
            length_instruction = (
                "Please generate a short survey (around 2 minutes long)."
            )
        elif survey_length == "medium":
            length_instruction = (
                "Please generate a medium-length survey (3-5 minutes long)."
            )
        elif survey_length == "long":
            length_instruction = (
                "Please generate a detailed survey that is long (over 5 minutes)."
            )
        else:
            length_instruction = ""

        # UPDATED: Include intro, summary, questions, and outro in the JSON schema instruction.
        json_schema_instruction = (
            "Please generate a structured survey in JSON format using the following schema:\n\n"
            "json:\n"
            "{\n"
            '  "survey_title": "",\n'
            '  "intro": "",\n'
            '  "summary": "",\n'
            '  "questions": [\n'
            "    {\n"
            '      "id": 1,\n'
            '      "question": "",\n'
            '      "type": "multiple_choice", // Can be "multiple_choice", "short_answer", "paragraph", "rating"\n'
            '      "options": ["", "", "", ""], // Only for multiple_choice or rating\n'
            '      "answer": ""\n'
            "    }\n"
            "  ],\n"
            '  "outro": ""\n'
            "}\n\n"
            "Include a mix of multiple-choice, short-answer, paragraph, and rating questions. Also, generate an 'intro' that explains the purpose of the survey, a 'summary' that briefly recaps the survey's focus, and an 'outro' that thanks the user. Ensure that the survey_title is informative and derived from the initial prompt. Return only the JSON content without additional commentary."
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
        survey_response_text = survey_response_completion.choices[
            0
        ].message.content.strip()

        # Remove Markdown code fences if present.
        survey_response_text = re.sub(r"^```(?:json)?\n", "", survey_response_text)
        survey_response_text = re.sub(r"\n```$", "", survey_response_text)

        # Extract only the JSON block from the response.
        json_match = re.search(r"(\{.*\})", survey_response_text, re.DOTALL)
        if json_match:
            json_content = json_match.group(1)
        else:
            return (
                jsonify(
                    {
                        "error": "No valid JSON found in response",
                        "raw_response": survey_response_text,
                    }
                ),
                500,
            )

        # Parse the cleaned survey response text into a JSON object.
        try:
            structured_survey = json.loads(json_content)
        except Exception as parse_error:
            return (
                jsonify(
                    {
                        "error": f"Failed to parse survey JSON: {parse_error}",
                        "raw_response": json_content,
                    }
                ),
                500,
            )

        public_survey_id = str(uuid.uuid4())
        # Optionally, add a new record to your JSON history and capture the new record.
        new_record = add_history_record(prompt, json_content, public_survey_id)
        new_chat_record = add_chat_survey_record(prompt, json_content, public_survey_id)

        # Return the structured survey response, including intro, summary, questions, and outro, along with record id.
        return jsonify(
            {
                "survey": structured_survey,
                "record_id": new_record["id"],
                "public_survey_id": new_record["public_survey_id"],
                "chatRecordId": new_chat_record["id"],
            }
        )
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
        records = records[:20]

        # Format records for display, including the record id
        formatted_records = []
        for record in records:
            formatted_records.append(
                {
                    "id": record.get("id", ""),
                    "public_survey_id": record.get(
                        "public_survey_id", ""
                    ),  # Add this line
                    "prompt": record.get("prompt", ""),
                    "prompt_summary": record.get("prompt_summary", ""),
                    "call_summary": record.get("call_summary", ""),
                    "structured_data": record.get("structured_data","")
                }
            )

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


@app.route("/chat_with_survey", methods=["POST"])
def chat_with_survey():
    data = request.get_json()
    question = data.get("question")
    survey_response = data.get("survey_response", "")
    record_id = str(data.get("record_id", ""))
    public_survey_id = data.get("public_survey_id", "")

    if not question:
        return jsonify({"error": "Missing question"}), 400

    CHAT_FILE = "chat_with_survey.json"
    if os.path.exists(CHAT_FILE):
        try:
            with open(CHAT_FILE, "r") as f:
                chat_history = json.load(f)
        except Exception:
            chat_history = []
    else:
        chat_history = []

    # Find an existing conversation by record_id
    existing_record = next(
        (record for record in chat_history if record.get("id") == record_id), None
    )
    is_new_conversation = existing_record is None

    if is_new_conversation:
        if not record_id:
            record_id = str(uuid.uuid4())
        new_record = {
            "id": record_id,
            "employee_id": None,
            "public_survey_id": public_survey_id,
            "survey_response": survey_response,
            "survey_prompt": "",
            "summary": "",
            "chat_with_survey": [],
            "created_at": datetime.utcnow().isoformat(),
            "waiting_for_employee_id": True,
            "survey_started": False,
        }
        chat_history.append(new_record)
        current_record = new_record
    else:
        current_record = existing_record

    conversation = current_record["chat_with_survey"]
    employee_id = current_record.get("employee_id")
    waiting_for_employee_id = current_record.get("waiting_for_employee_id", False)
    survey_started = current_record.get("survey_started", False)

    # Append user's message to the conversation
    conversation.append(
        {
            "sender": "user",
            "message": question,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

    # Initialize Groq client
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    # If we don't have an employee ID or are waiting for one, try to extract it.
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
                                "description": "The employee ID mentioned by the user",
                            },
                            "found": {
                                "type": "boolean",
                                "description": "Whether an employee ID was found in the message",
                            },
                        },
                        "required": ["employee_id", "found"],
                    },
                },
            }
        ]

        # Use .get() to safely access keys in each message
        raw_transcript = "\n".join(
            [
                f"{msg.get('sender', 'unknown')}: {msg.get('message', '')}"
                for msg in conversation
            ]
        )
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

            if hasattr(response_message, "tool_calls") and response_message.tool_calls:
                tool_call = response_message.tool_calls[0]
                function_args = json.loads(tool_call.function.arguments)
                extracted_employee_id = function_args.get("employee_id", "")
                found = function_args.get("found", False)

                if found and extracted_employee_id.strip():
                    # Update the current record directly
                    current_record["employee_id"] = extracted_employee_id
                    current_record["waiting_for_employee_id"] = False

                    dynamic_intro_prompt = (
                        f"Based on the following survey details:\n\n{survey_response}\n\n"
                        "Generate a brief, friendly introduction that explains what this survey is about in a conversational tone and also tell that their responses are confidential"
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
                    current_record["waiting_for_employee_id"] = True
                    is_repeat = any(
                        msg.get("sender") == "bot"
                        and "provide your employee ID" in msg.get("message", "")
                        for msg in conversation
                    )

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
                current_record["waiting_for_employee_id"] = True
                response_text = (
                    "Welcome to our survey! To get started, please provide your employee ID number. "
                    "This helps us keep track of your responses."
                )

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    else:
        # Employee ID is already captured.
        if not survey_started:
            if question.strip().lower() in ["yes", "y", "ready", "ok"]:
                current_record["survey_started"] = True
                survey_started = True
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
            # Use .get() for safety when accessing conversation messages
            transcript_lines = [
                f"{msg.get('sender', 'unknown')}: {msg.get('message', '')}"
                for msg in conversation
            ]
            raw_transcript = (
                current_record.get("survey_response", "")
                + "\n"
                + "\n".join(transcript_lines)
            )

            prompt = (
                f"Based on the following survey conversation:\n\n{raw_transcript}\n\n"
                "You are provided with a survey that must be followed exactly in order. "
                "Do not skip any questions. Begin with the first unanswered question. "
                "Strictly ask each question in a conversational, warm, and empathetic tone. "
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

    # Append the bot's response to the conversation
    conversation.append(
        {
            "sender": "bot",
            "message": response_text,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

    # Save the updated chat history back to the file
    with open(CHAT_FILE, "w") as f:
        json.dump(chat_history, f, indent=2, default=str)

    return jsonify({"response": response_text, "record_id": record_id})


@app.route("/stop-survey", methods=["POST"])
def stop_survey():
    data = request.get_json()
    record_id = str(data.get("record_id"))
    # print("record_id", record_id)
    if not record_id:
        return jsonify({"error": "Missing record id"}), 400

    CHAT_FILE = "chat_with_survey.json"

    # Load existing chat history as an array
    if os.path.exists(CHAT_FILE):
        try:
            with open(CHAT_FILE, "r") as f:
                chat_history = json.load(f)
        except Exception:
            chat_history = []
    else:
        chat_history = []

    # Find the record in the chat history with the matching record_id
    record = next((rec for rec in chat_history if rec.get("id") == record_id), None)
    if not record:
        return jsonify({"error": "Chat record not found"}), 404

    # Build the raw transcript using the stored survey_response and chat conversation.
    conversation = record.get("chat_with_survey", [])
    raw_transcript = (
        record.get("survey_response", "")
        + "\n"
        + "\n".join(
            [
                f"{msg.get('sender', 'unknown')}: {msg.get('message', '')}"
                for msg in conversation
            ]
        )
    )

    # Update the prompt to enforce a JSON-only response.
    summary_prompt = (
        "Summarize the following survey conversation into the JSON schema below. "
        "Ensure that your entire response is strictly valid JSON with no additional text or explanation.\n\n"
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

        # Validate that the output is strictly valid JSON.
        try:
            summary_json = json.loads(summary_text)
        except Exception as e:
            return jsonify({"error": f"Invalid JSON generated: {str(e)}"}), 500

        # Store the JSON summary in the record
        record["summary"] = summary_json

        # Save the updated chat history back to the file
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
    public_survey_id = data.get(
        "public_survey_id", ""
    )  # Extract public_survey_id from the request body
    # print("Public survey id:", public_survey_id)
    if not recordid:
        return jsonify({"error": "Missing recordid"}), 400

    # Read history records from the JSON file.
    history = read_history()
    # Search for the record with matching ID.
    record = next((rec for rec in history if str(rec.get("id")) == str(recordid)), None)
    if not record:
        return jsonify({"error": "Record not found"}), 404

    # Create a new history record with the provided public_survey_id.
    new_record = add_history_record(
        prompt="", prompt_summary="", public_survey_id=public_survey_id
    )
    return jsonify(
        {
            "prompt_summary": record.get("prompt_summary", ""),
            "record_id": new_record["id"],
        }
    )


@app.route("/public-survey-chat-survey", methods=["POST"])
def public_survey_chat_survey():
    data = request.get_json()
    recordid = data.get("recordid")
    # chatrecordid = data.get("chatrecordid")
    if not recordid:
        return jsonify({"error": "Missing recordid or chatrecordid"}), 400
    # Read the chat history file
    # Read the chat history file
    # chat_history = read_chat_survey()

    # # Filter keys that are valid integers
    #     valid_keys = [int(key) for key in chat_history.keys() if key is not None and key.isdigit()]

    # # Determine new chatrecordid by incrementing the highest existing key
    #     max_chat_id = max(valid_keys) if valid_keys else 0
    #     new_chatrecordid = str(max_chat_id + 1)

    # Generate a new unique chat record ID using uuid
    new_chatrecordid = str(uuid.uuid4())

    # Read history records from the JSON file.
    history = read_history()
    # Search for the record with matching ID.
    record = next((rec for rec in history if str(rec.get("id")) == str(recordid)), None)
    if not record:
        return jsonify({"error": "Record not found"}), 404

    # Return the prompt_summary for that record.
    return jsonify(
        {
            "prompt_summary": record.get("prompt_summary", ""),
            "chatrecordid": new_chatrecordid,
        }
    )


def read_json_file(filepath):
    """Read JSON file and return its data, or an empty structure if not found."""
    if not os.path.exists(filepath):
        return []  # or {} if you expect a dict
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        print("Error reading", filepath, ":", e)
        return []


@app.route("/get_rating_distribution", methods=["GET"])
def get_rating_distribution():
    # Optionally filter responses by public_survey_id if provided
    public_survey_id = request.args.get("public_survey_id")

    # Initialize the distribution dictionary.
    # It will be of the form:
    # { "HAPPINESS": {"0": count, "3": count, ...}, "MENTAL_HEALTH": { ... }, ... }
    distribution = {}

    # Process chat_with_survey.json
    chat_data = read_json_file(CHAT_SURVEY_FILE)
    # chat_with_survey.json is stored as a dict; iterate over its values.
    if isinstance(chat_data, dict):
        chat_records = chat_data.values()
    else:
        chat_records = chat_data

    for record in chat_records:
        if not isinstance(record, dict):
            continue
        if public_survey_id and record.get("public_survey_id") != public_survey_id:
            continue
        summary = record.get("summary")
        if isinstance(summary, dict):
            ratings = summary.get("ratings", {})
            for parameter, info in ratings.items():
                rating = info.get("rating")
                if rating is None:
                    continue
                param_key = parameter.upper()
                distribution.setdefault(param_key, {})
                rating_key = str(rating)
                distribution[param_key][rating_key] = (
                    distribution[param_key].get(rating_key, 0) + 1
                )

    # Process history.json (which is stored as a list)
    history_data = read_json_file(HISTORY_FILE)
    for record in history_data:
        if not isinstance(record, dict):
            continue
        if public_survey_id and record.get("public_survey_id") != public_survey_id:
            continue
        structured = record.get("structured_data")
        if isinstance(structured, dict):
            ratings = structured.get("ratings", {})
            for parameter, info in ratings.items():
                rating = info.get("rating")
                if rating is None:
                    continue
                param_key = parameter.upper()
                distribution.setdefault(param_key, {})
                rating_key = str(rating)
                distribution[param_key][rating_key] = (
                    distribution[param_key].get(rating_key, 0) + 1
                )

    # Debug print to see the aggregated counts (optional)
    # print("Distribution:", distribution)
    return jsonify({"distribution": distribution})


@app.route("/manager")
def manager_page():
    # public_key = os.getenv("VITE_VAPI_PUBLIC_KEY")  # Or any other env variable
    # api_key = os.getenv("VITE_VAPI_API_TOKEN")
    return render_template("manager.html")


@app.route("/manager", methods=["POST"])
def manager_endpoint():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    public_survey_id = data.get("public_survey_id")
    record_id = data.get("record_id")
    print(record_id, public_survey_id)
    if not public_survey_id or not record_id:
        return jsonify({"error": "public_survey_id and record_id are required"}), 400

    # Load the history records
    history = read_history()

    # Search for the matching record
    found_record = None
    for record in history:
        # Make sure the record has both fields and they match
        if record.get("public_survey_id") == public_survey_id and str(
            record.get("id")
        ) == str(record_id):
            found_record = record
            break

    if not found_record:
        return jsonify({"error": "Record not found"}), 404

    # Return the prompt_summary for that record
    prompt_summary = found_record.get("prompt_summary", "")
    return jsonify({"prompt_summary": prompt_summary}), 200


def search_json_all(file_path, public_survey_id):
    """
    Searches the given JSON file and returns a list of all entries with a matching public_survey_id.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
    with open(file_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON from {file_path}: {e}")
            return []

    matching_entries = []
    # Expecting data to be a list of objects.
    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                try:
                    entry = json.loads(entry)
                except Exception as ex:
                    print(f"Skipping non-dict entry: {ex}")
                    continue
            if entry.get("public_survey_id") == public_survey_id:
                matching_entries.append(entry)
    elif isinstance(data, dict):
        if data.get("public_survey_id") == public_survey_id:
            matching_entries.append(data)
    return matching_entries


@app.route("/manager_analysis", methods=["POST"])
def manager_analysis():
    data = request.get_json()
    public_survey_id = data.get("public_survey_id")
    if not public_survey_id:
        return jsonify({"error": "Missing public_survey_id"}), 400

    # Get all matching entries from both files
    history_entries = search_json_all("history.json", public_survey_id)
    chat_entries = search_json_all("chat_with_survey.json", public_survey_id)

    if not history_entries and not chat_entries:
        print("No data found for the given public_survey_id")
        return jsonify({"error": "No data found for the given public_survey_id"}), 404

    # Define the parameters for which to compute averages.
    parameters = [
        "happiness",
        "mental_health",
        "job_satisfaction",
        "enps",
        "communication",
    ]
    # This dict will collect all ratings per parameter.
    rating_values = {param: [] for param in parameters}

    # Process history entries: ratings are located in structured_data.ratings
    for entry in history_entries:
        structured_data = entry.get("structured_data", {})
        ratings = structured_data.get("ratings", {})
        for param in parameters:
            if param in ratings and isinstance(ratings[param], dict):
                rating = ratings[param].get("rating")
                if isinstance(rating, (int, float)):
                    rating_values[param].append(rating)

    # Process chat entries: ratings are located in summary.ratings
    for entry in chat_entries:
        summary = entry.get("summary", {})
        # If summary is empty or a blank string, default to empty dict.
        if not summary:
            summary = {}
        # If summary is a string, attempt to parse it as JSON
        elif isinstance(summary, str):
            try:
                summary = json.loads(summary)
                print("Parsed chat summary string into JSON.")
            except Exception as e:
                print(f"Error parsing chat summary: {str(e)}")
                summary = {}

        chat_ratings = summary.get("ratings", {})
        for param in parameters:
            if param in chat_ratings and isinstance(chat_ratings[param], dict):
                rating = chat_ratings[param].get("rating")
                if isinstance(rating, (int, float)):
                    rating_values[param].append(rating)

    # Compute average ratings for each parameter.
    averages = {}
    for param in parameters:
        ratings = rating_values[param]
        averages[param] = sum(ratings) / len(ratings) if ratings else None

    return jsonify(averages)


def search_all_chat_entries_with_ratings(file_path, public_survey_id):
    """
    Searches the given JSON file and returns a list of dictionaries, each containing:
      - chat_with_survey: the list of chat messages
      - ratings: the summary.ratings dictionary (if present)
      - jsoncontent: an object containing summary (text), ratings, and next_steps from the summary
    for entries matching the given public_survey_id.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error loading JSON from {file_path}: {e}")
        return []

    results = []

    # Function to safely parse summary data
    def parse_summary(summary_field):
        if not summary_field:
            return {}
        if isinstance(summary_field, dict):
            return summary_field
        # If it's a string, attempt to parse JSON
        if isinstance(summary_field, str):
            try:
                return json.loads(summary_field)
            except json.JSONDecodeError:
                # If parsing fails, return an empty dict
                return {}
        # If it's neither dict nor string, return empty
        return {}

    # Handle if data is a list of objects
    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                try:
                    entry = json.loads(entry)
                except Exception as ex:
                    print(f"Skipping non-dict entry: {ex}")
                    continue

            if entry.get("public_survey_id") == public_survey_id:
                chat_data = entry.get("chat_with_survey", [])

                # Safely parse the summary
                summary_data = parse_summary(entry.get("summary", {}))
                ratings = summary_data.get("ratings", {})

                jsoncontent = {
                    "summary": summary_data.get("summary", ""),
                    "ratings": ratings,
                    "next_steps": summary_data.get("next_steps", ""),
                }

                results.append(
                    {
                        "chat_with_survey": chat_data,
                        "ratings": ratings,
                        "jsoncontent": jsoncontent,
                    }
                )

    # Handle if data is a single dictionary
    elif isinstance(data, dict):
        if data.get("public_survey_id") == public_survey_id:
            chat_data = data.get("chat_with_survey", [])

            summary_data = parse_summary(data.get("summary", {}))
            ratings = summary_data.get("ratings", {})

            jsoncontent = {
                "summary": summary_data.get("summary", ""),
                "ratings": ratings,
                "next_steps": summary_data.get("next_steps", ""),
            }

            results.append(
                {
                    "chat_with_survey": chat_data,
                    "ratings": ratings,
                    "jsoncontent": jsoncontent,
                }
            )

    return results


@app.route("/chat-with-survey-responses", methods=["POST"])
def chat_with_survey_responses():
    req_data = request.get_json()
    public_survey_id = req_data.get("public_survey_id")
    if not public_survey_id:
        return jsonify({"error": "Missing public_survey_id"}), 400

    matching_entries = search_all_chat_entries_with_ratings(
        "chat_with_survey.json", public_survey_id
    )
    if not matching_entries:
        return (
            jsonify(
                {"error": "No chat responses found for the given public_survey_id"}
            ),
            404,
        )

    return jsonify(matching_entries)


@app.route("/voice-ai-response", methods=["POST"])
def voice_ai_response():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON payload"}), 400

    public_survey_id = data.get("public_survey_id")
    if not public_survey_id:
        return jsonify({"error": "Missing public_survey_id parameter"}), 400

    try:
        # Load history.json file (adjust path as needed)
        with open("history.json", "r") as f:
            history = json.load(f)
    except Exception as e:
        return jsonify({"error": f"Error loading history.json: {str(e)}"}), 500

    # Filter the history for matching conversations
    matching_conversations = [
        conv for conv in history if conv.get("public_survey_id") == public_survey_id
    ]

    if not matching_conversations:
        return jsonify({"message": "No matching conversations found."}), 404

    results = []
    for conv in matching_conversations:
        # Instead of using the outer summary, go inside structured_data
        structured = conv.get("structured_data", {})

        overall = structured.get("overall", "No overall available")
        summary = structured.get("summary", "No summary available")

        ratings_data = structured.get("ratings", {})
        extracted_ratings = {}
        for key, rating_obj in ratings_data.items():
            rating_value = (
                rating_obj.get("rating") if isinstance(rating_obj, dict) else None
            )
            extracted_ratings[key] = rating_value

        results.append(
            {"overall": overall, "ratings": extracted_ratings, "summary": summary}
        )

    return jsonify(results), 200


@app.route("/chat-with-survey-manager", methods=["POST"])
def chat_with_survey_manager():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided."}), 400

        chat_with_survey = data.get("chat_with_survey")
        prompt_summary = data.get("prompt_summary")
        summary = data.get("summary", "No summary available")
        messages = data.get("messages", [])

        if not chat_with_survey or not prompt_summary:
            return (
                jsonify({"error": "Missing chat_with_survey or prompt_summary."}),
                400,
            )

        # Create system message with context including the additional summary field
        system_context = (
            "You are a helpful manager analyzing employee survey results. Use this context:\n"
            f"Survey Summary: {summary}\n"
            f"Survey Structure: {prompt_summary}\n"
            f"Survey Responses: {chat_with_survey}"
        )

        # Start with the system message
        formatted_messages = [{"role": "system", "content": system_context}]

        # Validate and append messages from the frontend
        if not isinstance(messages, list):
            return jsonify({"error": "Messages must be a list."}), 400

        for message in messages:
            if (
                not isinstance(message, dict)
                or "role" not in message
                or "content" not in message
            ):
                return (
                    jsonify(
                        {
                            "error": "Each message must be an object with 'role' and 'content'."
                        }
                    ),
                    400,
                )
            formatted_messages.append(message)

        app.logger.info("Formatted messages for Groq API: %s", formatted_messages)

        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        chat_completion = client.chat.completions.create(
            messages=formatted_messages,
            model="llama-3.3-70b-versatile",  # Fast model, good for this use case
            temperature=0.3,
            max_tokens=1024,
        )

        response_text = chat_completion.choices[0].message.content
        usage_info = chat_completion.usage

        # Convert the usage object to a JSON-serializable dict
        try:
            usage_info = json.loads(
                json.dumps(usage_info, default=lambda o: o.__dict__)
            )
        except Exception as conv_err:
            usage_info = str(usage_info)

        return jsonify({"response": response_text, "usage": usage_info})

    except Exception as e:
        app.logger.error("Error in /chat-with-survey-manager: %s", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/voiceai-manager", methods=["POST"])
def voiceai_manager():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided."}), 400

        voice_ai_data = data.get("voice_ai_data")
        prompt_summary = data.get("prompt_summary")
        summary = data.get("summary", "No summary available")
        messages = data.get("messages", [])

        if not voice_ai_data or not prompt_summary:
            return jsonify({"error": "Missing voice_ai_data or prompt_summary."}), 400

        # Create system message with context for Voice AI responses
        system_context = (
            "You are a helpful voice assistant analyzing voice AI responses. Use this context:\n"
            f"Response Summary: {summary}\n"
            f"Voice AI Prompt Summary: {prompt_summary}\n"
            f"Voice AI Data: {voice_ai_data}"
        )

        # Start with the system message
        formatted_messages = [{"role": "system", "content": system_context}]

        # Validate and append messages from the frontend
        if not isinstance(messages, list):
            return jsonify({"error": "Messages must be a list."}), 400

        for message in messages:
            if (
                not isinstance(message, dict)
                or "role" not in message
                or "content" not in message
            ):
                return (
                    jsonify(
                        {
                            "error": "Each message must be an object with 'role' and 'content'."
                        }
                    ),
                    400,
                )
            formatted_messages.append(message)

        app.logger.info(
            "Formatted messages for Groq API (voice): %s", formatted_messages
        )

        # Initialize the Groq client and get a response
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        chat_completion = client.chat.completions.create(
            messages=formatted_messages,
            model="llama-3.3-70b-versatile",  # Fast model, good for this use case
            temperature=0.3,
            max_tokens=1024,
        )

        response_text = chat_completion.choices[0].message.content
        usage_info = chat_completion.usage

        # Convert the usage object to a JSON-serializable format
        try:
            usage_info = json.loads(
                json.dumps(usage_info, default=lambda o: o.__dict__)
            )
        except Exception as conv_err:
            usage_info = str(usage_info)

        return jsonify({"response": response_text, "usage": usage_info})

    except Exception as e:
        app.logger.error("Error in /voiceai-manager: %s", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/mega-chat-manager", methods=["POST"])
def mega_chat_manager():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided."}), 400

        survey_chat_data = data.get("survey_chat_data")
        voice_ai_data = data.get("voice_ai_data")
        prompt_summary = data.get("prompt_summary")
        messages = data.get("messages", [])

        if not survey_chat_data or not voice_ai_data or not prompt_summary:
            return (
                jsonify(
                    {
                        "error": "Missing one or more required fields: survey_chat_data, voice_ai_data, prompt_summary."
                    }
                ),
                400,
            )

        # Convert values to strings if they are dicts
        if isinstance(survey_chat_data, dict):
            survey_chat_data = json.dumps(survey_chat_data, indent=2)
        else:
            survey_chat_data = str(survey_chat_data)

        if isinstance(voice_ai_data, dict):
            voice_ai_data = json.dumps(voice_ai_data, indent=2)
        else:
            voice_ai_data = str(voice_ai_data)

        if isinstance(prompt_summary, dict):
            prompt_summary = json.dumps(prompt_summary, indent=2)
        else:
            prompt_summary = str(prompt_summary)

        # Construct the system context string
        system_context = (
            "You are an intelligent assistant that integrates data from multiple sources. Use this context:\n\n"
            "Survey Chat Data:\n"
            + survey_chat_data
            + "\n\n"
            + "Voice AI Data:\n"
            + voice_ai_data
            + "\n\n"
            + "Prompt Summary:\n"
            + prompt_summary
        )

        # Build the messages for Groq API
        formatted_messages = [{"role": "system", "content": system_context}]
        if not isinstance(messages, list):
            return jsonify({"error": "Messages must be a list."}), 400

        for message in messages:
            if (
                not isinstance(message, dict)
                or "role" not in message
                or "content" not in message
            ):
                return (
                    jsonify(
                        {
                            "error": "Each message must be an object with 'role' and 'content'."
                        }
                    ),
                    400,
                )
            formatted_messages.append(message)

        app.logger.info("Formatted messages for Mega Chat: %s", formatted_messages)

        # Call Groq API
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        chat_completion = client.chat.completions.create(
            messages=formatted_messages,
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1024,
        )

        response_text = chat_completion.choices[0].message.content
        usage_info = chat_completion.usage

        # Convert usage_info to a JSON-serializable object
        try:
            usage_info = json.loads(
                json.dumps(usage_info, default=lambda o: o.__dict__)
            )
        except Exception:
            usage_info = str(usage_info)

        return jsonify({"response": response_text, "usage": usage_info})

    except Exception as e:
        app.logger.error("Error in /mega-chat-manager: %s", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/public-get-heading", methods=["GET"])
def public_get_heading():
    record_id = request.args.get("recordid")
    if not record_id:
        return jsonify({"error": "Missing recordid parameter"}), 400
    try:
        records = read_history()
        # Find the matching record by record id
        matching_record = next(
            (record for record in records if str(record.get("id", "")) == record_id),
            None,
        )
        if not matching_record:
            return jsonify({"error": "Record not found"}), 404

        # Get prompt_summary; assume it contains a JSON string with a key "survey_title"
        prompt_summary = matching_record.get("prompt_summary", "")
        if not prompt_summary:
            return jsonify({"error": "Prompt summary not found"}), 404

        # Convert the prompt_summary to a dict (assuming it is a JSON string)
        try:
            summary_data = json.loads(prompt_summary)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid prompt summary format"}), 500

        survey_title = summary_data.get("survey_title", "")
        if not survey_title:
            return jsonify({"error": "survey_title not found in prompt_summary"}), 404

        return jsonify({"survey_title": survey_title})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/new-session", methods=["POST"])
def new_session():
    data = request.get_json() or {}
    survey = data.get("survey", "")
    survey_prompt = data.get("survey_prompt", "")
    # Create a new history record and a new chat survey record.
    # public_survey_id = str(uuid.uuid4())
    public_survey_id = data.get("public_survey_id")

    new_record = add_history_record(survey_prompt,survey, public_survey_id)
    # new_chat_record = add_chat_survey_record(survey_prompt, "", public_survey_id)
    return jsonify(
        {
            "record_id": new_record["id"],
            # "chatRecordId": new_chat_record["id"],
            "public_survey_id": new_record["public_survey_id"],
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
