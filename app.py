from flask import Flask, request, jsonify
import time
import re
from typing import List, Dict, Optional, Union
import json
from openai import OpenAI
import os
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
import uuid
import psycopg2
from psycopg2.extras import Json

app = Flask(__name__)

# TODO
# accept a full file
# make this a bunch of events that are shown in a timeline

# Load environment variables from .env file
load_dotenv()

# === Event and Session Models ===

@dataclass
class DrugAdministration:
    timestamp: float
    dose: float
    unit: str  # e.g., "mg", "mcg"
    route: str  # e.g., "IV", "IM"
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "dose": self.dose,
            "route": self.route
        }

@dataclass
class ActionInfo:
    action: str
    details: Optional[any] = None

class CodeBlueSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.iv_line_established: Optional[float] = None
        self.drug_admins: Dict[str, List[DrugAdministration]] = {
            "epinephrine": [],
            "amiodarone": [],
            "lidocaine": []
        }
        self.next_administer: Optional[float] = None
        self.pulse_cpr_shock: Dict[str, List[float]] = {
            "pulse": [],
            "cpr": [],
            "shock": []
        }
        self.timeline: List[Dict] = []


    def to_dict(self):
        return {
            "session_id": self.session_id,
            "iv_line_established": self.iv_line_established,
            "drug_admins": {
                drug: [admin.to_dict() for admin in admins]
                for drug, admins in self.drug_admins.items()
            },
            "next_administer": self.next_administer,
            "pulse_cpr_shock": self.pulse_cpr_shock,
            "timeline": self.timeline
        }

    def get_total_drug_dose(self, drug_name: str) -> float:
        """Calculate total dose of a specific drug administered."""
        return sum(admin.dose for admin in self.drug_admins[drug_name.lower()])

    def get_last_administration(self, drug_name: str) -> Optional[DrugAdministration]:
        """Get the most recent administration of a specific drug."""
        admins = self.drug_admins[drug_name.lower()]
        return admins[-1] if admins else None

# === In-memory store ===
session_store: Dict[str, CodeBlueSession] = {}
timelines = []


# === Action Definitions ===
VALID_ACTIONS = {
    "CODE_BLUE_START": {
        "description": "Code Blue activation",
        "examples": [
            "Code blue",
            "Activate code blue",
            "Starting code blue",
            "Code blue called",
            "Code blue activation",
            "Initiating code blue"
        ]
    },
    "COMPRESSION": {
        "description": "Chest compressions",
        "examples": [
            "Start compressions",
            "Beginning chest compressions",
            "Starting CPR",
            "Resume compressions",
            "Stop compressions",
            "Hold compressions",
            "Pause for pulse check"
        ]
    },
    "MEDICATION": {
        "description": "Medication administration",
        "examples": [
            "Give 1mg of epinephrine",
            "Push 300mg amiodarone",
            "Administering 100mg lidocaine",
            "1 amp of epi",
            "Give another round of epi",
            "Push amio"
        ]
    },
    "PULSE_CHECK": {
        "description": "Check for pulse",
        "examples": [
            "Check for pulse",
            "Pulse check",
            "Let's check rhythm",
            "Time for rhythm check",
            "Check carotid pulse"
        ]
    },
    "DEFIBRILLATE": {
        "description": "Defibrillation",
        "examples": [
            "Shock indicated",
            "Prepare to shock",
            "Charging",
            "Clear for shock",
            "Delivering shock",
            "Shock the patient"
        ]
    },
    "AIRWAY_MANAGEMENT": {
        "description": "Airway management procedures",
        "examples": [
            "Prepare for intubation",
            "Get the airway kit",
            "Starting bag valve mask",
            "Stop bagging",
            "Resume ventilation"
        ]
    },
    "EKG_CONNECTED": {
        "description": "Connect EKG/ECG monitor",
        "examples": [
            "Get the EKG",
            "Connect the monitor",
            "Place the leads",
            "Need EKG monitoring",
            "Put on the cardiac monitor"
        ]
    },
    "ESTABLISH_IV": {
        "description": "Establish IV access",
        "examples": [
            "Need IV access",
            "Starting an IV",
            "Get IV access",
            "Place an IV line",
            "Establishing IV access",
            "Need a line"
        ]
    },
    "ROSC": {
        "description": "Return of spontaneous circulation",
        "examples": [
            "Got a pulse",
            "Pulse is back",
            "ROSC achieved",
            "We have a rhythm",
            "Strong pulse present",
            "Carotid pulse palpable"
        ]
    }
}

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set")

def call_llm_for_sequence(events: List[Dict]) -> List[Dict]:
    """Use OpenAI to analyze the full sequence of events and return structured actions."""
    
    # Build the transcript text in chronological order
    transcript = "\n".join([
        f"{event['timestamp']} - {event['speaker']}: {event['text']}"
        for event in events
    ])

    prompt = f"""
You are a clinical assistant analyzing a Code Blue transcript to identify medical events and their sequence.
Given the following transcript, identify the sequence of events that occurred:

{transcript}

Rules for event sequencing:
1. No identical events can occur consecutively (e.g., can't have two pulse checks in a row)
2. Medications (epinephrine, amiodarone, lidocaine) and pulse checks occur at specific points in time
3. Compressions and airway management have start and stop times, typically interrupted by pulse checks
4. A pulse check must immediately follow when compressions are stopped
5. Code Blue activation, if present, must be the first event
6. ROSC (return of spontaneous circulation) ends the sequence - no events should occur after
7. For events with start/end times (like compressions), the timestamp should be the start time of the action

Valid event types:
- CODE_BLUE_START
- COMPRESSION (requires start_time and end_time)
- MEDICATION (requires drug name, dose, unit, and route)
- PULSE_CHECK
- AIRWAY_MANAGEMENT (requires start_time and end_time)
- DEFIBRILLATE
- EKG_CONNECTED
- ESTABLISH_IV
- ROSC

Return the events as a JSON array, with each event containing:
{{
    "timestamp": float,  // seconds from start of transcript, must match start_time for compression/airway events
    "type": string,     // one of the valid event types
    "details": {{        // specific details based on event type
        // For COMPRESSION and AIRWAY_MANAGEMENT:
        "start_time": float,  // must match the event timestamp
        "end_time": float | null,
        
        // For MEDICATION:
        "drug": string,
        "dose": number,
        "unit": string,
        "route": string
    }} | null
}}
"""

    try:
        print("Making OpenAI API call...")
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={ "type": "json_object" }
        )
        print("OpenAI response received")
        
        # Parse the response content
        try:
            print("Response content:", response.choices[0].message.content)
            parsed_response = json.loads(response.choices[0].message.content)
            print("Parsed JSON:", parsed_response)
            
            if "events" in parsed_response:
                events = parsed_response["events"]
            else:
                events = parsed_response
            
            print("Events to validate:", events)
            
            # Validate the sequence follows all rules
            if not validate_event_sequence(events):
                print("Event validation failed")
                return None
                
            return events
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None

    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None

def validate_event_sequence(events: List[Dict]) -> bool:
    """Validate that the event sequence follows all required rules."""
    # if not events:
    #     print("Validation failed: Empty events list")
    #     return False
        
    # # Check first event is CODE_BLUE_START if present
    # if any(e["type"] == "CODE_BLUE_START" for e in events):
    #     if events[0]["type"] != "CODE_BLUE_START":
    #         print("Validation failed: CODE_BLUE_START is present but not first event")
    #         return False
    
    # # Check no consecutive duplicate events
    # for i in range(len(events) - 1):
    #     if events[i]["type"] == events[i + 1]["type"]:
    #         print(f"Validation failed: Consecutive duplicate events of type {events[i]['type']}")
    #         return False
    
    # Check compression stops are followed by pulse checks
    # for i in range(len(events) - 1):
    #     if (events[i]["type"] == "COMPRESSION" and 
    #         events[i]["details"] and 
    #         events[i]["details"].get("end_time") is not None):
    #         if events[i + 1]["type"] != "PULSE_CHECK":
    #             print("Validation failed: Compression stop not followed by pulse check")
    #             return False
    
    # Check no events after ROSC
    for i, event in enumerate(events):
        if event["type"] == "ROSC" and i < len(events) - 1:
            print("Validation failed: Events found after ROSC")
            return False
            
    return True

# === Routes ===

# PostgreSQL connection details
DB_URL = os.environ['DATABASE_URL']
PG_HOST = os.environ['PGHOST']
PG_DATABASE = os.environ['PGDATABASE']
PG_USER = os.environ['PGUSER']
PG_PASSWORD = os.environ['PGPASSWORD']

def init_db():
    """Initialize the database with required tables."""
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    # Create timeline table if it doesn't exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS timelines (
            id UUID PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            events JSONB
        );
    """)
    
    conn.commit()
    cur.close()
    conn.close()

def save_timeline_to_store(timeline_id: str, timeline: List[Dict]):
    """Save timeline to memory store."""
    timeline_data = {
        "id": timeline_id,
        "created_at": time.time(),
        "events": timeline
    }
    timelines.append(timeline_data)

def save_timeline_to_db(timeline_id: str, timeline: List[Dict]) -> str:
    """Save timeline to PostgreSQL and return a unique ID."""
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        
        # Insert timeline into database
        cur.execute(
            "INSERT INTO timelines (id, events) VALUES (%s, %s)",
            (timeline_id, Json(timeline))
        )
        
        conn.commit()
        cur.close()
        conn.close()
        
        return timeline_id
    except Exception as e:
        print(f"Database error: {e}")
        return None

def generate_clinical_feedback(transcript: str, timeline: List[Dict]) -> Dict:
    """Generate clinical feedback on the code blue response."""
    
    prompt = f"""
You are an experienced member of a hospital's Code Blue Committee, reviewing a code blue response.
Please analyze the following code blue transcript and its structured timeline to provide clinical feedback.

Transcript:
{transcript}

Structured Timeline:
{json.dumps(timeline, indent=2)}

Provide a comprehensive analysis in the following format:

1. Event Summary:
   - Patient demographics and relevant history
   - Outcome (ROSC achieved or not)
   - Key Characteristics of the code blue response, including:
        - Total duration of the code blue response
        - Total number of medications administered
        - Rounds of compressions performed
        - Airway established (yes/no)
        
2. Performance Score:
   - Score out of 27 based on AHA's ACLS Performance Grading Rubric
   - Convert to percentage

3. Strengths:
   List 2 key strengths, which consists of a detailed observation supporting this strength.

4. To Improve:
   List 2-4 areas for improvement, which consists of a detailed observation supporting this improvement area.

Strength and improvements should only focus on events that occur between the time the code blue is called and the patient is ROSC. They should focus on the following:
- Timing of interventions
- Communication
- Team coordination
- Use of ACLS guidelines
- Specific examples from the transcript/timeline
take into account times to apply/prepare equipment or meds when providing feedback.



Format your response as a JSON object with these sections.
The response should be similar to a code blue committee report, focusing on clinical accuracy and actionable feedback.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={ "type": "json_object" }
        )
        
        feedback = json.loads(response.choices[0].message.content)
        return feedback
    except Exception as e:
        print(f"Feedback generation error: {e}")
        return None

# Initialize database when app starts
init_db()

@app.route("/transcript", methods=["POST"])
def receive_transcript():
    data = request.json
    events = data.get("events", [])

    if not events:
        return jsonify({"error": "Missing events"}), 400

    # Build the transcript text for feedback
    transcript = "\n".join([
        f"{event['timestamp']} - {event['speaker']}: {event['text']}"
        for event in events
    ])

    timeline = call_llm_for_sequence(events)
    if timeline is None:
        return jsonify({"error": "Failed to process events"}), 400

    # Generate timeline ID
    timeline_id = str(uuid.uuid4())
    
    # Save to memory store right after getting the timeline
    save_timeline_to_store(timeline_id, timeline)
    
    # Save to DB
    save_timeline_to_db(timeline_id, timeline)
    
    # Generate clinical feedback
    feedback = generate_clinical_feedback(transcript, timeline)
    
    return jsonify({
        "timeline_id": timeline_id,
        "events": timeline,
        "feedback": feedback
    })

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "Server is alive."})

if __name__ == "__main__":
    app.run(debug=True)
