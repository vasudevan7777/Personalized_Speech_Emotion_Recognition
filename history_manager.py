"""
Prediction History Manager.
Handles persistence, retrieval, and analysis of past prediction records.
Stores data in feedback_data/prediction_history.json.
"""

import os
import json
import datetime
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_DIR = os.path.join(BASE_DIR, "feedback_data")
HISTORY_FILE = os.path.join(FEEDBACK_DIR, "prediction_history.json")

def save_prediction(emotion, confidence, risk_score, wellness_status, recommendations=None):
    """
    Save a prediction record to the history file.
    """
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    
    # Create prediction record
    record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "emotion": emotion,
        "confidence": float(confidence),
        "risk_score": float(risk_score),
        "wellness_status": wellness_status,
        "recommendations": recommendations or []
    }
    
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except Exception:
            # Handle corrupted files
            history = []
            
    history.append(record)
    
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
        return True
    except Exception:
        return False

def load_history():
    """
    Load prediction history as a list of dictionaries.
    """
    if not os.path.exists(HISTORY_FILE):
        return []
        
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def load_history_as_df():
    """
    Load prediction history as a Pandas DataFrame for visualization.
    """
    history = load_history()
    if not history:
        return pd.DataFrame(columns=["timestamp", "emotion", "confidence", "risk_score", "wellness_status"])
        
    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def clear_history():
    """
    Clear all saved predictions.
    """
    if os.path.exists(HISTORY_FILE):
        try:
            os.remove(HISTORY_FILE)
            return True
        except Exception:
            return False
    return True
