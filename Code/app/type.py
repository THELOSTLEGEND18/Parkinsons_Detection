import gradio as gr
from pynput import keyboard
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import pickle
import time

MODEL_PATH = "../../model/sgd_model.pkl"

def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

model = load_model()
expected_features = model.feature_names_in_
keystroke_times = []
typed_text = ""

# Variable to hold the forwarding function
_forward_callback = None

def forward_typing_results(callback_function):
    """Register a callback to forward typing test results"""
    global _forward_callback
    _forward_callback = callback_function

def on_press(key):
    global typed_text
    try:
        if hasattr(key, 'char') and key.char is not None:
            typed_text += key.char  # Append character
        elif key == keyboard.Key.space:
            typed_text += " "
        elif key == keyboard.Key.backspace and typed_text:
            typed_text = typed_text[:-1]  # Remove last character
    except:
        pass
    keystroke_times.append(time.time())

listener = keyboard.Listener(on_press=on_press)
listener.start()

def analyze_keystrokes():
    global keystroke_times, typed_text
    if len(keystroke_times) <= 5:
        return "⚠️ Not enough keystrokes! Please type more.", typed_text, pd.DataFrame()

    intervals = np.diff(keystroke_times)
    features = {
        "mean_L": np.mean(intervals),
        "std_L": np.std(intervals),
        "skew_L": skew(intervals),
        "skew_R": skew(intervals),
        "kurtosis_L": kurtosis(intervals),
        "mean_hold_diff": np.mean(np.abs(intervals - np.mean(intervals))),
        "mean_LL": np.mean(intervals[:len(intervals)//2]),  
        "mean_RL": np.mean(intervals[len(intervals)//2:]),  
        "std_LL": np.std(intervals[:len(intervals)//2]),
        "std_LR": np.std(intervals),  
        "skew_LL": skew(intervals[:len(intervals)//2]),
        "skew_LR": skew(intervals[len(intervals)//2:]),
        "skew_RL": skew(intervals),
        "skew_RR": skew(intervals),
        "kurtosis_LL": kurtosis(intervals[:len(intervals)//2]),
        "kurtosis_LR": kurtosis(intervals),
        "kurtosis_RL": kurtosis(intervals),
        "kurtosis_RR": kurtosis(intervals),
        "mean_LL_RR_diff": np.mean(intervals[:len(intervals)//2]) - np.mean(intervals[len(intervals)//2:])
    }

    features_df = pd.DataFrame([features])
    features_df = features_df.reindex(columns=expected_features, fill_value=0)

    prediction = model.predict(features_df)[0]
    result = "🔴 Parkinson's Detected" if prediction == 1 else "🟢 No Parkinson's"

    # Forward results to registered callback
    if _forward_callback:
        _forward_callback(
            test_type="Keystroke Analysis",
            prediction=result,
            details="Keystroke Analysis Results",
            features=features_df.to_dict()
        )

    return result, typed_text, features_df

type= gr.Interface(
    fn=analyze_keystrokes,
    inputs=[],
    outputs=[
        gr.Textbox(label="Prediction"), 
        gr.Textbox(label="Typed Text"), 
        gr.Dataframe(label="Extracted Features")
    ],
    live=True,
    title="Keystroke-Based Parkinson's Detection",
    description="Type naturally. Click 'Submit' when ready. The model will predict Parkinson's and display extracted features.")