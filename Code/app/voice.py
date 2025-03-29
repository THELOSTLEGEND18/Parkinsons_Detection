import gradio as gr
import sounddevice as sd
import wavio
import numpy as np
import librosa
import librosa.feature
import pandas as pd
import parselmouth
import pickle
from scipy.signal import find_peaks
import tempfile

MODEL_PATH = "../../model/voicemodel.pkl"

with open(MODEL_PATH, "rb") as file:
    voice_model = pickle.load(file)

expected_features = voice_model.feature_names_in_

# Variable to hold the forwarding function
_forward_callback = None

def forward_voice_results(callback_function):
    """Register a callback to forward voice test results"""
    global _forward_callback
    _forward_callback = callback_function

def record_audio(duration=3, samplerate=8000):
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.float32)
    sd.wait()
    wavio.write(temp_file, recording, samplerate, sampwidth=2)
    return temp_file

def extract_features(file):
    try:
        y, sr = librosa.load(file, sr=8000)
        snd = parselmouth.Sound(file)
        pitch = snd.to_pitch()
        fo = pitch.selected_array['frequency']
        fo = fo[fo > 0]

        if len(fo) == 0:
            return None, "⚠️ No voiced segments detected in the audio!"

        fhi = np.max(fo)
        flo = np.min(fo)
        nhr = np.mean(librosa.feature.spectral_flatness(y=y))
        intensity = snd.to_intensity()
        hnr = np.mean(intensity.values)
        rpde = np.var(fo) / (np.abs(np.mean(fo)) + 1e-6)
        dfa = np.mean(librosa.feature.rms(y=y)) / (np.std(y) + 1e-6)
        spread1 = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)) * -0.002
        spread2 = np.std(librosa.feature.spectral_bandwidth(y=y, sr=sr)) * 0.05
        d2 = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)) / 50
        peaks, _ = find_peaks(y, height=0)
        ppe = len(peaks) / len(y)

        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 600)

        def safe_praat_call(obj, command, *args):
            try:
                return parselmouth.praat.call(obj, command, *args)
            except:
                return np.nan

        jitter_local = safe_praat_call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_rap = safe_praat_call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_ppq = safe_praat_call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_ddp = 3 * jitter_rap if not np.isnan(jitter_rap) else np.nan

        shimmer_local = safe_praat_call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_db = safe_praat_call([snd, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq3 = safe_praat_call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq5 = safe_praat_call([snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq11 = safe_praat_call([snd, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_dda = 3 * shimmer_apq3 if not np.isnan(shimmer_apq3) else np.nan

        features = {
            "MDVP:Fo(Hz)": np.mean(fo),
            "MDVP:Fhi(Hz)": fhi,
            "MDVP:Flo(Hz)": flo,
            "MDVP:Jitter(%)": jitter_local,
            "MDVP:RAP": jitter_rap,
            "MDVP:PPQ": jitter_ppq,
            "Jitter:DDP": jitter_ddp,
            "MDVP:Shimmer": shimmer_local,
            "MDVP:Shimmer(dB)": shimmer_db,
            "Shimmer:APQ3": shimmer_apq3,
            "Shimmer:APQ5": shimmer_apq5,
            "MDVP:APQ": shimmer_apq11,
            "Shimmer:DDA": shimmer_dda,
            "NHR": nhr,
            "HNR": hnr,
            "RPDE": rpde,
            "DFA": dfa,
            "spread1": spread1,
            "spread2": spread2,
            "D2": d2,
            "PPE": ppe
        }

        features_df = pd.DataFrame([features])
        features_df = features_df.reindex(columns=expected_features, fill_value=0)

        return features_df, None

    except Exception as e:
        return None, f"⚠️ Error extracting features: {e}"

def analyze_voice():
    file = record_audio()
    features_df, error_message = extract_features(file)

    if error_message:
        return error_message, None, None, None

    prediction_prob = voice_model.predict_proba(features_df)[0][1]
    threshold = 0.6
    prediction = 1 if prediction_prob >= threshold else 0
    result = f"🔴 Parkinson's Detected" if prediction == 1 else "🟢 No Parkinson's"

    features_df.to_csv("user_voice_features.csv", index=False)

    # Forward results to chatbot
    if _forward_callback:
        _forward_callback(
            test_type="Voice Analysis",
            prediction=result,
            details=f"Probability: {prediction_prob:.2f}",
            features=features_df.to_dict()
        )

    return result, f"Probability: {prediction_prob:.2f}", features_df, "✅ Features saved!"

voice = gr.Interface(
    fn=analyze_voice,
    inputs=[],
    outputs=[
        gr.Textbox(label="Prediction"), 
        gr.Textbox(label="Probability"), 
        gr.Dataframe(label="Extracted Features"),
        gr.Textbox(label="Status")
    ],
    title="Voice-Based Parkinson's Detection",
    description="Click the button to record your voice for 3 seconds. The model will predict Parkinson's and display extracted features."
)