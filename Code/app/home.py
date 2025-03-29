import gradio as gr
from PIL import Image
import os

# Function to locate images
def get_image_path(filename):
    possible_paths = ["../../assets/", "../assets/", "./assets/"]
    for path in possible_paths:
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            return full_path
    return None

# Load images if available
logo_path = get_image_path("logo.png")
brain_path = get_image_path("brain.png")
typing_path = get_image_path("typing.png")
voice_path = get_image_path("voice.png")

# Apply a Modern Theme
custom_theme = gr.themes.Base(
    primary_hue="blue",  
    secondary_hue="gray",
    neutral_hue="gray"  # Change "cool" to "gray"
)

def create_home_interface():
    with gr.Blocks(theme=custom_theme) as home_interface:
        # Header
        with gr.Row():
            if logo_path:
                gr.Image(logo_path, show_label=False, height=120)
            with gr.Column():
                gr.Markdown("## 🧠 Parkinson's Disease Detection Tool")
                gr.Markdown("### AI-powered insights for early detection")

        gr.Markdown("---")

        # About Section
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("""
                ## About Parkinson's Disease
                
                Parkinson's disease is a progressive neurological disorder that affects movement. 
                **Early symptoms include:**
                - Tremor in hands
                - Slowed movement (bradykinesia)
                - Muscle stiffness
                - Changes in speech and handwriting
                """)
            with gr.Column(scale=1):
                if brain_path:
                    gr.Image(brain_path, label="Brain with Parkinson's", height=200)
                else:
                    gr.Markdown("🧠 Parkinson's affects movement-controlling brain areas.")

        gr.Markdown("---")

        # Features Section
        gr.Markdown("## 🛠️ Features")

        with gr.Row():
            with gr.Column():
                if typing_path:
                    gr.Image(typing_path, height=120)
                gr.Markdown("""
                ### ⌨️ Keystroke Analysis  
                - Analyzes typing speed and patterns  
                - Detects subtle motor control changes  
                - Uses AI-based models for analysis  
                **[Try Keystroke Analysis]**(Keystroke Page)
                """)

            with gr.Column():
                if voice_path:
                    gr.Image(voice_path, height=120)
                gr.Markdown("""
                ### 🎤 Voice Analysis  
                - Records and analyzes speech stability  
                - Detects tremors and breathiness  
                - AI evaluates potential risks  
                **[Try Voice Analysis]**(Voice Page)
                """)

            with gr.Column():
                gr.Markdown("""
                ### 🤖 AI Chatbot  
                - Answers Parkinson’s-related queries  
                - Explains test results in simple terms  
                - Provides health insights and lifestyle tips  
                **[Ask the Chatbot]**(Chatbot Page)
                """)

        gr.Markdown("---")

        # Disclaimer
        gr.Markdown("""
        ### ⚠️ Disclaimer  
        ```
        This application is **not a substitute for medical advice**. 
        AI predictions serve as screening tools, not diagnoses.  
        Consult a medical professional for any concerns.
        ```
        """)

        gr.Markdown("---")

        # Quick Start Guide
        gr.Markdown("""
        ## 🚀 Getting Started
        1. **Try Keystroke Analysis** – Type naturally to analyze typing patterns.  
        2. **Take Voice Analysis** – Record a short voice sample.  
        3. **Ask the AI Chatbot** – Get insights about Parkinson’s symptoms and reports.  
        """)

        gr.Markdown("---")
        gr.Markdown("© 2024 Parkinson's Disease Detection Tool - AOML FINAL PROJECT")

    return home_interface

home = create_home_interface()

if __name__ == "__main__":
    home.launch()
