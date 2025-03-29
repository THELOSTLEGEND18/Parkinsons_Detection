import gradio as gr
from PIL import Image
import os

def get_image_path(filename):
    possible_paths = ["../../assets/", "../assets/", "./assets/"]
    for path in possible_paths:
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            return full_path
    return None

def resize_image(image_path, new_width):
    if image_path:
        img = Image.open(image_path)
        width, height = img.size
        new_height = int((new_width / width) * height)
        img = img.resize((new_width, new_height))
        return img
    return None

logo_path = get_image_path("logo.png")
brain_path = get_image_path("brain.png")
typing_path = get_image_path("typing.png")
voice_path = get_image_path("voice.png")
chatbot_path = get_image_path("chatbot.png")

def create_home_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as home_interface:
        with gr.Row():
            with gr.Column(scale=1):
                if logo_path:
                    gr.Image(resize_image(logo_path, 300), show_label=False, container=False)
                gr.Markdown("### **ParkiSense – An AI-powered tool for early Parkinson’s detection using keystroke dynamics, voice analysis, and a Gemini chatbot for Parkinson’s-related queries.**")
            with gr.Column(scale=1):
                if brain_path:
                    gr.Image(resize_image(brain_path, 300), show_label=False, container=False)
                gr.Markdown("### **🧠 Parkinson's affects movement-controlling brain areas.**")
                gr.Markdown("""
                ## About Parkinson's Disease
                
                Parkinson's disease is a progressive neurological disorder that affects movement. 
                **Early symptoms include:**
                - Tremor in hands
                - Slowed movement (bradykinesia)
                - Muscle stiffness
                - Changes in speech and handwriting
                """)
                

        gr.Markdown("---")

        gr.Markdown("## 🛠️ Features")

        with gr.Row():
            with gr.Column():
                if typing_path:
                    gr.Image(typing_path, show_label=False, container=False)
                gr.Markdown("""
                ### ⌨️ Keystroke Analysis  
                - Analyzes typing speed and patterns  
                - Detects subtle motor control changes  
                - Uses AI-based models for analysis  
                """)

            with gr.Column():
                if voice_path:
                    gr.Image(voice_path, show_label=False, container=False)
                gr.Markdown("""
                ### 🎤 Voice Analysis  
                - Records and analyzes speech stability  
                - Detects tremors and breathiness  
                - AI evaluates potential risks  
                """)

            with gr.Column():
                if chatbot_path:
                    gr.Image(chatbot_path, show_label=False, container=False)
                gr.Markdown("""
                ### 🤖 AI Chatbot  
                - Answers Parkinson’s-related queries  
                - Explains test results in simple terms  
                - Provides health insights and lifestyle tips  
                """)

        gr.Markdown("---")

        gr.Markdown("© 2024 Parkinson's Disease Detection Tool - AOML FINAL PROJECT")

    return home_interface

home = create_home_interface()

if __name__ == "__main__":
    home.launch()
