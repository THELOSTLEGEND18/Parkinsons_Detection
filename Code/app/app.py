import gradio as gr
from home import home
from type import type, forward_typing_results
from voice import voice, forward_voice_results
from chatbot import chatbot_ui, forward_results_to_chatbot

# Connect test modules to chatbot for result forwarding
forward_typing_results(forward_results_to_chatbot)
forward_voice_results(forward_results_to_chatbot)

# Create a tabbed interface with all components
app = gr.TabbedInterface(
    [home, type, voice, chatbot_ui],
    ["Home", "Keystroke Analysis", "Voice Analysis", "Medical Chatbot"],
    title="Parkinson's Disease Detection & Information Tool",
    theme=gr.themes.Soft()
)

# Launch the application with PWA support
if __name__ == "__main__":
    app.launch(pwa=True)
