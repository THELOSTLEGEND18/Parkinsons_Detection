import os
import gradio as gr
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
api_key = os.getenv("GOOGLE_API_KEY")

# Ensure the API key is found
if not api_key:
    raise ValueError("⚠️ API key not found. Please check your .env file.")

# Configure the Google Gemini API
genai.configure(api_key=api_key)

# Chatbot prompt
chatbot_prompt = """You are a medical assistant specializing in Parkinson's detection.
Your goal is to help users understand Parkinson's symptoms, keystroke/voice test results, 
and provide lifestyle tips. You do NOT diagnose but suggest consulting a doctor when needed.

When explaining test results:
- Keystroke analysis measures typing patterns which can reveal motor impairments common in Parkinson's
- Voice analysis examines speech characteristics like tremor, breathiness, and pronunciation issues
- Explain features in simple terms that a non-medical person can understand
- Always be compassionate and clear about what the results might suggest
- Use emojis to make the conversation friendly and engaging
- Avoid using medical jargon unless necessary
- If the user asks about their test results, provide a summary of the findings
- If the user asks about Parkinson's disease, provide general information about symptoms and management"""

# Initialize Gemini chatbot
model = genai.GenerativeModel("gemini-2.0-flash")

# Chat history storage
chat_history = []
forwarded_results = []

def forward_results_to_chatbot(test_type, prediction, details, features):
    """Stores test results for chatbot explanation."""
    global forwarded_results
    forwarded_results.append({
        "test_type": test_type,
        "prediction": prediction,
        "details": details,
        "features": features
    })

def chat_with_bot(user_input):
    """Handles chatbot conversation and integrates test results if available."""
    global chat_history, forwarded_results
    
    try:
        if "test" in user_input.lower() or "result" in user_input.lower():
            if forwarded_results:
                result_data = []
                for result in forwarded_results:
                    result_data.append(f"Test Type: {result['test_type']}")
                    result_data.append(f"Prediction: {result['prediction']}")
                    result_data.append(f"Details: {result['details']}")
                    if result['features']:
                        feature_text = "Features measured:\n"
                        for feature, value in result['features'].items():
                            feature_text += f"- {feature}: {value}\n"
                        result_data.append(feature_text)
                
                results_summary = "\n".join(result_data)
                
                context = [
                    chatbot_prompt,
                    "Please explain these test results in medical context:",
                    results_summary,
                    "Explanation should be in simple terms for the patient."
                ]
                
                response = model.generate_content(context)
                bot_response = response.text
                
                chat_history.append({"role": "user", "content": user_input})
                chat_history.append({"role": "assistant", "content": bot_response})
                forwarded_results = []
                return chat_history
            else:
                msg = "No test results available. Please complete a voice or typing test first."
                chat_history.append({"role": "user", "content": user_input})
                chat_history.append({"role": "assistant", "content": msg})
                return chat_history

        conversation = [chatbot_prompt]
        for message in chat_history:
            conversation.append(message["content"])
        conversation.append(user_input)
        
        response = model.generate_content(conversation)
        bot_response = response.text
        
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": bot_response})
        
        return chat_history
    
    except Exception as e:
        error_msg = f"⚠️ Error: {e}"
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": error_msg})
        return chat_history

def clear_chat():
    """Clears the chat history."""
    global chat_history, forwarded_results
    chat_history = []
    forwarded_results = []
    return []

def create_chatbot_interface():
    """Creates Gradio chatbot UI."""
    with gr.Blocks() as chatbot_interface:
        with gr.Column():
            gr.Markdown("## 🩺 Parkinson's Medical Chatbot")
            gr.Markdown("Ask me about Parkinson's disease, test results, or lifestyle tips!")
            
            chatbox = gr.Chatbot(height=500, type="messages")
            
            with gr.Row():
                user_input = gr.Textbox(placeholder="Type your question here...", show_label=False, scale=9)
                send_button = gr.Button("Send", scale=1)
            
            clear_button = gr.Button("Clear Chat")
            
            send_button.click(chat_with_bot, inputs=user_input, outputs=chatbox)
            user_input.submit(chat_with_bot, inputs=user_input, outputs=chatbox)
            clear_button.click(clear_chat, outputs=chatbox)
            
    return chatbot_interface

# Create the chatbot interface
chatbot_ui = create_chatbot_interface()

# If running as a standalone script
if __name__ == "__main__":
    chatbot_ui.launch()
