import streamlit as st
from googletrans import Translator
from gtts import gTTS
import os
from PIL import Image

import os
import pyttsx3
import speech_recognition as sr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain

from streamlit.server.server import Server
import threading
import time

# Define Streamlit app
def main():
    st.write("Welcome to my Streamlit app!")

# Custom CORS middleware
def cors_middleware(self, handler):
    def wrapped_app(environ, start_response):
        # Add CORS headers to the response
        def my_start_response(status, response_headers, exc_info=None):
            response_headers.append(('Access-Control-Allow-Origin', '*'))
            response_headers.append(('Access-Control-Allow-Methods', 'GET, POST, OPTIONS'))
            response_headers.append(('Access-Control-Allow-Headers', 'Origin, Content-Type'))
            return start_response(status, response_headers, exc_info)
        return handler(environ, my_start_response)
    return wrapped_app

# Replace Streamlit server's middleware with custom CORS middleware
def run():
    global main
    server = Server.get_current()._server
    server.app.wsgi_app = cors_middleware(server.app.wsgi_app)
    main()

# Start Streamlit app in a separate thread
if _name_ == '_main_':
    threading.Thread(target=run).start()

# Function to translate and speak
def translate_and_speak(input_text, target_language):
    translator = Translator()
    translated = translator.translate(input_text, dest=target_language)
    
    # Translate and speak the translated text
    tts = gTTS(translated.text, lang=target_language)
    tts.save("translated.mp3")
    os.system("start translated.mp3")

    # Display the translated text
    st.write('Translated Text:')
    st.write(translated.text)

# Function for the chatbot
def chatbot(question):
    if 'hello' in question.lower():
        return 'Hello! How can I assist you today?'
    elif 'weather' in question.lower():
        return 'The weather is sunny and warm.'
    else:
        return "I'm sorry, I didn't understand your question."

    
# Function to recognize speech and process the API
def speech_recognition_and_api():
    # Initialize the speech recognizer
    recognizer = sr.Recognizer()

    # Adjust for ambient noise
    with sr.Microphone() as source:
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source)

        try:
            # Listen for speech input
            audio_data = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio_data, language="en-US")
            st.write("You said:", text)

            # Set up Hugging Face API token
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

            # Load PDF document
            url = "https://drive.google.com/file/d/1u3l219nhEDD9rRqoh_L4uiQUcln5ciYe/view?usp=drive_link"
            file =r"C:\Users\DELL\Desktop\finance.pdf"

            loader = PyPDFLoader(file)
            #loader = PyPDFLoader(url)
            docs = loader.load()

            # Split documents and create embeddings
            splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
            split_docs = splitter.split_documents(docs)
            embeddings = HuggingFaceEmbeddings()
            db = Chroma.from_documents(split_docs, embeddings)

            # Initialize Hugging Face Hub
            llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.1, "max_length": 512})

            # Create conversational retrieval chain
            qa = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=db.as_retriever(search_kwargs={"k": 1})
            )

            # Invoke the conversational retrieval chain
            response = qa.invoke({"question": text, "chat_history": []})

            # Initialize pyttsx3 engine and speak the response
            engine = pyttsx3.init()
            engine.say(response['answer'])
            engine.runAndWait()

            return text, response['answer']  # Return the recognized text and the AI's answer

        except sr.UnknownValueError:
            st.write("Google Speech Recognition could not understand audio.")
        except sr.WaitTimeoutError:
            st.write("Timeout: No speech detected after 5 seconds.")
        except sr.RequestError as e:
            st.write(f"Could not request results from Google Speech Recognition service; {e}")
    
    
    
# Page layout
st.title('DhanShakti by SPQR')
st.subheader('Your Personal Assistant')

# Logo and Title
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.write("")
with col2:
    image = Image.open('logo.jfif')
    st.image(image, use_column_width=True)
with col3:
    st.write("")

st.markdown("<h1 style='text-align: center; color: #ffffff;'>Dhanshakti</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #ffffff;'>Your Personal Assistant</h2>", unsafe_allow_html=True)


# Text input for asking questions
input_mode = st.radio('Select Input Mode:', ['Text Input', 'Voice Input'])

if input_mode == 'Text Input':
    question = st.text_input('Ask me anything...')
    if st.button('Submit'):
        if not question.strip():
            st.error('Please enter a question.')
        else:
            # Translate the input text to Hindi
            translated_text = chatbot(question)
            
            # Display
