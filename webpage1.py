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

# Set page title and icon
st.set_page_config(page_title='Dhanshakti by SPQR', page_icon=':sparkles:')

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

def process_pdf_question(question, docs):
    # Process the question based on the PDF content
    # For example, you can search for keywords in the PDF
    # and return relevant information
    for doc in docs:
        if question.lower() in doc.text.lower():
            return doc.text[:500]  # Return the first 500 characters of the matched text
    return "I'm sorry, I couldn't find an answer in the PDF for your question."

    
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

            # Translate the recognized text to English
            translator = Translator()
            translated_text = translator.translate(text, dest='en').text

            # Set up Hugging Face API token
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

            # Load PDF document
            file =r"C:\Users\DELL\Desktop\finance.pdf"
            loader = PyPDFLoader(file)
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
            response = qa.invoke({"question": translated_text, "chat_history": []})

            # Translate the response back to Hindi
            translated_response = translator.translate(response['answer'], dest='hi').text

            # Initialize pyttsx3 engine and speak the response
            engine = pyttsx3.init()
            engine.say(translated_response)
            engine.runAndWait()

            return text, translated_response  # Return the recognized text and the translated response

        except sr.UnknownValueError:
            st.write("Google Speech Recognition could not understand audio.")
        except sr.WaitTimeoutError:
            st.write("Timeout: No speech detected after 5 seconds.")
        except sr.RequestError as e:
            st.write(f"Could not request results from Google Speech Recognition service; {e}")
    


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
            
            # Display the translated text
            st.markdown("<hr style='margin-top: 20px; margin-bottom: 20px; border: 0; border-top: 1px solid #ffffff;'>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: left; color: #ffffff;'>Translated Text:</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 16px; color: #ffffff;'>{translated_text}</p>", unsafe_allow_html=True)
            
            # Play audio if the translated text is available
            if translated_text:
                translate_and_speak(translated_text, 'hi')
            else:
                st.error("I'm sorry, I didn't understand your question.")

elif input_mode == 'Voice Input':
    if st.button('Start Recording'):
        recognized_text, translated_response = speech_recognition_and_api()
        st.write('Recognized Text:', recognized_text)

    # Speak the translated response
        if translated_response:
            translate_and_speak(translated_response, 'hi')
        else:
            st.error("I'm sorry, I couldn't understand your question.")

