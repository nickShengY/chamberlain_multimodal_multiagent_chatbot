import speech_recognition as sr
import streamlit as st

from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import io
import openai
from agents import question_judge

from config import *
import base64

import csv
import json
from langchain.retrievers.merger_retriever import MergerRetriever


from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma
from langchain.docstore import InMemoryDocstore

from typing import Optional


def text_to_speech(text, filename="speech.mp3"):
    """
    Convert text to speech and save the audio to a file.

    Args:
    text (str): The text to be converted to speech.
    filename (str, optional): The name of the file where the audio will be saved. Defaults to "speech.mp3".

    Returns:
    str: The filename where the audio is saved.
    """

    # Create an audio speech response from the text using the specified TTS model and voice
    response = t2sp.audio.speech.create(
        model="tts-1",    # Specifies the TTS model to use
        voice="onyx",     # Specifies the voice to use for the TTS
        input=text        # The input text to convert to speech
    )

    # Stream the audio response to a file with the given filename
    response.stream_to_file(filename)

    # Return the filename where the audio is saved
    return filename


# Function to recognize speech
def recognize_speech():
    # Initialize recognizer
    r = sr.Recognizer()

    # Use the microphone as source for input
    with sr.Microphone() as source:
        st.write("Listening...")
        # Listen for the first phrase and extract it into audio data
        audio_data = r.listen(source)
        st.write("Recognizing...")
        try:
            # Recognize speech using Google Web Speech API
            text = r.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "I don't really know what are you talking about"
        except sr.RequestError as e:
            return f"Could not request results from Speech Recognition service; {e}"
        
def predict_question(question, model, tokenizer, label_map, device = DEVICE):
    # Tokenize the question
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Model inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    predicted_class_idx = outputs.logits.argmax(dim=1).item()

    predicted_class = [label for label, idx in label_map.items() if idx == predicted_class_idx][0]
    return predicted_class

       
def answer_judgement(response):
    predicted_label = predict_question(response, VMODEL, TOKENIZER, YNLABLES)
    print(f"Predicted Label: {predicted_label}")
        
    return predicted_label

def gpt_verify(label, prompt): 
    # Constructing the query to the model
    field = ''
    if label == 'Eat' or label == 'Grocery': ## Issue: unstable with grocery
        
        query = f"If this person said going out to eat, simply give me reply of 'Fun', If this person said about grocery or he just bought grocery, simply reply 'Grocery', If this person said want to eat some thing or want food suggestions or he is hungry simply reply 'Eat'. no explainations needed in anywhere."
        try:
            completion = question_judge.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": query},
                    {"role": "user", "content": prompt}
                ]
                )

            # Extracting the text response
            answer = completion.choices[0].message.content
            print("change the ", label, " to ", answer)
            # print("gpt answer is",answer)

        except Exception as e:
            answer = f"An error occurred: {str(e)}"
        if not (answer == 'Fun' or answer == 'Eat' or answer == 'Grocery'):
            answer = 'Other'
            print("changed to Other")
        return answer
    else:
        if label == 'Dress':
            field = 'asking for perfessonal stylist suggestions, and this person wants to know if this dressing combination looks good'
        elif label == 'Bill':
            field = 'wants to know if there is any bills need to be paid, and amount of the payment of the bills'        
        elif label == 'Finance':
            field = 'wants to know about their own financial situations and investment strategies'
        elif label == 'Planner':
            field = 'wants to schedule their current daily plans'
        elif label == 'Laundry':
            field = 'wants to do laundry or wondering if he needs to do laundry'
        elif label == 'Fun':
            field = 'wants to go out for fun, like go to the bar, go to a sport event, go to the movie, go to a concert, go clubbing, go to some other entertainment events or going to places of interests. This does not include anything that related to fact checks, home entertainment or somewhere that needs to take a trip'
        elif label == 'IoT':
            field = "wants to use the iot devices in home, television, netflix, computer, speakers, kareoke, or other iot devices"
        elif label == 'Shopping':
            field = "wants to do online shopping or has already finished online shopping"
        elif label == 'Flight':
            field = "wants to go somewhere that needs to take a flight to arrived, or the person said he wants to take a flight or want a flight ticket"
        elif label == 'Coding':
            field = "wants to develop a program or game"
        elif label == 'Task':
            field = "has a hard task and need some help to make plan of it"
        else:
            field = "is trying to have a conversation or chat, or wants some information that has a ground fact or just simply ask a general question"
        query = f"You are now a human question verifier ( you should answer yes or no), \
        and you need to judge if it means this person {field} or related questions or commands? You need to tend to answer yes but if it is really off the topic, you should negate this."
        print('got verified', label, query)
        try:
            completion = question_judge.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": query},
                    {"role": "user", "content": prompt}
                ]
                )

            # Extracting the text response
            answer = completion.choices[0].message.content
            # print("gpt answer is",answer)

        except Exception as e:
            answer = f"An error occurred: {str(e)}"
    print("gpt answer is",answer)
    return answer



def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def unpack_dicts_to_string(lst_of_dict):
    # Initialize an empty list to store strings
    string_list = []

    # Loop through each dictionary in the list
    for item in lst_of_dict:
        # Unpack each dictionary and format it into a string
        dict_string = ', '.join([f"'{key}': '{value}'" for key, value in item.items()])
        # Append the formatted string to the list
        string_list.append(dict_string)

    # Join all the formatted strings with a separator (e.g., newline)
    result_string = '\n'.join(string_list)
    
    return result_string


