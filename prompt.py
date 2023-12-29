from utils import *
import torch
from config import *
from agents import *



# Define the function for classifying the task based on the user's input
def task_classification(prompt, n):
    # Load the tokenizer and model
    predicted_label = predict_question(prompt, QMODEL, TOKENIZER, LABELS, DEVICE)
    prompt, predicted_label = label_verfication(predicted_label, prompt)
    return predicted_label


# Define the function to verify and potentially adjust the predicted task label    
def label_verfication(label, prompt):
    answer = gpt_verify(label, prompt) # verify with gpt2
    if answer in ['Fun', 'Eat', 'Grocery', 'Other']:
        return prompt, answer
    else:
        # Further judgment for labels not directly verified
        judgement = answer_judgement(answer)
        # Confirming the label if judgment is positive and not categorized as 'Other'
        if judgement == 'Yes' and label != "Other":
            return prompt, label
        else:
            # Default to 'Other' if judgment is negative
            return prompt, "Other"

# Define the function to generate a response based on the recognized speech
def response_generator(recognize_speech, hist = ""):
    label = task_classification(recognize_speech, 1)
    if label == 'Eat':
        reply = "Oh master, you want some food? I can give you assist on that. Do you want your planned meal to be cooked or you want something different? \
    I will find something in the fridge for your taste requests, tell me what kind of food you want I will cook for you."
        print(reply)
        return reply, label
    
    elif label == 'Dress':
        reply = "Oh master, Let me see how you look right now, and I will try my best to help you" 
        return reply, label
    elif label == 'Bill':
        reply = "Oh master, Let me see if you have bills to take care of"
        print(reply)
        return reply, label
    elif label == 'Finance':
        reply = "Oh master, if you have portfolio to ask about, please put it here. If you just want to see how your situation is at this point, you can also ask me."
        print(reply)
        return reply, label
    elif label == 'Planner':
        reply = "Oh master, Let me know if you want to change your plans for today or for the future."
        print(reply)
        raise NotImplementedError("This function is not yet implemented.")
        return reply, label
    elif label == 'Grocery':
        reply = "Oh master, what did you buy from the grocery?"
        print(reply)
        return reply, label
    elif label == 'Laundry':
        reply = "Oh master, let me check if you need to do laundry"
        print(reply)
        raise NotImplementedError("This function is not yet implemented.")
        return reply, label
    elif label == 'Fun':
        reply = "Oh master, Let me check if there is anything I can find from the nearby areas"
        print(reply)
        return reply, label
    elif label == 'IoT':
        reply = "Oh master, Let me help you."
        print(reply)
        raise NotImplementedError("This function is not yet implemented.")
        return reply, label
    elif label == 'Shopping':
        reply = "Oh master, browse now, and I will make note of it"
        print(reply)
        raise NotImplementedError("This function is not yet implemented.")
        return reply, label
    elif label == 'Flight':
        reply = "Oh master, Let me check the available flights. where and when you want to go?"
        print(reply)
        return reply, label
    elif label == 'Coding':
        reply = "Oh master, Let me know the requirements, and I will work on it now"
        print(reply)
        raise NotImplementedError("This function is not yet implemented.")

        return reply, label
    elif label == 'Task':
        reply = "Oh master, Let me know the requirements, and I will help you to arrange it in a better way"
        print(reply)
        return reply, label
    
    
    elif label == 'Other':
        query = f"{hist}. You are a friendly chat bot called 'Chamberlain', previous is a summary of our chat history (ignore it if not given), \
        now you have to answer the user's questions or conversation, and treat/ call him as a Master Nick. You must be polite and knowledgable."
        try:
            completion = question_judge.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": query},
                    {"role": "user", "content": recognize_speech}
                ]
                )
            # Extracting the text response
            reply = completion.choices[0].message.content
            print("Chamberlain said: ", reply)

        except Exception as e:
            reply = f"An error occurred: {str(e)}"
        return reply, label



