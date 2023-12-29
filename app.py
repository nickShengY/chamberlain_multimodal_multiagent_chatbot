import streamlit as st
from utils import *
from prompt import *
from PIL import Image

# Initialize session state variables if they don't exist
if 'image_uploaded' not in st.session_state:
    st.session_state['image_uploaded'] = False

if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None

if 'greeting' not in st.session_state:
    st.session_state['greeting'] = False
    
if 'type_chat' not in st.session_state:
    st.session_state['type_chat'] = None
    
if 'chain' not in st.session_state:
    st.session_state['chain'] = None
    
if 'go_items' not in st.session_state:
    st.session_state['go_items'] = None
    
    
if 'chat_hist' not in st.session_state:
    st.session_state['chat_hist'] = "Chat History: \n"

if 'chat_round' not in st.session_state:
    st.session_state['chat_round'] = 0
    
#Page helpers: 
def change_greeting():
    st.session_state.greeting ==True
    st.session_state['chat_round'] += 1     
        
def putinplace():        
    speech_file = text_to_speech("I now know what you bought, do you want me to put them into the fridge?")
    st.audio(speech_file, format='audio/mp3')
    input_fridge(items_got)        
        
def run_bill_port():
    
# The vectorstore to use to index the child chunks
    st.write("Running Model to inspect your documents...")
    if st.session_state['type_chat'] == 'Bill':
        op_dir = bills_dir
    else:
        op_dir = Finance_port_dir
    lotrs = []
    for file_name in os.listdir(op_dir):
        file_path = os.path.join(op_dir, file_name)

        # Ensure that it's a file and not a directory
        if os.path.isfile(file_path):
            try:
                # Process the bill file
                retriever = loader_bill(file_path)
                lotrs.append(retriever)
                # Do something with the result if needed
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    lotr = MergerRetriever(retrievers=lotrs)
    # Prompt template
    template = """Answer my question based only on the following context, which can include text and tables:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4")

    # RAG pipeline
    chain = (
        {"context": lotr, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )      
    st.session_state['chain'] = chain

# App imple starts: 
st.sidebar.title('Chat Types')
selection = st.sidebar.radio("Go to", ["First", "Second Chat"])

if selection == "First":
    if st.button('Start Speaking to Chamberlain', on_click=change_greeting):
        recognized_text = recognize_speech()
        st.write(recognized_text)
        st.session_state['greeting'] = True
        st.session_state['chat_hist'] += f'Master: {recognized_text}\n'
        response, label = response_generator(recognized_text)
        st.session_state['type_chat'] = label
        if response:
            speech_file = text_to_speech(response)
            st.audio(speech_file, format='audio/mp3')
            st.session_state['chat_hist'] += f'Chamberlain: {response}\n'
            
        
# Handling the second chat option
elif selection == "Second Chat":
    # Increment the chat round counter
    st.session_state['chat_round'] += 1
    # Handling the 'Dress' chat mode
    if st.session_state['greeting'] == True and st.session_state['type_chat'] == 'Dress':
        print("running the dress mode")
        if not st.session_state['image_uploaded']:
            # Prompt user to upload an image
            st.session_state['type_chat'] = 'Dress'
            try:
                with Image.open("D:/Projects/DeepL/NLP_app/data/photos/image.png") as img:
                    # Display the image in Streamlit
                    st.image(img, caption='Uploaded Image', use_column_width=True)
                img = encode_image("D:/Projects/DeepL/NLP_app/data/photos/image.png")
                st.success("Upload successfully.")
                response = styling(img)
            except (IOError) as e:
                st.error("The image is not readable.")
                print(e)
                st.session_state['image_uploaded'] = False  # Reset the flag if the image is not valid
            if response:
                speech_file = text_to_speech(response)
                st.session_state['chat_hist'] += f'Chamberlain: {response}\n'
                st.audio(speech_file, format='audio/mp3')
     # Handling the 'Bill' and 'Finance' chat modes           
    elif st.session_state['greeting'] == True and (st.session_state['type_chat'] == 'Bill' or st.session_state['type_chat'] == 'Finance'):
        
        if st.button('Start Speaking to Chamberlain', on_click=run_bill_port):
            recognized_text = recognize_speech()
            st.session_state['chat_hist'] += f'Master: {recognized_text}\n'
            st.write(recognized_text)
            response = st.session_state['chain'].invoke(recognized_text)

            if response:
                    speech_file = text_to_speech(response)
                    st.audio(speech_file, format='audio/mp3')
                    st.session_state['chat_hist'] += f'Chamberlain: {response}\n'
        
    # Handling the 'Grocery' chat mode    
    elif st.session_state['greeting'] == True and st.session_state['type_chat'] == 'Grocery':
        print("running the grocery mode")
        if not st.session_state['image_uploaded']:
            # Prompt user to upload an image
            st.session_state['type_chat'] = 'Grocery'
            try:
                with Image.open("D:/Projects/DeepL/NLP_app/data/reciepts/sample_recipt.jpg") as img:
                    # Display the image in Streamlit
                    st.image(img, caption='Uploaded Reciept', use_column_width=True)
                img = encode_image("D:/Projects/DeepL/NLP_app/data/reciepts/sample_recipt.jpg")
                st.success("Upload successfully.")
                # Scan the receipt and display the items
                items_got = scanner(img)
                st.session_state["go_items"] = items_got
                st.write(st.session_state["go_items"])
                st.session_state['chat_hist'] += f'Chamberlain: You got items:{items_got} from grocery\n'
            except (IOError) as e:
                st.error("The image is not readable.")
                print(e)
                st.session_state['image_uploaded'] = False  # Reset the flag if the image is not valid

        if items_got is not None:
            # Button to add items to the fridge
            if st.button('Put in the fridge?', on_click=putinplace):
                st.write("Added Items in Fridge")
                st.session_state['chat_hist'] += f'Chamberlain: Added those items in Fridge\n'
                
    # Handling 'Flight', 'Fun', and 'Task' chat modes            
    elif st.session_state['greeting'] == True and (st.session_state['type_chat'] == 'Flight' or st.session_state['type_chat'] == 'Fun' or st.session_state['type_chat'] == 'Task'):
        
        if st.button('Start Speaking to Chamberlain'):
            recognized_text = recognize_speech()
            st.session_state['chat_hist'] += f'Master: {recognized_text}\n'
            st.write(recognized_text)
            # Generate response for miscellaneous tasks using Google APis
            response = multi_use_api(recognized_text)

            if response:
                    speech_file = text_to_speech(response)
                    st.audio(speech_file, format='audio/mp3')
                    st.session_state['chat_hist'] += f'Chamberlain: {speech_file}\n'
    # Handling the 'Eat' chat mode
    elif st.session_state['greeting'] == True and (st.session_state['type_chat'] == 'Eat'): 
        
        if st.button('Start Speaking to Chamberlain'):
            recognized_text = recognize_speech()
            st.session_state['chat_hist'] += f'Master: {recognized_text}\n'
            st.write(recognized_text)
            response = get_recipe(recognized_text)
     # Return the recipe and instruction and update the fridge
            if response:
                    speech_file = text_to_speech(response)
                    st.audio(speech_file, format='audio/mp3')
                    st.session_state['chat_hist'] += f'Chamberlain: {speech_file}\n'
                

        