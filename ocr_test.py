import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
import requests
from PIL import Image
from utils import calculate_image_token_cost, get_index_results
import openai
import json
from sentence_transformers import SentenceTransformer
import hnswlib

#set params and variables
DETAIL = "high"
MODEL =  "gpt-4"
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
api_key = os.environ['OPENAI_API_KEY']

#load model
model = SentenceTransformer(EMBEDDING_MODEL)


st.title('Diagram Extractor')
st.subheader('Use Public Data Only')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # Display the uploaded image
    st.image(image, caption='Uploaded JPEG Image', use_column_width=True)

    bytes_data = uploaded_file.getvalue()
    base64_image = base64.b64encode(bytes_data).decode("utf-8")

    
    # load prompts
    with open('prompt1.txt', 'r') as file:
        PROMPT1 = file.read()

    with open('prompt2.txt', 'r') as file:
        PROMPT2 = file.read()

    with open('prompt3.txt', 'r') as file:
        PROMPT3 = file.read()
    print("1")
    
    
    ### Use OpenAI's API to get the image name and image description for USNPSC code lookup
    try:
        
        messages= [
            {
            "role": "user",
            "content": [
                {"type": "text",  "text": PROMPT2},
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": DETAIL
                }
                }
            ]
            }
        ],
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens = 4096,
            temperature = 0
            # response_format={ "type": "json_object" }
            )
        print("2")
        # st.write(response.json())
        resp_text2 = response['choices'][0]['message']["content"]
        print(resp_text2)
        # resp_json = json.loads(resp_text2)
        # print(resp_json)
        #embed input
        string_vector = model.encode([resp_text2])
        print(string_vector)
        results = get_index_results(string_vector)
        print(results)
    except Exception as e:
        st.error(f"An error occurred: {e}")
    
    ### Use OpenAI's API to get the best matching UNSPNC code input.
    try:
        
        messages= [
            {
            "role": "system",
            "content": PROMPT3 + resp_text2 +"\nAnd here is are the potential USNPSC matches:\n"+results
            }
        ]

        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=messages,
            max_tokens = 4096,
            temperature = 0
            )
        
        # st.write(response.json())
        resp_text3 = response['choices'][0]['message']["content"]
        st.markdown(resp_text3)
    except Exception as e:
        st.error(f"An error occurred: {e}")

    ### Use OpenAI's API to extract information from the image and structure it
    try:
        
        messages= [
            {
            "role": "user",
            "content": [
                {"type": "text",  "text": PROMPT1},
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": DETAIL
                }
                }
            ]
            }
        ],
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens = 4096,
            temperature = 0
            )
        
        # st.write(response.json())
        resp_text = response['choices'][0]['message']["content"]

        #calculate costs
        image_tokens = calculate_image_token_cost(image, DETAIL)
        response_tokens = len(resp_text)/4.0
        input_price = .01
        output_price = .03

        cost = round(image_tokens*input_price/1000 + response_tokens*output_price/1000,2)*100

        # st.markdown(f"Approximate cost: :green[***{cost:.2f} cents***]")

        st.markdown(resp_text)
        

    except Exception as e:
        st.error(f"An error occurred: {e}")