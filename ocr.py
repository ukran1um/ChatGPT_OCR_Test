import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
import requests
from PIL import Image
from utils import calculate_image_token_cost


detail = "high"

api_key = os.environ['OPENAI_API_KEY']

st.title('Diagram Extractor')
st.subheader('Use Public Data Only')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # Display the uploaded image
    st.image(image, caption='Uploaded JPEG Image', use_column_width=True)

    bytes_data = uploaded_file.getvalue()
    base64_image = base64.b64encode(bytes_data).decode("utf-8")

    
    # load prompt
    with open('prompt.txt', 'r') as file:
        prompt = file.read()

    # Use OpenAI's API to extract information from the image
    try:
        
        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        }

        payload = {
        "model": "gpt-4-vision-preview",
        # "response_format":{ "type": "json_object" },
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": detail
                }
                }
            ]
            }
        ],
        "max_tokens": 4096
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        # st.write(response.json())
        resp_text = response.json()['choices'][0]['message']["content"]

        #calculate costs
        image_tokens = calculate_image_token_cost(image, detail)
        response_tokens = len(resp_text)/4.0
        input_price = .01
        output_price = .03

        cost = round(image_tokens*input_price/1000 + response_tokens*output_price/1000,2)*100

        st.markdown(f"Approximate cost: :green[***{cost:.2f} cents***]")

        st.markdown(resp_text)
        

    except Exception as e:
        st.error(f"An error occurred: {e}")

