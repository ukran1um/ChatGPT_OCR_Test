import requests
import pandas as pd
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
import numpy as np
import os
import hnswlib


# Load the data
unspsc_codes_df = pd.read_csv('./data/data-unspsc-codes.csv', encoding='ISO-8859-1')

# Prepare the text inputs from the "Commodity Name" column
text_inputs = unspsc_codes_df['Commodity Name'].tolist()


########API ROUTE
# # Hugging Face API parameters
# hf_api_url = 'https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2'
# hf_token = api_key = os.environ['HF_WRITE_TOKEN']

# headers = {
#     'Authorization': f'Bearer {hf_token}'
# }

# # Function to get embeddings from Hugging Face API
# def get_embeddings(text_list):
#     responses = requests.post(hf_api_url, headers=headers, json={'inputs': text_list})
#     return responses.json()

# # Get the embeddings
# embeddings = get_embeddings(text_inputs)

#########LOCAL ROUTE


# Define the repository ID of the model you want to download
model_name = 'sentence-transformers/all-MiniLM-L6-v2'

# Download the model files to a local folder
# local_folder = snapshot_download(model_name)

# Load the model from the local folder
model = SentenceTransformer(model_name)

print("Encoding")
# Generate embeddings
embeddings = model.encode(text_inputs)
print(embeddings.shape)

print("Indexing")

# docs = DocumentArray.empty(embeddings.shape[0])
# docs.embeddings = embeddings.astype(np.float32)

num_elements, dim = embeddings.shape

# Declaring index
p = hnswlib.Index(space='l2', dim=dim)

# Initing index
# max_elements - the maximum number of elements (capacity). Will throw an exception if exceeded
# during insertion of an element.
# The capacity can be increased by saving/loading the index, see below.
#
# ef_construction - controls index search speed/build speed tradeoff
#
# M - is tightly connected with internal dimensionality of the data. Strongly affects the memory consumption (~M)
# Higher M leads to higher accuracy/run_time at fixed ef/efConstruction
p.init_index(max_elements=num_elements, ef_construction=100, M=16)
p.add_items(embeddings)
p.save_index("./data/index.bin")

