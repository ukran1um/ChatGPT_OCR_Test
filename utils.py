from PIL import Image
import numpy as np
import hnswlib
import pandas as pd

# Load index and data
unspsc_codes_df = pd.read_csv('./data/data-unspsc-codes.csv', encoding='ISO-8859-1')
p = hnswlib.Index(space='l2', dim=384)  # the space can be changed - keeps the data, alters the distance function.
p.load_index("./data/index.bin", max_elements = len(unspsc_codes_df))

def get_index_results(string_vector, k=5):
    labels, distances = p.knn_query(string_vector, k=5)
    return unspsc_codes_df.iloc[labels[0]].head(k).to_string(index=False)
     
def get_reply(messages):
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages
    )
    return response.choices[0].message['content']

def calculate_image_token_cost(img, detail="high"):
    # Fixed cost for low detail images
    low_detail_cost = 85

    # Token cost per 512x512 square for high detail images
    high_detail_cost_per_square = 170

    # Additional fixed cost for high detail images
    additional_high_detail_cost = 85

    #get img dimensions
    width, height = img.size

    # Calculate cost for low detail images
    if detail == 'low':
        return low_detail_cost

    # Process for high detail images
    elif detail == 'high':
        # Scale down the image to fit within 2048x2048, if necessary
        if max(width, height) > 2048:
            scale_factor = 2048 / max(width, height)
            width = int(width * scale_factor)
            height = int(height * scale_factor)

        # Scale the image such that the shortest side is 768px long
        scale_factor = 768 / min(width, height)
        width = int(width * scale_factor)
        height = int(height * scale_factor)

        # Calculate the number of 512px squares needed
        num_squares = (width // 512) * (height // 512)

        # If either dimension is not exactly divisible by 512, add an extra square for the remainder
        if width % 512 != 0:
            num_squares += height // 512
        if height % 512 != 0:
            num_squares += width // 512

        # Calculate the total cost
        total_cost = num_squares * high_detail_cost_per_square + additional_high_detail_cost
        return total_cost

    else:
        raise ValueError("Detail level must be either 'low' or 'high'.")

# Example usage:
# cost = calculate_image_token_cost("path_to_your_image.jpg", "high")
# print("Token cost:", cost)