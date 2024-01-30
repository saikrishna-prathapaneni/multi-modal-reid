import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Assuming you have a function to encode audio and images
# These should return normalized embeddings for the audio and images
def encode_audio(audio_data):
    # return encoded_audio
    pass

def encode_images(image_data):
    # return encoded_images
    pass

# Function to calculate recall at k
def calculate_recall_at_k(similarity_matrix, k):
    num_correct = 0
    for query_idx in range(similarity_matrix.shape[0]):
        # Get the indices that would sort the row
        sorted_indices = np.argsort(-similarity_matrix[query_idx])
        # Check if the correct index is within the top k
        if query_idx in sorted_indices[:k]:
            num_correct += 1
    return num_correct / similarity_matrix.shape[0]

# Load your audio and image data
audio_data = # ... load your audio descriptions
image_data = # ... load your corresponding images

# Encode your data
encoded_audio = encode_audio(audio_data)
encoded_images = encode_images(image_data)

# Compute similarity matrix
# The shape of similarity_matrix should be (num_audio_descriptions, num_images)
similarity_matrix = cosine_similarity(encoded_audio, encoded_images)

# Calculate recall@k for different values of k
recall_at_1 = calculate_recall_at_k(similarity_matrix, 1)
recall_at_5 = calculate_recall_at_k(similarity_matrix, 5)
recall_at_10 = calculate_recall_at_k(similarity_matrix, 10)

print(f'Recall@1: {recall_at_1:.4f}')
print(f'Recall@5: {recall_at_5:.4f}')
print(f'Recall@10: {recall_at_10:.4f}')
