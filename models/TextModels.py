import torch
from transformers import BertTokenizer, BertModel


class BertEmbedding:
    def __init__(self):
        # Initialize the tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, text):
        # Encode the text, adding the special tokens needed for BERT
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids])

        # Get the embeddings
        
        outputs = self.model(input_tensor)
        last_hidden_states = outputs.last_hidden_state

        # Example: Get the embedding of the first token
        embedding_of_first_token = last_hidden_states[0][0]
        return embedding_of_first_token

# Example usage
if __name__ == "__main__":
    extractor = BertEmbedding()
    text = "Hello, my name is ChatGPT."
    embedding = extractor.get_embedding(text)
    print(embedding.shape)
