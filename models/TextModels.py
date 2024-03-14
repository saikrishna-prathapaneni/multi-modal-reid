import torch
from transformers import BertModel

class BertEmbedding(torch.nn.Module):
    def __init__(self, target_embedding=1024, hidden_state = "mean"):
        super(BertEmbedding, self).__init__()
        # Initialize the BERT model
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.hidden_state_mean = hidden_state
        # Linear layer to transform the CLS token's embedding
        self.embed = torch.nn.Linear(768, target_embedding)  # Projecting from 768 to 1024 dimensions

        # Activation and normalization layers
        self.activation = torch.nn.ReLU()
        self.norm = torch.nn.LayerNorm(1024)

    def forward(self, x):
        # Pass inputs through BERT model
        outputs = self.model(input_ids=x[0], attention_mask=x[1])

        # Use the output of the CLS token
        #print("last hidden state output: ",outputs.last_hidden_state.shape)
        cls_output = None
        if self.hidden_state_mean =="mean":
            cls_output = torch.mean(outputs.last_hidden_state, dim =1)
        elif self.hidden_state_mean =="cls":
            cls_output = outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, 768] just consider the CLS token
        else:
            cls_output = outputs.last_hidden_state
         

        # Pass CLS token's output through additional layers
        x = self.embed(cls_output)  # Shape: [batch_size, 1024]
        x = self.activation(x)
        x = self.norm(x)

        return x
