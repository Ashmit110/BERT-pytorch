import torch.nn as nn
import torch

class ClassifierHead(nn.Module):
    def __init__(self, input_dim):
        super(ClassifierHead, self).__init__()  
        self.input_d = input_dim
        self.linear_1 = nn.Linear(self.input_d, 1)  # Output is a single value for binary classification
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, cls_token_embedding):
        logits = self.linear_1(cls_token_embedding)  # Linear transformation
        prob = self.sigmoid(logits)  # Apply sigmoid to get a probability between 0 and 1
        return prob
