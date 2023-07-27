import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size,  num_layers):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_size, nhead=1, dim_feedforward = hidden_size),
            num_layers
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.fc_1 = nn.Linear(hidden_size, 10)
        self.fc1 = nn.Linear(10, 1)

    def forward(self, x, memory):
        x = self.embedding(x)
        #x = self.pos_decoder(x)

        output = self.transformer_decoder(x, memory)
        intermediate_severe = F.relu(self.fc_1(output))
        severe_out = self.fc1(intermediate_severe)
        severe_out = F.relu(torch.sum(severe_out, dim = 1))
    
        output = F.sigmoid(self.fc(output))
        
        return output, severe_out
    
