import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Linear(input_size, hidden_size)

        self.pos_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, nhead=1, dim_feedforward = hidden_size,),
            num_layers
        )

        
    def forward(self, x): #, mask):
        x = self.embedding(x)
        x = self.pos_encoder(x)


        x = x.permute(1, 0, 2)
        output = self.transformer_encoder(x)#, src_key_padding_mask=mask)
        output = output.permute(1, 0, 2)

        return output