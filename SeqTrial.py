import json
from Encoder import Encoder
from Decoder import Decoder
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from collections import OrderedDict
import pickle
import random
import os
from torch.utils.data import DataLoader
from torch.nn.functional import pairwise_distance
import pickle
import einops
import copy
import torch.nn.functional as F



# set seed
seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)


class MLMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(MLMEncoder, self).__init__()
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

        self.mlm_head = nn.Linear(hidden_size, input_size)
        
    def forward(self, x, mask):

        x = self.embedding(x)
        x = self.pos_encoder(x)


        x = x.permute(1, 0, 2)
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        output = output.permute(1, 0, 2)
        
        logits = self.mlm_head(output)
        return output, logits

class t2tVAE(nn.Module):
    def __init__(self, input_size, input_size1,  output_size, hidden_size,  num_layers):
        super(t2tVAE, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.encoder1 = Encoder(input_size, hidden_size,  num_layers)
        self.encoder2 = Encoder(input_size1, hidden_size, num_layers)
        self.decoder = Decoder(input_size, output_size, hidden_size,  num_layers)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
    def reparameterize(self, z_mean, z_logvar):
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return beta*eps * std + z_mean
    
    def forward(self, x, x_emb):
        output2 = self.encoder2 (x_emb)
        z_mean, z_logvar = self.fc1(output2), self.fc2(output2)
        z = self.reparameterize(z_mean, z_logvar)
        recon, severe_out = self.decoder(x, z)

        return recon, severe_out, z_mean, z_logvar

def create_flat_visits(data, feature_cols, patientID):
    visits_flat=[]
    for i in data[patientID].unique():
        temp = data[data[patientID] == i]
        visits_flat.append(temp[feature_cols].values.tolist())
    return visits_flat

def create_visits(data,treatment_cols,medication_cols, ae_cols ):
    visits = []
    for i in data.People.unique():
      sample=[]
      temp = data[data['People']==i]
      for index, row in temp.iterrows():
        visit=[]
        visit.append(np.nonzero(row[treatment_cols].to_list())[0].tolist())
        visit.append(np.nonzero(row[medication_cols].to_list())[0].tolist())
        visit.append(np.nonzero(row[ae_cols].to_list())[0].tolist())
        sample.append(visit)
      visits.append(sample)
    return visits

def create_biobert_visits(visits):
    visits_biobert_train=[]
    for i in range(len(visits)):
        visits_per_patient=[]
        for j in range(len(visits[i])):
            visit=[]
            for k in range(len(visits[i][j])):
                for l in visits[i][j][k]:
                    if k==0:
                        visit.append(treatment_emb_dict[treatment_id_dict[l]])
                    if k==1:
                        visit.append(med_emb_dict[l])
                    if k ==2:
                        visit.append(ae_emb_dict[l])
            visit = torch.cat(visit)
            visit = visit.mean(dim = 0).detach().numpy()
            visits_per_patient.append(visit)
        visits_biobert_train.append(visits_per_patient)
    return visits_biobert_train


def collate_fn_med(data):

    sequences, emb_seq, labels = zip(*data)

    y = torch.tensor(labels, dtype=torch.float)
    #baseline = torch.tensor(np.vstack(baselines), dtype=torch.float)
    num_patients = len(sequences)
    num_visits = [len(patient) for patient in sequences]
    max_num_visits = max(num_visits)
    lengths = [len(x) for x in sequences]

    x = torch.zeros((num_patients, max_num_visits, sum(vocab_size) ), dtype=torch.float)    
    for i_patient, patient in enumerate(sequences):
        for j_visit, visit in enumerate(patient):
                x[i_patient][j_visit] = torch.from_numpy(np.array(visit))
                
    x_ = torch.zeros((num_patients, max_num_visits, 768 ), dtype=torch.float)
    for i_patient, patient in enumerate(emb_seq):
        for j_visit, visit in enumerate(patient):
                x_[i_patient][j_visit] = torch.from_numpy(np.array(visit))
    
    masks = torch.sum(x, dim=-1) != 0
    
    return x, x_, masks, y, lengths



class CustomDataset(Dataset):
    
    def __init__(self, seqs, emb_seqs, labels):
        self.x = seqs
        self.x_= emb_seqs
        self.y = labels
    
    def __len__(self):

        return len(self.x)
    
    def __getitem__(self, index):

        return self.x[index], self.x_[index], self.y[index]











def read_config(file_path):
    with open(file_path, 'r') as config_file:
        config = json.load(config_file)
    return config

if __name__ == "__main__":
    config_file_path = "config.json"
    config = read_config(config_file_path)
    
    if torch.cuda.is_available(): 
        dev = "cuda:0" 
    else:
        dev = "cpu" 
    device = torch.device(dev)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Access hyperparameters
    learning_rate = config["hyperparameters"]["learning_rate"]
    alpha = config["hyperparameters"]["alpha"]
    beta = config["hyperparameters"]["beta"]
    num_epochs = config["hyperparameters"]["num_epochs"]
    hidden_size = config["hyperparameters"]["hidden_size"]
    # Access data paths
    train_data_path = config["data_paths"]["train_data"]
    valid_data_path = config["data_paths"]["valid_data"]
    train_label_path = config["data_paths"]["train_label"]
    valid_label_path = config["data_paths"]["valid_label"]
    ae_emb_dict = pd.read_pickle(config["data_paths"]["ae_emb_dict"])
    med_emb_dict = pd.read_pickle(config["data_paths"]["med_emb_dict"])
    treatment_emb_dict = pd.read_pickle(config["data_paths"]["treatment_emb_dict"])
    
    #Access model paths
    pretrained_encoder = config["model_paths"]["pretrained_encoder"]
    model_path = config["model_paths"]["best_model_path"]
    syn_path = config['data_paths']['syn_path']
    #read data
    train_data = pd.read_csv(train_data_path)
    valid_data = pd.read_csv(valid_data_path)
    
    
    with open(train_label_path, 'rb') as f:
        train_labels = pickle.load(f)
    with open(valid_label_path, 'rb') as f:
        valid_labels = pickle.load(f)
    
    ae_cols = [i for i in train_data.columns if i.startswith('AE_')]
    med_cols = [i for i in train_data.columns if i.startswith('CM_')]
    treatment_cols = [i for i in train_data.columns if i.startswith('Treatment_')]
    feature_cols =    treatment_cols + med_cols + ae_cols
    vocab_size = [len(treatment_cols), len(med_cols), len(ae_cols)]
    
    #load pretrained encoder
    encoder = MLMEncoder(input_size = 768, hidden_size = 256, num_layers = 1 )
    encoder.load_state_dict(torch.load(pretrained_encoder))
    
    
    treatment_id_dict={}
    for i in range(len(treatment_cols)):
        treatment_id_dict[i]=treatment_cols[i].split('_')[1]

    med_id_dict={}
    for i in range(len(med_cols)):
        med_id_dict[i]=med_cols[i].split('_')[1]

    ae_id_dict={}
    for i in range(len(ae_cols)):
        ae_id_dict[i]=ae_cols[i].split('_')[1]
        
        
    #create flat visits for training and validation data
    visits_train_flat = create_flat_visits(train_data, feature_cols, 'People')
    visits_valid_flat = create_flat_visits(valid_data, feature_cols, 'People')
    
    visits_train = create_visits(train_data, treatment_cols, med_cols, ae_cols)
    visits_valid = create_visits(valid_data, treatment_cols, med_cols, ae_cols)
    
    visits_biobert_train = create_biobert_visits(visits_train)
    visits_biobert_valid = create_biobert_visits(visits_valid)
    
    train_dataset = CustomDataset(visits_train_flat, visits_biobert_train, train_labels)
    valid_dataset = CustomDataset(visits_valid_flat, visits_biobert_valid, valid_labels)
    

    seq2seq = t2tVAE(input_size=sum(vocab_size), input_size1 = 768, output_size= sum(vocab_size[1:]), 
                     hidden_size= hidden_size, num_layers=1).to(device)

    encoder_state_dict_cuda = OrderedDict()
    for k, v in encoder.transformer_encoder.state_dict().items():
        encoder_state_dict_cuda[k] = v.to(device)

    seq2seq.encoder2.transformer_encoder.load_state_dict(encoder_state_dict_cuda)

    """for param in seq2seq.encoder2.transformer_encoder.parameters():
        param.requires_grad = False"""

    #loss
    criterion = nn.BCELoss(reduction='none')
    criterion1 = nn.MSELoss()
    optimizer = torch.optim.Adam(seq2seq.parameters(), lr=1e-3)
    
    # define batch size and number of workers for data loading
    batch_size = len(train_dataset)


    # create a DataLoader using the collate function
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_med)
    valid_dataloader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False, collate_fn=collate_fn_med)
    
    


    # Start the training loop
    best_validation_loss = float("inf")
    best_model = None
    best_epoch = 0

    for epoch in range(num_epochs):
            seq2seq.train()
            train_loss = 0
            for x, x_, masks, y, lengths in train_dataloader:

                optimizer.zero_grad()
                x_hat,severe_out,mu ,logvar= seq2seq(x.to(device), x_.to(device))
                loss1 = criterion(x_hat, x[:,:, 3:].to(device))
                loss2 = criterion1(torch.squeeze(severe_out), y.to(device))
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


                loss1 = loss1 * masks.float().unsqueeze(-1).to(device)
                loss1 = torch.mean(torch.sum(loss1, dim=[1,2]))
                loss = loss1 + kl_loss +  alpha*loss2 
                #print('loss: ', loss, 'loss2: ', alpha*loss2)
                loss.backward(retain_graph=True)
                optimizer.step()

                train_loss += loss.item()

            train_loss = train_loss / len(train_dataloader)
            print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch+1, train_loss))

            if epoch % 5 == 0:
                validation_loss = 0.0
                for x, x_,  masks, y, lengths in valid_dataloader:
                    x_hat,severe_out,mu ,logvar= seq2seq(x.to(device), x_.to(device))
                    loss1 = criterion(x_hat, x[:,:, 3:].to(device))
                    loss2 = criterion1(torch.squeeze(severe_out), y.to(device))
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                    loss1 = loss1 * masks.float().unsqueeze(-1).to(device)
                    loss1 = torch.mean(torch.sum(loss1, dim=[1,2]))
                    loss = loss1 + kl_loss + alpha*loss2
                    validation_loss += loss.item()
                validation_loss /= len(valid_dataloader)

                # Save the best model
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_model = seq2seq
                    best_epoch = epoch
                print('Epoch: {} \t Validation Loss: {:.6f}'.format(epoch+1, validation_loss))

    print("Best validation loss:", best_validation_loss, " in best epoch: ", best_epoch)
    torch.save(best_model.state_dict(), model_path+"best_model.pt")
    
    #load best model
    seq2seq.load_state_dict(torch.load(model_path+"best_model.pt"))
    
    #process synthetic data
    reconstructed_visits =[]
    severe_out =0
    seq2seq.eval()
    for x, x_, masks, y, lengths in train_dataloader:
        x_hat,severe_out,mu ,logvar= seq2seq(x.to(device), x_.to(device))
        reconstructed_visits.append(x_hat)

    reconstructed_visits = torch.round(torch.cat(reconstructed_visits, dim = 0))

    recon_patients=[]
    for i in range(len(reconstructed_visits)):
        recon_visits=[]
        for j in range(len(reconstructed_visits[i][0:lengths[i]])):
            recon_visits.append(reconstructed_visits[i][j].cpu().detach().numpy().tolist())
        recon_patients.append(recon_visits)

    feature_cols = med_cols+ae_cols
    syn_df = pd.DataFrame(columns = feature_cols)
    for i in range(len(reconstructed_visits)):
        rec = pd.DataFrame(recon_patients[i],  columns = feature_cols)
        syn_df = pd.concat([syn_df, rec], ignore_index = True)

    syn_df['People']= train_data['People'].values
    syn_df['Visit'] = train_data['Visit'].values
    syn_df[treatment_cols] = train_data[treatment_cols].values

    feature_cols = treatment_cols + med_cols + ae_cols
    syn_df = syn_df.set_index(['People', 'Visit'])[feature_cols]
    syn_df.to_csv(syn_path+'digital_twins.csv')
    
    print("Digital twins are saved in" + syn_path)
    

    

        
