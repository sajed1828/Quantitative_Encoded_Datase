import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

#
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=72):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class genlenDataset(Dataset):
    def __init__(self, df, FEATURE_COLUMNS, qen_len):
        self.qen_len = qen_len
        self.scaler = MinMaxScaler()
        print("Original df shape:", df.shape)
        print("Missing values per column:\n", df.isna().sum())
        print("Number of inf values:\n", np.isinf(df.select_dtypes(include=[np.number])).sum())
        
        data = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")
        scaled = self.scaler.fit_transform(data)
        self.target = scaled[:, FEATURE_COLUMNS.index("close")]
        self.data = scaled

    def __len__(self):
        return len(self.data) - self.qen_len
    
    def __getitem__(self, idx):
        X = self.data[idx: idx+self.qen_len]
        y = self.target[idx+self.qen_len]
        return torch.tensor(X,dtype= torch.float), torch.tensor(y,dtype= torch.float)     


def train_autoencoder(dataloader, model, optimizer, num_epochs=10):
    model.train()
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, _ in dataloader:
            batch_x_flat = batch_x.view(batch_x.size(0), -1)
            optimizer.zero_grad()
            outputs = model(batch_x_flat)
            loss = criterion(outputs, batch_x_flat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")


def encode_data(dataloader, model):
    model.eval()
    all_encoded = []
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x_flat = batch_x.view(batch_x.size(0), -1)
            encoded = model.encoder(batch_x_flat)
            all_encoded.append(encoded)
    return torch.cat(all_encoded).numpy()

