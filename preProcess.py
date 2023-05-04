import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Load the CSV data into a Pandas DataFrame
df = pd.read_csv('variables.csv')

# Convert the date column to a datetime object and extract year and month
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df = df.drop('date', axis=1)

# Drop the columns containing string values
df = df.drop('area', axis=1)
df = df.drop('code', axis=1)
df = df.drop('no_of_crimes', axis=1)

# Split the data into inputs (X) and targets (y)
X = df.drop(['average_price'], axis=1).values
y = df['average_price'].values.reshape(-1, 1)

# Scale the input data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert X and y to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Define a linear regression model
class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

# Create a PyTorch Dataset
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        # Return a single data point
        return self.X[index], self.y[index]

    def __len__(self):
        # Return the number of data points
        return len(self.X)

dataset = MyDataset(X, y)
# Create a PyTorch DataLoader for the dataset
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Instantiate the model and optimizer
model = LinearRegression(X.shape[1])
optimizer = torch.optim.SGD(model.parameters(), lr=0.000001)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, targets = batch
        outputs = model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Predict the housing prices for new data
new_data = torch.tensor([[1996, 2, 17,1], [1997, 3, 7,1], [1998, 4, 14,1]], dtype=torch.float32)
new_data = scaler.transform(new_data)
predictions = model(torch.tensor(new_data, dtype=torch.float32))
print(predictions)
