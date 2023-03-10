import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(TransformerClassifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=8),
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        output = self.transformer(x)
        output = output.mean(dim=0)  # Average over the sequence length
        output = self.fc(output)
        return output


iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)

X_train = torch.from_numpy(X_train).long()
y_train = torch.from_numpy(y_train).long()

X_test = torch.from_numpy(X_test).long()
y_test = torch.from_numpy(y_test).long()

input_dim = 10  # Change input_dim to 10
output_dim = 3
hidden_dim = 32
num_layers = 2

model = TransformerClassifier(input_dim, output_dim, hidden_dim, num_layers)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print("Epoch {}: Loss = {}".format(epoch, loss.item()))

model.eval()

# Evaluate the model on the test data
with torch.no_grad():
    y_pred = model(X_test)
    loss_fn = nn.CrossEntropyLoss()
    test_loss = loss_fn(y_pred, y_test)
    test_acc = (y_pred.argmax(dim=1) == y_test).float().mean()

print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

torch.save(model.state_dict(), "weight.pth")