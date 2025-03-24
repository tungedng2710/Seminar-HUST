import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ---------------------
# 1. Hyperparameters
# ---------------------
batch_size = 64
learning_rate = 0.01
max_iterations_per_step = 20  # Number of iterations per LBFGS step
epochs = 1  # For demonstration, we set it to 1. Increase for better results.

# ---------------------
# 2. Data Preparation
# ---------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# ---------------------
# 3. Define a Simple CNN
# ---------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # A small CNN for MNIST
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))  # 16 x 14 x 14
        x = self.pool(nn.functional.relu(self.conv2(x)))  # 32 x 7 x 7
        x = x.view(x.size(0), -1)  # Flatten
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------------
# 4. Define Loss and Optimizer
# ---------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.LBFGS(model.parameters(), lr=learning_rate)

# ---------------------
# 5. Training Loop (Using LBFGS)
# ---------------------
model.train()
for epoch in range(epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)

        # LBFGS in PyTorch requires defining a closure
        def closure():
            # Clear gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            # Backprop
            loss.backward()
            return loss

        # Perform step
        # 'max_iter' controls how many times the closure is called per .step()
        # LBFGS can be slow if this is large or if the dataset is large
        loss = optimizer.step(closure)
        
        # Print training info occasionally
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

# ---------------------
# 6. Testing
# ---------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = 100.0 * correct / total
print(f"Accuracy on the test set: {accuracy:.2f}%")