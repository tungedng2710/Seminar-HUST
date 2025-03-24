import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import copy
import matplotlib.pyplot as plt

# ---------------------
# 1. Hyperparameters
# ---------------------
batch_size = 64
learning_rate = 0.01
epochs = 5             # Number of epochs to train
max_iter_lbfgs = 20    # LBFGS parameter

# ---------------------
# 2. Data Preparation (MNIST)
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
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # 1->6
        self.pool  = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) # 6->16

        # After two conv + pool layers, feature map size is 16 x 4 x 4 for MNIST
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.relu  = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)                
        x = self.relu(self.conv2(x))
        x = self.pool(x)                
        x = x.view(x.size(0), -1)       # flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.log_softmax(x)

# ---------------------
# 4. Setup Device, Loss
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

# ---------------------
# 5. Training Function
# ---------------------
def train_one_epoch(model, optimizer, loader, device):
    """
    Trains the model for one epoch using the specified optimizer.
    Returns the average loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        
        if isinstance(optimizer, torch.optim.LBFGS):
            # LBFGS requires a closure
            def closure():
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
        else:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

# ---------------------
# 6. Testing Function
# ---------------------
def test_model(model, loader, device):
    """
    Evaluates the model on the given loader.
    Returns test accuracy (in percentage).
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    
    accuracy = 100.0 * correct / total
    return accuracy

# ---------------------
# 7. Compare Multiple Optimizers
# ---------------------
optimizers_to_try = {
    "SGD": lambda params: optim.SGD(params, lr=learning_rate),
    "Adam": lambda params: optim.Adam(params, lr=learning_rate),
    "RMSprop": lambda params: optim.RMSprop(params, lr=learning_rate),
    # "LBFGS": lambda params: optim.LBFGS(params, lr=learning_rate, max_iter=max_iter_lbfgs)
}

# Dictionary to store loss curves for each optimizer
loss_curves = {name: [] for name in optimizers_to_try.keys()}
results = {}

for opt_name, opt_func in optimizers_to_try.items():
    print(f"\nTraining with {opt_name} optimizer...")
    
    # Create a fresh model for each optimizer
    model = LeNet().to(device)
    
    # Create the optimizer
    optimizer = opt_func(model.parameters())
    
    # Train for multiple epochs, track loss
    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device)
        loss_curves[opt_name].append(avg_loss)
        
        print(f"[{opt_name}] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluate on the test set
    acc = test_model(model, test_loader, device)
    results[opt_name] = acc
    print(f"[{opt_name}] Final Test Accuracy: {acc:.2f}%")

# ---------------------
# 8. Plot and Save Loss Curves
# ---------------------
plt.figure()
for opt_name, losses in loss_curves.items():
    plt.plot(losses, label=opt_name)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Loss Curves for Different Optimizers (MNIST)")
plt.legend()

# Save the figure to a file
plt.savefig("loss_curves.png")
print("\nLoss curves saved to 'loss_curves.png'.")

# ---------------------
# 9. Print Final Results
# ---------------------
print("\nFinal Test Accuracy Comparison:")
for opt_name, acc in results.items():
    print(f"{opt_name}: {acc:.2f}%")