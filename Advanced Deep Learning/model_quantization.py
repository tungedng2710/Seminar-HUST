import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from models.lenet import LeNet

# -------------------------
# 1. Define a simple LeNet
# -------------------------
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        # A classic LeNet has 2 conv layers + 2 linear layers, but we’ll keep it simple
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # input shape: (batch, 1, 28, 28)
        x = F.relu(self.conv1(x))       # -> (batch, 6, 28, 28)
        x = F.max_pool2d(x, 2)         # -> (batch, 6, 14, 14)
        x = F.relu(self.conv2(x))      # -> (batch, 16, 10, 10)
        x = F.max_pool2d(x, 2)         # -> (batch, 16, 5, 5)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))        # -> (batch, 120)
        x = F.relu(self.fc2(x))        # -> (batch, 84)
        x = self.fc3(x)                # -> (batch, 10)
        return x

# ---------------------------------------
# 2. Utilities: train loop & test loop
# ---------------------------------------
def train_one_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 200 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data):5d}/{len(train_loader.dataset):5d}] "
                  f"Loss: {loss.item():.6f}")

def test_model(model, device, test_loader, desc="Model"):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f"{desc} Accuracy: {accuracy:.2f}%")
    return accuracy

# -------------------------------
# 3. Data loading (MNIST)
# -------------------------------
def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# -----------------------------
# 4. Main script
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    epochs = 10
    batch_size = 128
    lr = 0.01

    # Get data
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

    # Create and train a LeNet model in FP32
    model_fp32 = LeNet().to(device)
    optimizer = optim.SGD(model_fp32.parameters(), lr=lr, momentum=0.9)

    for epoch in range(1, epochs + 1):
        train_one_epoch(model_fp32, device, train_loader, optimizer, epoch)
        test_model(model_fp32, device, test_loader, desc=f"FP32 (Epoch {epoch})")

    print("Final FP32 evaluation on test set:")
    baseline_acc = test_model(model_fp32, device, test_loader, desc="FP32")

    # -------------------------------------
    # 5A. Convert to FP16 (for inference)
    # -------------------------------------
    # We'll create a copy of the model and convert it. 
    # On CPU, this may have limited support. On GPU, this is more typical.
    model_fp16 = LeNet().to(device)
    model_fp16.load_state_dict(model_fp32.state_dict())  # copy weights
    model_fp16 = model_fp16.half()                       # cast to half
    # Evaluate
    # Note: For correct FP16 inference, also cast inputs to half in the test loop.
    def test_model_fp16(model, device, test_loader, desc="Model FP16"):
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                # Cast input to half
                data, target = data.to(device), target.to(device)
                data = data.half()
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        accuracy = 100.0 * correct / len(test_loader.dataset)
        print(f"{desc} Accuracy: {accuracy:.2f}%")
        return accuracy

    fp16_acc = test_model_fp16(model_fp16, device, test_loader, desc="FP16 Inference")

    # -------------------------------------
    # 5B. Convert to BF16 (for inference)
    # -------------------------------------
    # PyTorch allows .bfloat16() similarly.
    # On CPU, BF16 might still have limited support. On GPU (esp. newer versions), it can be faster.
    model_bf16 = LeNet().to(device)
    model_bf16.load_state_dict(model_fp32.state_dict())
    model_bf16 = model_bf16.to(torch.bfloat16)

    def test_model_bf16(model, device, test_loader, desc="Model BF16"):
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = data.to(torch.bfloat16)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        accuracy = 100.0 * correct / len(test_loader.dataset)
        print(f"{desc} Accuracy: {accuracy:.2f}%")
        return accuracy

    bf16_acc = test_model_bf16(model_bf16, device, test_loader, desc="BF16 Inference")

    # -------------------------------------
    # 5C. Dynamic Quantization (INT8)
    # -------------------------------------
    # In dynamic quantization, we typically quantize linear layers and LSTM layers.
    # For LeNet, let's quantize the linear layers to int8. 
    # (Conv layers aren't dynamically quantized in PyTorch.)
    # Make a copy from the trained FP32 model:
    model_int8 = LeNet()
    model_int8.load_state_dict(model_fp32.state_dict())
    model_int8.eval()

    # We'll move it to CPU for dynamic quantization (it only works on CPU).
    model_int8.cpu()
    # We specify which layers to quantize (here, all Linear layers).
    model_int8_q = torch.quantization.quantize_dynamic(
        model_int8, {nn.Linear}, dtype=torch.qint8
    )

    # Evaluate int8 model on CPU
    def test_model_int8(model, test_loader, desc="Model INT8"):
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.cpu(), target.cpu()
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        accuracy = 100.0 * correct / len(test_loader.dataset)
        print(f"{desc} Accuracy: {accuracy:.2f}%")
        return accuracy

    int8_acc = test_model_int8(model_int8_q, test_loader, desc="INT8 Dynamic Quantization")

    # ------------------------
    # 6. Compare accuracies
    # ------------------------
    print("\n--- Accuracy Summary ---")
    print(f"FP32: {baseline_acc:.2f}%")
    print(f"FP16: {fp16_acc:.2f}%")
    print(f"BF16: {bf16_acc:.2f}%")
    print(f"INT8 (dynamic): {int8_acc:.2f}%")

if __name__ == "__main__":
    main()