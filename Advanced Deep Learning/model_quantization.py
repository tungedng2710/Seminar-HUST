import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import time

# -------------------------
# 1. Define a simple LeNet
# -------------------------
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        # Classic LeNet has 2 conv layers + 2 linear layers, but this is a slightly adapted version
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # x shape: (batch, 1, 28, 28)
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

def measure_inference_performance(model, device, test_loader, desc="Model", dtype=None):
    """
    Evaluates accuracy and measures inference time (forward pass over the entire test set).
    Optionally casts input data to `dtype` (e.g., torch.float16 or torch.bfloat16).
    Returns (accuracy, total_time_seconds, samples_per_second).
    """
    model.eval()
    correct = 0
    total_samples = len(test_loader.dataset)

    # Synchronize before timing (esp. if using GPU)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if dtype is not None:
                data = data.to(dtype)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    # Synchronize again to ensure all kernels have finished
    if device.type == 'cuda':
        torch.cuda.synchronize()

    end = time.time()
    total_time = end - start
    accuracy = 100.0 * correct / total_samples
    samples_per_second = total_samples / total_time
    print(f"{desc} -> Accuracy: {accuracy:.2f}%, Inference time: {total_time:.4f}s, "
          f"Throughput: {samples_per_second:.2f} samples/s")
    return accuracy, total_time, samples_per_second

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
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# -----------------------------
# 4. Main script
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    epochs = 2
    batch_size = 64
    lr = 0.01

    # Get data
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

    # 4A: Create and train a LeNet model in FP32
    model_fp32 = LeNet().to(device)
    optimizer = optim.SGD(model_fp32.parameters(), lr=lr, momentum=0.9)

    for epoch in range(1, epochs + 1):
        train_one_epoch(model_fp32, device, train_loader, optimizer, epoch)
        measure_inference_performance(model_fp32, device, test_loader, desc=f"[FP32, Epoch {epoch}]")

    print("Final FP32 evaluation on test set:")
    baseline_acc, _, _ = measure_inference_performance(model_fp32, device, test_loader, desc="[FP32 baseline]")

    # ------------------------------------------------------
    # 4B. Convert to FP16 for inference (on GPU typically)
    # ------------------------------------------------------
    model_fp16 = LeNet().to(device)
    model_fp16.load_state_dict(model_fp32.state_dict())  # copy weights
    model_fp16.half()  # cast all parameters to half

    # Evaluate (with input data cast to half)
    fp16_acc, _, _ = measure_inference_performance(model_fp16, device, test_loader,
                                                   desc="[FP16 inference]",
                                                   dtype=torch.float16)

    # ------------------------------------------------------
    # 4C. Convert to BF16 for inference
    # ------------------------------------------------------
    model_bf16 = LeNet().to(device)
    model_bf16.load_state_dict(model_fp32.state_dict())
    model_bf16 = model_bf16.to(torch.bfloat16)

    bf16_acc, _, _ = measure_inference_performance(model_bf16, device, test_loader,
                                                   desc="[BF16 inference]",
                                                   dtype=torch.bfloat16)

    # ------------------------------------------------------
    # 4D. Dynamic Quantization (INT8) on CPU
    # ------------------------------------------------------
    # We'll move the original FP32 model to CPU and quantize its linear layers.
    # Convolutions are not dynamically quantized in PyTorch.
    model_int8 = LeNet()
    model_int8.load_state_dict(model_fp32.state_dict())
    model_int8.eval()
    # model_int8.cpu()

    # We specify which layers to quantize; here we use {nn.Linear}
    model_int8_q = torch.quantization.quantize_dynamic(
        model_int8, {nn.Linear}, dtype=torch.qint8
    )

    # Evaluate on CPU
    device_cpu = torch.device("cpu")  # dynamic quant only works on CPU
    device_gpu = torch.device("cuda")
    int8_acc, _, _ = measure_inference_performance(model_int8_q, device, test_loader,
                                                   desc="[INT8 dynamic quant]")

    # ------------------------
    # 5. Compare accuracies
    # ------------------------
    print("\n--- Accuracy Summary ---")
    print(f"FP32: {baseline_acc:.2f}%")
    print(f"FP16: {fp16_acc:.2f}%")
    print(f"BF16: {bf16_acc:.2f}%")
    print(f"INT8 (dynamic): {int8_acc:.2f}%")

if __name__ == "__main__":
    main()