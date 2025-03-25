import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from models.resnet import resnet18

# Teacher Model: ResNet-18 pre-trained on CIFAR-10
def get_teacher_model(device):
    # Create a ResNet-18 with 10 output classes
    # teacher = models.resnet18(num_classes=10)
    # # Load pretrained weights (ensure the checkpoint is available)
    # teacher.load_state_dict(torch.load("pretrained_cifar10/resnet18.pt", map_location=device))
    teacher = resnet18(pretrained=True)
    teacher.to(device)
    teacher.eval()
    # Freeze teacher parameters
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher

# Student Model: Modified LeNet-5 for 3-channel CIFAR-10 images
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Modify conv1 to accept 3 channels
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)  # For 32x32 input, output: [6, 28, 28]
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # Output: [16, 10, 10]
        # After pooling, image becomes 5x5, flatten size: 16*5*5 = 400.
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # [batch, 6, 28, 28]
        x = self.pool(x)            # [batch, 6, 14, 14]
        x = F.relu(self.conv2(x))   # [batch, 16, 10, 10]
        x = self.pool(x)            # [batch, 16, 5, 5]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

def distillation_loss(student_logits, teacher_logits, targets, T, alpha):
    # Soft targets loss (KL divergence between softened outputs)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    # Hard targets loss (cross entropy with true labels)
    hard_loss = F.cross_entropy(student_logits, targets)
    return alpha * soft_loss + (1 - alpha) * hard_loss

def train_student_distillation(student, teacher, device, train_loader, optimizer, epoch, T, alpha):
    student.train()
    teacher.eval()  # Teacher is frozen
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        student_logits = student(data)
        with torch.no_grad():
            teacher_logits = teacher(data)
        loss = distillation_loss(student_logits, teacher_logits, target, T, alpha)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"[Distillation] Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")

def train_student_scratch(student, device, train_loader, optimizer, epoch):
    student.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits = student(data)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"[Scratch] Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")

def test(model, device, test_loader):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            total_loss += F.cross_entropy(logits, target, reduction='sum').item()
            preds = logits.argmax(dim=1)
            correct += preds.eq(target).sum().item()
    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return avg_loss, accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CIFAR-10 transforms and loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True, transform=transform),
        batch_size=256, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transform),
        batch_size=1000, shuffle=False
    )

    # Initialize teacher and student models
    teacher = get_teacher_model(device)
    student_distilled = LeNet5().to(device)
    student_scratch = LeNet5().to(device)

    optimizer_distilled = optim.Adam(student_distilled.parameters(), lr=0.001)
    optimizer_scratch = optim.Adam(student_scratch.parameters(), lr=0.001)

    # Distillation hyperparameters
    T = 5.0
    alpha = 0.7
    epochs = 25

    # Lists to track test loss and accuracy for each epoch
    epochs_list = []
    distilled_loss_history = []
    distilled_acc_history = []
    scratch_loss_history = []
    scratch_acc_history = []

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_student_distillation(student_distilled, teacher, device, train_loader, optimizer_distilled, epoch, T, alpha)
        train_student_scratch(student_scratch, device, train_loader, optimizer_scratch, epoch)

        # Evaluate both student models
        dist_loss, dist_acc = test(student_distilled, device, test_loader)
        scratch_loss, scratch_acc = test(student_scratch, device, test_loader)
        print(f"Epoch {epoch} Results:")
        print(f"  Distilled Student - Loss: {dist_loss:.4f}, Accuracy: {dist_acc:.2f}%")
        print(f"  Scratch Student   - Loss: {scratch_loss:.4f}, Accuracy: {scratch_acc:.2f}%")

        # Save metrics for plotting
        epochs_list.append(epoch)
        distilled_loss_history.append(dist_loss)
        distilled_acc_history.append(dist_acc)
        scratch_loss_history.append(scratch_loss)
        scratch_acc_history.append(scratch_acc)

    # Final evaluation
    teacher_loss, teacher_acc = test(teacher, device, test_loader)
    dist_loss, dist_acc = test(student_distilled, device, test_loader)
    scratch_loss, scratch_acc = test(student_scratch, device, test_loader)
    print("\nFinal Test Results:")
    print(f"Teacher Model (ResNet-18)   - Loss: {teacher_loss:.4f}, Accuracy: {teacher_acc:.2f}%")
    print(f"Distilled Student (LeNet-5) - Loss: {dist_loss:.4f}, Accuracy: {dist_acc:.2f}%")
    print(f"Scratch Student (LeNet-5)   - Loss: {scratch_loss:.4f}, Accuracy: {scratch_acc:.2f}%")

    # Plot the test loss and accuracy for both student models
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Plot loss
    ax1.plot(epochs_list, distilled_loss_history, label='Distilled Student', marker='o')
    ax1.plot(epochs_list, scratch_loss_history, label='Scratch Student', marker='o')
    ax1.set_title('Test Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    # Plot accuracy
    ax2.plot(epochs_list, distilled_acc_history, label='Distilled Student', marker='o')
    ax2.plot(epochs_list, scratch_acc_history, label='Scratch Student', marker='o')
    ax2.set_title('Test Accuracy per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("logs/kd_results.png")
    print("Saved training curves to 'logs/kd_results.png'.")

if __name__ == '__main__':
    main()