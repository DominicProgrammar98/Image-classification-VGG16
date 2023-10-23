from Data import get_data_loader
from Model import VGG16
import torch.optim as optim
import torch
import torch.nn as nn
import time


# Define hyperparameters
num_classes = 2
num_epochs = 10
learning_rate = 0.001
batch_size = 8
image_size = 224

data_dir = r"E:\pycharm\prj\IC_VGG16\archive\train"

data_loader = get_data_loader(data_dir=data_dir, batch_size=batch_size, image_size=image_size)


model = VGG16(num_classes)
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Device setup (GPU or CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

inference_times_before_quantization = []

checkpoint_interval = 1
# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        # Measure inference time
        start_time = time.time()
        outputs = model(images)
        end_time = time.time()

        # Calculate inference time for this batch
        inference_time = end_time - start_time
        inference_times_before_quantization.append(inference_time)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}] Batch Loss: {loss.item()}, Inference Time: {inference_time:.4f} seconds")

    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(data_loader)}")

    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint_path = f"checkpoint_epoch{epoch + 1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

print("Training finished")

model = model.to(torch.float16)


model_save_path = r"E:\pycharm\prj\IC_VGG16\model_quantized.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Quantized model saved to {model_save_path}")
