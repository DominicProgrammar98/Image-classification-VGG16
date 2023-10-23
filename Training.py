
from Model import model
from Data import data_loader
import torch.optim as optim
import torch
import torch.nn as nn
import time

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

checkpoint_interval = 1
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
