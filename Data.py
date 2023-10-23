
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_data_loader(data_dir, batch_size, image_size, shuffle=True):
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create an ImageFolder dataset
    dataset = ImageFolder(data_dir, transform=transform)

    # Create a data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader

if __name__ == "__main__":
    data_dir = "E:\pycharm\prj\IC_VGG16\archive\train"
    batch_size = 8
    image_size = 224

    # Test the data loader
    data_loader = get_data_loader(data_dir, batch_size, image_size)
    for images, labels in data_loader:
        print(f"Batch shape: {images.shape}, Labels: {labels}")
