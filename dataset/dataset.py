from torchvision.datasets import ImageFolder
import torchvision.transforms as T

transform = T.Compose([
  T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
  T.ToTensor(),
  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def getDataset(train_path, val_path):
  train_dataset = ImageFolder(root=train_path, transform=transform)
  val_dataset = ImageFolder(root=val_path, transform=transform)
  
  return train_dataset, val_dataset