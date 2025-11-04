import torch
from dataset import getDataset

def getDataLoader(train_path, val_path):
  train, val = getDataset(train_path, val_path)

  train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True, num_workers=2)
  val_loader = torch.utils.data.DataLoader(val, batch_size=32, shuffle=False)

  return train_loader, val_loader