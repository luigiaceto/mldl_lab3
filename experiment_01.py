from models.customnet import CustomNet
from models.basicNet import BasicNet
from dataset.dataloader import getDataLoader
from train import train
from eval import validate
import torch
import os
import shutil

def setupFolders():
  with open('data/tiny-imagenet/tiny-imagenet-200/val/val_annotations.txt') as f:
    for line in f:
      fn, cls, *_ = line.split('\t')
      os.makedirs(f'data/tiny-imagenet/tiny-imagenet-200/val/{cls}', exist_ok=True)
      shutil.copyfile(f'data/tiny-imagenet/tiny-imagenet-200/val/images/{fn}', f'data/tiny-imagenet/tiny-imagenet-200/val/{cls}/{fn}')

  shutil.rmtree('data/tiny-imagenet/tiny-imagenet-200/val/images')


if __name__ == "__main__":
  setupFolders()

  train_path = 'data/tiny-imagenet/tiny-imagenet-200/train'
  val_path = 'data/tiny-imagenet/tiny-imagenet-200/val'

  model = BasicNet().cuda()
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

  best_acc = 0

  train_loader, val_loader = getDataLoader(train_path, val_path)

  # Run the training process for {num_epochs} epochs
  num_epochs = 10
  print("--- started looping on epochs ---")
  for epoch in range(1, num_epochs + 1):
    print(f"started epoch {epoch}")
    train(epoch, model, train_loader, criterion, optimizer)

    # At the end of each training iteration, perform a validation step
    val_accuracy = validate(model, val_loader, criterion)
    print(f"finished epoch {epoch}")

    # Best validation accuracy
    best_acc = max(best_acc, val_accuracy)

print(f'Best validation accuracy: {best_acc:.2f}%')
