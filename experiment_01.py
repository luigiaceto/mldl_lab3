from models.customnet import CustomNet
from models.basicNet import BasicNet
from dataset.dataloader import getDataLoader
from train import train
from eval import validate
import torch
import os
import shutil
import wandb

def setupFolders():
  original_val_images_path = 'data/tiny-imagenet/tiny-imagenet-200/val/images'

  # Esegui setupFolders() SOLO SE la cartella originale esiste ancora
  if os.path.isdir(original_val_images_path) == False:
      return
  else:
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
  
  learning_rate = 0.001
  momentum = 0.9
  num_epochs = 10
  batch_size = 32

  model = BasicNet().cuda()
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

  best_acc = 0

  train_loader, val_loader = getDataLoader(train_path, val_path, batch_size)

  wandb.init(
    project="mldl-lab3-tinyimagenet",  # Scegli un nome per il tuo progetto
    name="experiment_basicnet_01",   # Scegli un nome per questo run
    config={
      "model_architecture": "BasicNet",
      "learning_rate": learning_rate,
      "epochs": num_epochs,
      "batch_size": batch_size,
      "optimizer": "SGD_momentum"
    }
  )

  print("--- started looping on epochs ---")
  for epoch in range(1, num_epochs + 1):
    print(f"started epoch {epoch}")
    
    train_loss, train_accuracy = train(epoch, model, train_loader, criterion, optimizer)

    # At the end of each training iteration, perform a validation step
    val_loss, val_accuracy = validate(model, val_loader, criterion)
    print(f"finished epoch {epoch}")

    wandb.log({
      "epoch": epoch,
      "train_loss": train_loss,
      "train_accuracy": train_accuracy,
      "val_loss": val_loss,
      "val_accuracy": val_accuracy
    })

    # Best validation accuracy
    best_acc = max(best_acc, val_accuracy)

  print(f'Best validation accuracy: {best_acc:.2f}%')
  wandb.summary["best_val_accuracy"] = best_acc
  wandb.finish()
