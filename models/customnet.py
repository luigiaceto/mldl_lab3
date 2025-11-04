from torch import nn

# Define the custom neural network
# Eredita da nn.Module così da poter definire il metodo forward() etc
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        # Lo 'stride' di default è 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), # -> [64, 224, 224]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # -> [64, 112, 112]
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # -> [128, 112, 112],
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # -> [128, 56, 56]
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # -> [256, 56, 56],
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # -> [256, 56, 56]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # -> [256, 28, 28]
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1), # -> [512, 28, 28]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # -> [512, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # -> [512, 14, 14]
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # -> [512, 14, 14]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # -> [512, 7, 7]
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # -> [512, 1, 1]
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 200) # in TinyImageNet ci sono 200 classi
        )

    def forward(self, x):
        # Define forward pass

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        logits = self.head(x) # tensore [batch_size, 200]

        return logits