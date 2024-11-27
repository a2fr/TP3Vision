import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomCNN, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)  # 32 filters
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # 64 filters
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # 128 filters

        # Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces spatial dimensions

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # Adjust dimensions based on image size (28x28 here)
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional Layers with ReLU and Pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten
        x = x.view(-1, 128 * 28 * 28)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    # Test the model with dummy data
    model = CustomCNN(num_classes=2)
    print(model)

    # Dummy input (batch size = 4, 3 channels, 224x224 image)
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print(output.shape)  # Should output (4, 2) for batch size 4 and 2 classes
