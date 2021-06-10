import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        layers = []
        in_channel = 3
        out_channels = [5, 6, 7]  # 2的幂次
        conv_nums = [2, 2, 3]
        for index, out_channel in enumerate(out_channels):
            for _ in range(conv_nums[index]):
                layers += [
                    nn.Conv2d(in_channels=in_channel, out_channels=2 ** out_channel, kernel_size=3, padding=1),
                    nn.ReLU(True)]
                in_channel = 2 ** out_channel
            layers.append(nn.MaxPool2d(2, 2))
            layers.append(nn.BatchNorm2d(in_channel))
        layers += [
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, num_classes)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        x = self.layers(inputs)
        return x


