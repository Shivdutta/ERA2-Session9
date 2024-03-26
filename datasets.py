from typing import Tuple
from torchvision import datasets, transforms


class TransformedDataset(datasets.CIFAR10):
    def __init__(self, root: str = "./data", train: bool = True, download: bool = True, transform: list = None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index: int) -> Tuple:
        image, label = self.data[index], self.targets[index]

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label
