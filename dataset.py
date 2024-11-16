from torchvision.transforms import transforms
from PIL import Image
import os
from torch.utils.data import Dataset

class FlagDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}  # Mapping label names to numeric indices

        # Create a mapping of folder names (labels) to integers
        label_folders = sorted(os.listdir(root_dir))  # Ensure consistent order
        self.label_to_idx = {label: idx for idx, label in enumerate(label_folders)}

        for country_dir in label_folders:
            label = self.label_to_idx[country_dir]
            country_path = os.path.join(root_dir, country_dir)
            for image_file in os.listdir(country_path):
                image_path = os.path.join(country_path, image_file)
                self.image_paths.append(image_path)
                self.labels.append(label)

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')  # Convert to RGB format
        if self.transform:
            image = self.transform(image)
        return image, label
