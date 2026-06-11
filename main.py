import kagglehub
import glob
import os
import PIL.Image as Image

from torch.utils.data import Dataset

dataset_path = kagglehub.dataset_download("tolgadincer/labeled-chest-xray-images")
print(dataset_path)

class ChestXrayDataset(Dataset):
    def __init__ (self, root_dir):
        self.image_paths = glob.glob(os.path.join(root_dir, '**/*.jpeg'), recursive=True)
        self.class_to_idx = {'NORMAL': 0, 'BACTERIA': 1, 'VIRUS': 2}

    def __len__(self, root_dir):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        filename = os.path.basename(image_path)
        label_str = filename.split('-')[0]
        label = self.class_to_idx[label_str]

        return image, label

