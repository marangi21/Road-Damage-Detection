import torch
from torchvision import transforms
import os
from PIL import Image
import torch.utils.data
import xml.etree.ElementTree as ET
from pathlib import Path

class YOLO11dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, img_size, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

        #Lista delle immagini (strings)
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        #Apri l'immagine
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert("RGB")
        #Applica trasformazioni
        if self.transform:
            image = self.transform(image)

        #Prendi il path della label
        label_path = os.path.join(self.label_dir, os.path.splitext(self.img_files[idx])[0] + '.txt')
    
        #Estrai bbox e classe
        bboxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                #Leggi le annotazioni [class_id, x_c, y_c, w, h]
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Converti l'annotazione al formato (x_min, y_min, x_max, y_max)
                    x_min = (x_center - width / 2) * self.img_size
                    y_min = (y_center - height / 2) * self.img_size
                    x_max = (x_center + width / 2) * self.img_size
                    y_max = (y_center + height / 2) * self.img_size

                    bboxes.append([class_id, x_min, y_min, x_max, y_max])

        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        return image, bboxes