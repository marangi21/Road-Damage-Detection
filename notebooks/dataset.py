import torch
from torchvision import transforms
import os
from PIL import Image
import torch.utils.data
import xml.etree.ElementTree as ET
from pathlib import Path

class RDDdataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        #Prendi i path delle immagini e delle labels
        BASE_DIR = Path.cwd().parent
        data_dir = os.path.join(BASE_DIR, "data", "United_States")
        train_data_dir = os.path.join(data_dir, "train")
        self.train_images_path = os.path.join(train_data_dir, "images")
        self.train_labels_path = os.path.join(train_data_dir, "annotations", "xmls")

        #Lista delle immagini (strings)
        self.images = [img for img in os.listdir(self.train_images_path) if img.endswith(".jpg")]

        #Dizionario per codifica delle classi
        self.class_to_idx = {
            'D00': 0,  # Longitudinal Crack
            'D10': 1,  # Transverse Crack
            'D20': 2,  # Alligator Crack
            'D40': 3   # Pothole
        }

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        #Apri l'immagine
        img_name = self.images[index]
        img_path = os.path.join(self.train_images_path, img_name)
        img = self.transform(Image.open(img_path).convert("RGB"))

        #Prendi il path della label
        label_name = img_name.replace(".jpg", ".xml")
        label_path = os.path.join(self.train_labels_path, label_name)
        tree = ET.parse(label_path)
        root = tree.getroot()

        #Estrai bbox e classe
        bboxes = []
        classes = []
        for object in root.findall("object"):
            class_name = object.find("name").text
            classes.append(self.class_to_idx[class_name])

            bbox = object.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            bboxes.append([xmin, ymin, xmax, ymax])

        #Converti in tensori
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        classes = torch.as_tensor(classes, dtype=torch.int64)

        #Crea il dizionario delle labels
        target = {
            "boxes": bboxes,
            "labels": classes
        }

        return img, target