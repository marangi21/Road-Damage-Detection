from ultralytics import YOLO
import torch
import argparse
from datetime import datetime

#torch.cuda.get_device_name(0)
#print(torch.cuda.is_available())
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='Parametri training')
parser.add_argument('--model', type=str, default="yolov8n", help='Nome del modello')
parser.add_argument('--epochs', type=int, default=300, help='Numero di epoche di addestramento')
parser.add_argument('--batch', type=int, default=16, help='Dimensione del batch')
parser.add_argument('--pat', type=int, default=10, help='EarlyStopping patience')
args = parser.parse_args()

model_name = args.model
epochs = args.epochs
batch_size = args.batch
patience = args.pat
current_date = datetime.now().strftime("%Y%m%d_%H%M")
experiment_name = "[JAP_ONLY] " + model_name + "_b_" + str(batch_size) + "_p_" + str(patience) + "_d_" + current_date

model = YOLO(model_name)
results = model.train(
    data = r"/home/eu/Marangi/projects/Road-Damage-Detection/Jap_only/Jap_only_lincracks/jap_only.yaml",
    epochs = epochs,
    imgsz = 640,
    batch = batch_size,
    patience = patience,
    device = 0,
    name = experiment_name
)