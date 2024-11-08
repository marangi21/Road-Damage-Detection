import os
import torch
import torch.utils
import torch.utils.data
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pascal import annotation_from_xml

def custom_pascalVOC_collate_fn(batch):
    """
    Custom collate function per permettere al dataloader di ricevere immagini con diverso numero di bboxes (Pascal_VOC labels)
    Ritorna una immagine e una lista di bounding boxes

    """
    images = []
    targets = []
    
    for sample in batch:
        images.append(sample[0])
        targets.append(sample[1])
    
    images = torch.stack(images, 0)
    return images, targets

##########################################################################################################################################################

def pascalVOC_to_YOLO(input_dir, output_dir) -> None:
    """
    Converte i files .xml con annotazioni pascalVOC in file .txt con annotazioni YOLO

    """

    # Mappatura delle classi
    label_map = {
        'D00': 0,  # Longitudinal Crack
        'D01': 0,  # Longitudinal Crack
        'D10': 1,  # Transverse Crack
        'D11': 1,  # Transverse Crack
        'D20': 2,  # Alligator Crack
        'D40': 3   # Pothole
    }

    # Assicurati che la cartella di output esista
    os.makedirs(output_dir, exist_ok=True)

    # Funzione per convertire e salvare ogni file XML
    for xml_file in os.listdir(input_dir):
        if xml_file.endswith('.xml'):
            ann_file = os.path.join(input_dir, xml_file)

            # Legge il file XML
            ann = annotation_from_xml(ann_file)

            # Converte in formato YOLO
            yolo_ann = ann.to_yolo(label_map)

            # Salva il file nel formato YOLO con lo stesso nome dell'immagine
            output_path = os.path.join(output_dir, os.path.splitext(xml_file)[0] + '.txt')
            with open(output_path, 'w') as f:
                f.write(yolo_ann)

##########################################################################################################################################################

def save_sorted_folder_structure(root_dir, output_file='folder_structure.txt'):
    '''
    Stampa la folder structure dalla cartella root_dir in giù in un file di testo
    Non include i nomi dei files, solo le cartelle
    
    '''
    with open(output_file, 'w') as f:
        for dirpath, dirnames, _ in os.walk(root_dir):
            # Ordina le directory in ordine alfabetico
            dirnames.sort()
            level = dirpath.replace(root_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            f.write(f"{indent}{os.path.basename(dirpath)}/\n")

##########################################################################################################################################################

def visualize_batch(dataloader, num_images) -> None:
    """
    Visualizzazione di un batch di immagini con relative bboxes per 
    verificare la correttezza del processo di data loading

    """
    #prendi un batch di immagini
    images, targets = next(iter(dataloader))

    for idx in range(min(num_images, len(images))):
        #visualizza le immagini
        plt.figure(figsize=(10, 10))
        plt.imshow(images[idx].permute(1, 2, 0))

        #plotta bounding boxes
        boxes = targets[idx]['boxes']
        for box in boxes:
            x1, y1, x2, y2 = box.numpy()
            rectangle = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='b', facecolor='none')
            plt.gca().add_patch(rectangle)
        plt.axis('off')
        plt.show()

##########################################################################################################################################################

def plot_img_bbox(train_images, train_labels, idx) -> None:
    """
    Plot di immagini di training e relative bounding boxes per 
    verificare la correttezza delle annotazioni

    """
    
    img_0 = train_images[idx]
    root_0 = train_labels[idx].getroot()

    fig, ax = plt.subplots()
    ax.imshow(img_0)

    for object in root_0.findall('object'):
        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='b', facecolor='none')

        ax.add_patch(rect)
        class_name = object.find('name').text
        plt.text(xmin, ymin-5, class_name, color='blue')
    plt.show()