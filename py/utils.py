import torch
import torch.utils
import torch.utils.data
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def custom_collate_fn(batch):
    """
    Custom collate function per permettere al dataloader di ricevere immagini con diverso numero di bboxes
    
    """
    images = []
    targets = []
    
    for sample in batch:
        images.append(sample[0])
        targets.append(sample[1])
    
    images = torch.stack(images, 0)
    
    # Non stacko le labels perché non sono tutte della stessa dimensione (diverso numero di bboxes per ogni immagine)
    # Ritorno una lista di labels invece
    
    return images, targets



def visualize_batch(dataloader, num_images):
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



def plot_img_bbox(train_images, train_labels, idx):
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