import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from ultralytics import YOLO
from PIL import Image

def load_test_dataset(images_dir, labels_file):
    """
    Devuelve:
      - images: lista de rutas a las imágenes de prueba
      - y_true: lista de enteros con la etiqueta real de cada imagen
    Assumes labels_file lines: "imagen.jpg  clase"
    """
    images, y_true = [], []
    with open(labels_file, 'r') as f:
        for line in f:
            fn, cls = line.strip().split()
            images.append(os.path.join(images_dir, fn))
            y_true.append(int(cls))
    return images, y_true

def predict_labels(model, images, input_size=(840, 840)):
    """
    Para cada imagen:
      - carga con PIL y convierte a RGB
      - redimensiona a input_size
      - hace inferencia y elige la clase de mayor confianza
    """
    y_pred = []
    for img_path in images:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(input_size)  
        results = model(img, imgsz=input_size)[0]
        if len(results.boxes) > 0:
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)
            confs   = results.boxes.conf.cpu().numpy()
            best = np.argmax(confs)
            y_pred.append(int(cls_ids[best]))
        else:
            # Sin detecciones → clase background (-1) o ajusta a tu convención
            y_pred.append(-1)
    return y_pred

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation='vertical')
    plt.title("Matriz de Confusión (840×840 RGB)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Ruta a tu modelo (.pt o engine)
    model_path = "Modelos/yolov8l.pt"
    images_dir  = "data/test_images"
    labels_file = "data/test_labels.txt"
    class_names = ["clase0", "clase1", "clase2", "..."]

    # Carga el modelo
    model = YOLO(model_path)

    # Carga datos y etiquetas reales
    images, y_true = load_test_dataset(images_dir, labels_file)

    # Predice usando tamaño de entrada 840×840
    y_pred = predict_labels(model, images, input_size=(840, 840))

    # Dibuja la matriz de confusión
    plot_confusion_matrix(y_true, y_pred, class_names)
