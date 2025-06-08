from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

class Model:
    def __init__(self, name, yolo_input_resolution):
        self.modeloTRT=YOLO(name)
        self.yolo_input_resolution = yolo_input_resolution
        print("modelo leído")    
        
    def preprocess(self, input_image):

        image_raw, image_resized = self._load_and_resize(input_image)
        image_preprocessed = self._shuffle_and_normalize(image_resized)
        #print(f"[DEBUG] Preprocessed image min: {image_preprocessed.min()}, max: {image_preprocessed.max()}")
        return image_raw, image_preprocessed

    def _load_and_resize(self, input_image):


        #image_raw = Image.open(input_image)
        if isinstance(input_image, np.ndarray):
            # Si es un arreglo de NumPy (frame de OpenCV), convierte de BGR a RGB
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            image_raw = Image.fromarray(input_image)
        else:
            # Si es una ruta de archivo, usa Image.open
            image_raw = Image.open(input_image)
        # Expecting yolo_input_resolution in (height, width) format, adjusting to PIL
        # convention (width, height) in PIL:
        new_resolution = (self.yolo_input_resolution[1], self.yolo_input_resolution[0])
        image_resized = image_raw.resize(new_resolution, resample=Image.BICUBIC)
        image_resized = np.array(image_resized, dtype=np.float32, order="C")
        return image_raw, image_resized

    def _shuffle_and_normalize(self, image):
        image /= 255.0
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = np.array(image, dtype=np.float32, order="C")
        # Guardar imagen preprocesada para depuración
        img_debug = image[0].transpose([1, 2, 0]) * 255.0
        img_debug = img_debug.astype(np.uint8)
        cv2.imwrite("debug_preprocessed.jpg", cv2.cvtColor(img_debug, cv2.COLOR_RGB2BGR))
        #print(f"[DEBUG] Preprocessed image min: {image.min()}, max: {image.max()}")
        return img_debug
    
    def resultss(self, image):
        results=self.modeloTRT(image)
        return results