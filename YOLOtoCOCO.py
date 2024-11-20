import os
import json

# Función para convertir coordenadas de YOLO (normalizadas) a COCO (coordenadas absolutas)
def yolo_to_coco(yolo_bbox, img_width, img_height):
    x_center, y_center, width, height = yolo_bbox
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    width = width * img_width
    height = height * img_height
    return [x_min, y_min, width, height]

# Estructura básica para el archivo COCO
coco_data = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 0, "name": "corrosion"},
        {"id": 1, "name": "fisura"},
        {"id": 2, "name": "piel_de_cocodrilo"}
    ]
}

# Directorio con tus imágenes y anotaciones YOLO
images_dir = r'C:\Users\Bryan Santos\Desktop\Bryan\10MO CICLO\Taller de investigación\scripts\Abrir enlaces a la vez\dataset\train\images'
labels_dir = r'C:\Users\Bryan Santos\Desktop\Bryan\10MO CICLO\Taller de investigación\scripts\Abrir enlaces a la vez\dataset\train\labels'

annotation_id = 1
img_id = 1  # Contador para los ids de las imágenes
for img_filename in os.listdir(images_dir):
    if img_filename.endswith('.jpg'):
        img_path = os.path.join(images_dir, img_filename)

        # Información de la imagen
        img_width, img_height = 1920, 1080  # Reemplazar con el tamaño de cada imagen si varía
        coco_data['images'].append({
            "file_name": img_filename,
            "id": img_id,  # Usar el contador como img_id
            "width": img_width,
            "height": img_height
        })

        # Leer archivo de anotaciones YOLO
        label_filename = img_filename.replace('.jpg', '.txt')
        label_path = os.path.join(labels_dir, label_filename)

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    # Dividir la línea en partes y asegurarse de que tenga exactamente 5 valores
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)
                        bbox = yolo_to_coco([x_center, y_center, width, height], img_width, img_height)
                        coco_data['annotations'].append({
                            "id": annotation_id,
                            "image_id": img_id,  # Asociar la anotación a la imagen usando el contador
                            "category_id": int(class_id),
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0
                        })
                        annotation_id += 1

        img_id += 1  # Incrementar el contador para la siguiente imagen

# Guardar archivo COCO en formato JSON
with open('annotations_coco.json', 'w') as json_file:
    json.dump(coco_data, json_file, indent=4)
