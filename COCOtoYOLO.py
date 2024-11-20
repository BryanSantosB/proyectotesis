import json
import os

# Ruta al archivo de anotaciones y a las imágenes
annotation_file = r'C:\Users\Bryan Santos\Downloads\json\dataset corrosao.v5i.coco-segmentation\valid\_annotations.coco.json'
image_folder = r'C:\Users\Bryan Santos\Downloads\json\dataset corrosao.v5i.coco-segmentation\valid'

# Crear carpeta para las anotaciones YOLO
yolo_labels_folder = os.path.join(image_folder, 'labels')
os.makedirs(yolo_labels_folder, exist_ok=True)

# Cargar las anotaciones COCO
with open(annotation_file) as f:
    coco_data = json.load(f)


# Función para convertir anotaciones COCO a YOLO
def convert_coco_to_yolo(coco_data, image_folder, yolo_labels_folder):
    for image in coco_data['images']:
        image_id = image['id']
        file_name = image['file_name']
        width = image['width']
        height = image['height']

        yolo_annotation = []

        for annotation in coco_data['annotations']:
            if annotation['image_id'] == image_id:
                category_id = annotation['category_id'] - 1  # YOLO class index starts from 0
                bbox = annotation['bbox']

                # Convert COCO bbox to YOLO format
                x_center = (bbox[0] + bbox[2] / 2) / width
                y_center = (bbox[1] + bbox[3] / 2) / height
                bbox_width = bbox[2] / width
                bbox_height = bbox[3] / height

                yolo_annotation.append(f"{category_id} {x_center} {y_center} {bbox_width} {bbox_height}")

        # Guardar anotaciones YOLO en un archivo
        yolo_label_file = os.path.join(yolo_labels_folder, file_name.replace('.jpg', '.txt'))
        with open(yolo_label_file, 'w') as f:
            f.write('\n'.join(yolo_annotation))


# Convertir anotaciones
convert_coco_to_yolo(coco_data, image_folder, yolo_labels_folder)
