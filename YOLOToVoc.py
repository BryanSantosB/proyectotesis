import os
import xml.etree.ElementTree as ET
from PIL import Image

def convert_yolo_to_voc(yolo_file, image_path):
    try:
        # Obtener el tamaño de la imagen
        with Image.open(image_path) as img:
            image_size = img.size  # (width, height)

        root = ET.Element('annotation')
        ET.SubElement(root, 'folder').text = 'images'
        ET.SubElement(root, 'filename').text = os.path.basename(image_path)
        ET.SubElement(root, 'path').text = image_path

        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(image_size[0])
        ET.SubElement(size, 'height').text = str(image_size[1])
        ET.SubElement(size, 'depth').text = '3'  # Asumiendo imágenes RGB

        with open(yolo_file, 'r') as file:
            for line in file.readlines():
                # Ignorar líneas vacías o comentarios
                if not line.strip() or line.startswith('#'):
                    continue

                values = line.split()
                if len(values) < 5:
                    print(f'Formato incorrecto en la línea: {line.strip()}')
                    continue

                # Tomar solo los primeros 5 valores
                class_id, x_center, y_center, width, height = map(float, values[:5])
                obj = ET.SubElement(root, 'object')
                ET.SubElement(obj, 'name').text = str(int(class_id))  # Mapea class_id al nombre si es necesario
                ET.SubElement(obj, 'pose').text = 'Unspecified'
                ET.SubElement(obj, 'truncated').text = '0'
                ET.SubElement(obj, 'difficult').text = '0'
                bndbox = ET.SubElement(obj, 'bndbox')
                ET.SubElement(bndbox, 'xmin').text = str(int((x_center - width / 2) * image_size[0]))
                ET.SubElement(bndbox, 'ymin').text = str(int((y_center - height / 2) * image_size[1]))
                ET.SubElement(bndbox, 'xmax').text = str(int((x_center + width / 2) * image_size[0]))
                ET.SubElement(bndbox, 'ymax').text = str(int((y_center + height / 2) * image_size[1]))

        return ET.tostring(root, encoding='unicode')

    except Exception as e:
        print(f'Error procesando {yolo_file}: {e}')
        return None

# Directorios
yolo_dir = r'C:\Users\Bryan Santos\Desktop\Bryan\10MO CICLO\Taller de investigación\scripts\Abrir enlaces a la vez\dataset\train\labels'  # Directorio de archivos .txt
image_dir = r'C:\Users\Bryan Santos\Desktop\Bryan\10MO CICLO\Taller de investigación\scripts\Abrir enlaces a la vez\dataset\train\images'
output_dir = r'C:\Users\Bryan Santos\Desktop\Bryan\10MO CICLO\Taller de investigación\scripts\Abrir enlaces a la vez\datasetVoc'  # Directorio para guardar archivos XML

# Crear el directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Procesar todos los archivos .txt en el directorio de anotaciones
for yolo_file in os.listdir(yolo_dir):
    if yolo_file.endswith('.txt'):
        # Obtener el nombre de la imagen correspondiente
        image_name = yolo_file.replace('.txt', '.jpg')  # Cambia la extensión si es necesario
        image_path = os.path.join(image_dir, image_name)

        if os.path.exists(image_path):
            xml_output = convert_yolo_to_voc(os.path.join(yolo_dir, yolo_file), image_path)
            if xml_output:  # Solo guarda si la conversión fue exitosa
                output_file = os.path.join(output_dir, f'{image_name.replace(".jpg", ".xml")}')
                with open(output_file, 'w') as f:
                    f.write(xml_output)
        else:
            print(f'La imagen {image_name} no se encontró.')

print('Conversión completada.')
