import os
import cv2
import numpy as np
import shutil
import re
from PIL import Image


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def yolo_to_pixel(box, image_width, image_height):
    x_center, y_center, width, height = box
    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    return x1, y1, x2, y2


def find_image_file(images_dir, base_name):
    base_name_pattern = re.escape(base_name).replace(r'\.', r'.*?')
    pattern = re.compile(f'^{base_name_pattern}.*\.(jpg|jpeg|png)$', re.IGNORECASE)

    for file in os.listdir(images_dir):
        if pattern.match(file):
            return os.path.join(images_dir, file)
    return None


def process_dataset(images_dir, labels_dir, output_dir, problem_dir):
    create_dir(output_dir)
    create_dir(problem_dir)

    print(f"Permisos de escritura en el directorio de salida: {os.access(output_dir, os.W_OK)}")
    print(f"Espacio libre en disco: {shutil.disk_usage(output_dir).free / (1024 * 1024 * 1024):.2f} GB")

    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            base_name = os.path.splitext(label_file)[0]
            label_path = os.path.join(labels_dir, label_file)

            image_path = find_image_file(images_dir, base_name)

            if image_path is None:
                print(f"No se encontró imagen para {label_file}")
                shutil.copy(label_path, os.path.join(problem_dir, label_file))
                continue

            try:
                with Image.open(image_path) as img:
                    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Error al leer la imagen: {image_path}")
                print(f"Detalles del error: {str(e)}")
                shutil.copy(image_path, os.path.join(problem_dir, os.path.basename(image_path)))
                shutil.copy(label_path, os.path.join(problem_dir, label_file))
                continue

            height, width = image.shape[:2]

            with open(label_path, 'r') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                try:
                    values = line.strip().split()
                    if len(values) >= 5:
                        class_id, x_center, y_center, box_width, box_height = map(float, values[:5])
                        class_id = int(class_id)

                        x1, y1, x2, y2 = yolo_to_pixel([x_center, y_center, box_width, box_height], width, height)

                        # Asegurar que las coordenadas estén dentro de los límites de la imagen
                        x1, x2 = max(0, x1), min(width, x2)
                        y1, y2 = max(0, y1), min(height, y2)

                        if x2 <= x1 or y2 <= y1:
                            print(f"Advertencia: Coordenadas inválidas en {label_file}, línea {i + 1}")
                            continue

                        cropped_img = image[y1:y2, x1:x2]

                        if cropped_img.size == 0:
                            print(f"Advertencia: Imagen recortada vacía en {label_file}, línea {i + 1}")
                            continue

                        # Comprobar tamaño mínimo
                        min_size = 10  # Tamaño mínimo en píxeles
                        if cropped_img.shape[0] < min_size or cropped_img.shape[1] < min_size:
                            print(f"Advertencia: Imagen recortada demasiado pequeña en {label_file}, línea {i + 1}")
                            print(f"Tamaño de la imagen recortada: {cropped_img.shape[0]}x{cropped_img.shape[1]}")
                            continue

                        class_dir = os.path.join(output_dir, str(class_id))
                        create_dir(class_dir)

                        output_filename = f"{base_name}_{i}{os.path.splitext(image_path)[1]}"
                        output_path = os.path.join(class_dir, output_filename)

                        try:
                            # Intento 1: Usar cv2.imwrite
                            success = cv2.imwrite(output_path, cropped_img)
                            if success and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                                print(f"Imagen recortada guardada (cv2): {output_path}")
                            else:
                                print(f"Error: No se pudo guardar la imagen con cv2 en {output_path}")

                                # Intento 2: Usar PIL
                                pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                                pil_img.save(output_path)
                                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                                    print(f"Imagen recortada guardada (PIL): {output_path}")
                                else:
                                    print(f"Error: No se pudo guardar la imagen con PIL en {output_path}")
                                    raise Exception("No se pudo guardar la imagen con ningún método")

                        except Exception as e:
                            print(f"Error al guardar la imagen: {str(e)}")
                            print(f"Ruta de salida: {output_path}")
                            print(
                                f"Permisos de escritura en el directorio: {os.access(os.path.dirname(output_path), os.W_OK)}")
                            print(f"Tamaño de la imagen recortada: {cropped_img.shape[0]}x{cropped_img.shape[1]}")
                            print(f"Coordenadas de recorte: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                            print(f"Tipo de datos de la imagen: {cropped_img.dtype}")
                            print(f"Valores mínimo y máximo de la imagen: {np.min(cropped_img)}, {np.max(cropped_img)}")

                except ValueError:
                    print(f"Error al procesar línea en {label_file}: {line}")
                except Exception as e:
                    print(f"Error inesperado al procesar {label_file}: {str(e)}")

    print("Proceso completado.")


# Uso del script
images_dir = r"C:\Users\Bryan Santos\Desktop\Bryan\10MO CICLO\Taller de investigación\scripts\Abrir enlaces a la vez\dataset\train\images"
labels_dir = r"C:\Users\Bryan Santos\Desktop\Bryan\10MO CICLO\Taller de investigación\scripts\Abrir enlaces a la vez\dataset\train\labels"
output_dir = r"C:\Users\Bryan Santos\Desktop\Bryan\10MO CICLO\Taller de investigación\scripts\Abrir enlaces a la vez\ImagenesClasificacion"
problem_dir = r"C:\Users\Bryan Santos\Desktop\Bryan\10MO CICLO\Taller de investigación\scripts\Abrir enlaces a la vez\ProblematicFiles"

process_dataset(images_dir, labels_dir, output_dir, problem_dir)