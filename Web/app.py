from flask import Flask, request, render_template, redirect, url_for
from ultralytics import YOLO
import os
import cv2
import numpy as np
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any
import math
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Definir colores para cada tipo de daño
COLORS = {
    'crack': (0, 0, 255),     # Rojo
    'rebar': (0, 255, 0),     # Verde
    'spalling': (255, 0, 0),  # Azul
    'corrosion': (0, 255, 255)  # Amarillo   Los colores están invertidos
}

# Método para guardar la imagen con las detecciones
def save_image_with_detections(image_path, damages):
    image = cv2.imread(image_path)

    for damage in damages:
        x, y, w, h = damage['coordinates']
        damage_type = damage['type']
        color = COLORS.get(damage_type)  # Color gris por defecto si no se reconoce el daño
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    output_filename = f"detected_{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    cv2.imwrite(output_path, image)

    return output_filename

# Configuración de carpetas
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Cargar el modelo YOLOv8
#model = YOLO(r'C:\Users\Bryan Santos\Downloads\best.pt')
model = YOLO(r'C:\Users\Bryan Santos\Desktop\Bryan\10MO CICLO\Taller de investigación\Entrenamiento definitivo Yolov8\exp3\weights\best.pt')

@dataclass
class DamageAnalysis:
    type: str
    confidence: float
    coordinates: List[int]
    area: float
    color: List[float]
    texture: float
    severity: str
    severity_score: float
    relative_position: str

class DamageAnalyzer:
    def __init__(self):
        self.damage_types = {
            0: 'crack',
            1: 'rebar',
            2: 'spalling',
            3: 'corrosion'
        }

        # Pesos para el cálculo de severidad
        self.weights = {
            'area': 0.3,
            'confidence': 0.2,
            'texture': 0.2,
            'color': 0.15,
            'type': 0.15
        }

    def calculate_normalized_area(self, area: float, total_area: float) -> float:
        return min(1.0, area / (total_area * 0.1))  # Consideramos severo si ocupa más del 10%

    def analyze_texture(self, roi: np.ndarray) -> float:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return np.std(gray) / 128.0  # Normalizado a [0,1]

    def analyze_color_severity(self, color: List[float]) -> float:
        brightness = np.mean(color) / 255.0
        return 1.0 - brightness

    def get_type_severity_weight(self, damage_type: str) -> float:
        weights = {
            'crack': 0.9,
            'rebar': 0.7,
            'spalling': 0.8,
            'corrosion': 0.6
        }
        return weights.get(damage_type, 0.5)

    def calculate_severity(self, damage: Dict[str, Any], img_shape: tuple) -> tuple:
        total_area = img_shape[0] * img_shape[1]
        area_score = self.calculate_normalized_area(damage['area'], total_area)
        texture_score = damage['texture']
        color_score = self.analyze_color_severity(damage['color'])
        type_score = self.get_type_severity_weight(damage['type'])

        severity_score = (
                self.weights['area'] * area_score +
                self.weights['confidence'] * damage['confidence'] +
                self.weights['texture'] * texture_score +
                self.weights['color'] * color_score +
                self.weights['type'] * type_score
        )

        if severity_score > 0.7:
            return "Severa", severity_score
        elif severity_score > 0.4:
            return "Moderada", severity_score
        else:
            return "Leve", severity_score

    def get_relative_position(self, damage: Dict[str, Any], img_shape: tuple) -> str:
        x1, y1, x2, y2 = damage['coordinates']
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        if center_x < img_shape[1] / 3:
            horizontal_pos = 'izquierda'
        elif center_x > 2 * img_shape[1] / 3:
            horizontal_pos = 'derecha'
        else:
            horizontal_pos = 'centro'

        if center_y < img_shape[0] / 3:
            vertical_pos = 'superior'
        elif center_y > 2 * img_shape[0] / 3:
            vertical_pos = 'inferior'
        else:
            vertical_pos = 'medio'

        return f"{vertical_pos} {horizontal_pos}"

    def analyze_damage_pattern(self, damages: List[Dict[str, Any]], img_shape: tuple) -> Dict[str, Any]:
        if not damages:
            return {"pattern": "Sin daños", "risk_level": "Bajo"}

        centers = []
        total_area = 0
        for damage in damages:
            x1, y1, x2, y2 = damage['coordinates']
            centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
            total_area += damage['area']

        if len(centers) > 1:
            distances = []
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dist = math.sqrt((centers[i][0] - centers[j][0]) ** 2 +
                                     (centers[i][1] - centers[j][1]) ** 2)
                    distances.append(dist)

            avg_distance = np.mean(distances)
            coverage = total_area / (img_shape[0] * img_shape[1])

            if coverage > 0.2:
                pattern = "Daño extensivo"
                risk_level = "Alto"
            elif avg_distance < 100:
                pattern = "Daño concentrado"
                risk_level = "Moderado-Alto"
            else:
                pattern = "Daño disperso"
                risk_level = "Moderado"
        else:
            pattern = "Daño aislado"
            risk_level = "Bajo-Moderado"

        return {
            "pattern": pattern,
            "risk_level": risk_level,
            "coverage_percentage": round(coverage * 100, 2) if len(centers) > 1 else None
        }

    def generate_recommendations(self, damages: List[Dict[str, Any]], pattern_analysis: Dict[str, Any]) -> List[str]:
        recommendations = []
        severe_count = sum(1 for d in damages if d['severity'] == 'Severa')
        moderate_count = sum(1 for d in damages if d['severity'] == 'Moderada')

        if pattern_analysis['risk_level'] == "Alto":
            recommendations.append("Se requiere inspección estructural inmediata")
            recommendations.append("Considerar restricción de uso hasta evaluación profesional")

        if severe_count > 0:
            recommendations.append(f"Atención prioritaria a {severe_count} daños severos detectados")
            recommendations.append("Programar evaluación estructural detallada")

        if moderate_count > 2:
            recommendations.append("Implementar plan de mantenimiento preventivo")
            recommendations.append("Monitorear evolución de daños moderados")

        damage_types = set(d['type'] for d in damages)
        if 'crack' in damage_types:
            recommendations.append("Evaluar integridad estructural en zonas con grietas")
        if 'rebar' in damage_types:
            recommendations.append("Inspeccionar y evaluar la exposición del acero de refuerzo")
        if 'spalling' in damage_types:
            recommendations.append("Reparar áreas con desprendimiento de concreto")
        if 'corrosion' in damage_types:
            recommendations.append("Tratar y proteger áreas afectadas por corrosión")

        return recommendations

analyzer = DamageAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)

    files = request.files.getlist('image')
    processed_images = []

    for file in files:
        if file.filename == '':
            continue

        if not file.content_type.startswith('image/'):
            continue

        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        img = cv2.imread(image_path)
        results = model(image_path)

        damages = []

        for box in results[0].boxes.data:
            x1, y1, x2, y2, confidence, cls = box.tolist()
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            damage_type = analyzer.damage_types[int(cls)]

            # Dibujar el rectángulo (cuadro) en la imagen
            color = COLORS.get(damage_type)  # Color gris por defecto si no se reconoce el daño
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 40)
            print(damage_type)
            print(color)
            roi = img[y1:y2, x1:x2]

            area = (x2 - x1) * (y2 - y1)
            texture = analyzer.analyze_texture(roi)
            color = np.mean(roi, axis=(0, 1)).tolist()

            damage_data = {
                'type': damage_type,
                'confidence': confidence,
                'coordinates': [x1, y1, x2, y2],
                'area': area,
                'color': color,
                'texture': texture
            }

            severity, severity_score = analyzer.calculate_severity(damage_data, img.shape)
            damage_data['severity'] = severity
            damage_data['severity_score'] = severity_score
            damage_data['relative_position'] = analyzer.get_relative_position(damage_data, img.shape)

            damages.append(damage_data)

        pattern_analysis = analyzer.analyze_damage_pattern(damages, img.shape)
        recommendations = analyzer.generate_recommendations(damages, pattern_analysis)

        # Guardar imagen procesada con los cuadros
        processed_image_path = os.path.join(RESULTS_FOLDER, f'processed_{file.filename}')
        cv2.imwrite(processed_image_path, img)

        # Guardar resultados en archivo JSON
        results_data = {
            'image': file.filename,
            'damages': damages,
            'pattern_analysis': pattern_analysis,
            'recommendations': recommendations,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        json_filename = os.path.join(RESULTS_FOLDER, f'{file.filename}.json')
        with open(json_filename, 'w') as json_file:
            json.dump(results_data, json_file)

        results_data['processed_image'] = processed_image_path  # Ruta de la imagen procesada
        processed_images.append(results_data)

    return render_template('results.html', processed_images=processed_images)

if __name__ == '__main__':
    app.run(debug=True)