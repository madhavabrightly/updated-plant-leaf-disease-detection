import os
import json
import sys
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

# ==================== PYTORCH DEVICE CONFIGURATION ====================
print("\n" + "="*70)
print("FLASK APP STARTUP - PYTORCH MODEL")
print("="*70)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pt')
LABELS_PATH = os.path.join(BASE_DIR, 'classes.json')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    backend = 'AMD ROCm' if torch.version.hip else 'NVIDIA CUDA'
    print(f"PyTorch device: {backend} - {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch device: CPU")

app = Flask(__name__)

# =======================  MODEL LOADING  =======================
print(f"Loading trained model: {MODEL_PATH}")
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    checkpoint_classes = checkpoint['classes']

    model = models.efficientnet_b0(weights=None)
    input_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(input_features, len(checkpoint_classes)),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    confidence_temperature = max(float(checkpoint.get('temperature', 1.0)), 0.05)

    image_size = checkpoint.get('image_size', 224)
    normalization = checkpoint.get('normalization', {})
    inference_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=normalization.get('mean', [0.485, 0.456, 0.406]),
            std=normalization.get('std', [0.229, 0.224, 0.225]),
        ),
    ])
    print(f"Model loaded successfully on {device.type.upper()}")
    print(f"Checkpoint validation accuracy: {checkpoint.get('validation_accuracy', 0):.4f}")
except Exception as e:
    print(f'ERROR: Failed to load model.pt: {e}')
    print('Make sure model.pt exists in the project root')
    sys.exit(1)

# Load label mapping
print("\nLoading class labels...")
class_names = checkpoint_classes
if os.path.exists(LABELS_PATH):
    try:
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            file_classes = json.load(f)
        if file_classes != checkpoint_classes:
            raise ValueError('classes.json does not match the class order stored in model.pt')
        class_names = file_classes
    except Exception as e:
        print(f'WARNING: {e}. Using labels embedded in model.pt.')

labels = {i: name for i, name in enumerate(class_names)}
print(f'Loaded {len(labels)} class labels')

REMEDY_MAP = {
    'Pepper__bell___Bacterial_spot': (
        'Cause: Bacterial infection. Spray copper fungicide or streptomycin in early stages. ' 
        'Prevent with disease-free seeds, crop rotation, and avoid overhead watering.'
    ),
    'Pepper__bell___healthy': (
        'Plant is good. Maintain balanced NPK fertilizer, proper spacing, and regular inspection.'
    ),
    'Potato___Early_blight': (
        'Cause: Alternaria fungus. Spray Mancozeb or Chlorothalonil and remove infected leaves. ' 
        'Prevent with crop rotation and consistent watering.'
    ),
    'Potato___Late_blight': (
        'Cause: Phytophthora. Spray Metalaxyl+Mancozeb or Cymoxanil and remove infected plants. ' 
        'Prevent with good drainage and avoid wet leaves.'
    ),
    'Potato___healthy': (
        'Plant is good. Maintain loose soil, proper irrigation, and disease monitoring.'
    ),
    'Tomato_Bacterial_spot': (
        'Spray copper treatment and remove infected leaves. Prevent by avoiding leaf wetness and using resistant varieties.'
    ),
    'Tomato_Early_blight': (
        'Spray Mancozeb or Chlorothalonil. Prevent with mulching and crop rotation.'
    ),
    'Tomato_Late_blight': (
        'Spray Metalaxyl-based fungicides and remove infected plants quickly.'
    ),
    'Tomato_Leaf_Mold': (
        'Spray copper or sulfur fungicides. Prevent by improving ventilation and reducing humidity.'
    ),
    'Tomato_Septoria_leaf_spot': (
        'Spray Chlorothalonil or Mancozeb. Prevent by removing lower leaves and avoiding water splashes.'
    ),
    'Tomato_Spider_mites_Two_spotted_spider_mite': (
        'Treat with neem oil or insecticidal soap, and abamectin for severe cases. Prevent by maintaining humidity and washing leaves.'
    ),
    'Tomato__Target_Spot': (
        'Spray Azoxystrobin or Chlorothalonil to control target spot.'
    ),
    'Tomato__Tomato_YellowLeaf__Curl_Virus': (
        'Viral disease spread by whiteflies. Remove infected plants and control whiteflies with neem oil or imidacloprid.'
    ),
    'Tomato__Tomato_mosaic_virus': (
        'Viral disease with no cure. Remove infected plants, disinfect tools, and avoid tobacco contact.'
    ),
    'Tomato_healthy': (
        'Plant is good. Keep balanced nutrients, proper spacing, and regular pest monitoring.'
    ),
    'PlantVillage': (
        'This label is a dataset category, not a plant disease. Please use a real crop image for diagnosis.'
    ),
}

MEDICINE_MAP = {
    'Pepper__bell___Bacterial_spot': {
        'text': 'Copper-based fungicide is recommended. Example: Casa De Amor Copper Sulphate Fungicide.',
        'button_text': 'Buy Copper Fungicide',
        'button_url': 'https://www.example.com/casa-de-amor-copper-sulphate'
    },
    'Pepper__bell___healthy': {
        'text': 'No medicine needed. Use preventive sprays like Kocide or neem oil for protection.',
        'button_text': 'Preventive Spray Guide',
        'button_url': 'https://www.example.com/preventive-sprays'
    },
    'Potato___Early_blight': {
        'text': 'Mancozeb fungicide is recommended. Example product: POMAIS Mancozeb.',
        'button_text': 'Buy Mancozeb',
        'button_url': 'https://www.pomais.com/product/mancozeb/?utm_source=chatgpt.com'
    },
    'Potato___Late_blight': {
        'text': 'Metalaxyl + Mancozeb combination fungicide is recommended. Example product: Metalaxyl Mancozeb WP.',
        'button_text': 'Buy Metalaxyl + Mancozeb',
        'button_url': 'https://linuxcrop.com/public/shop/metalaxy-mancozeb-wp?utm_source=chatgpt.com'
    },
    'Potato___healthy': {
        'text': 'No medicine needed. Use preventive sprays like Kocide or neem oil for plant health.',
        'button_text': 'Preventive Spray Guide',
        'button_url': 'https://www.example.com/preventive-sprays'
    },
    'Tomato_Bacterial_spot': {
        'text': 'Copper fungicide is recommended. Example products: Hicopper or Gozaru Copper Oxychloride.',
        'button_text': 'Buy Copper Fungicide',
        'button_url': 'https://www.example.com/copper-oxychloride'
    },
    'Tomato_Early_blight': {
        'text': 'Mancozeb or Chlorothalonil fungicide is recommended.',
        'button_text': 'Buy Mancozeb',
        'button_url': 'https://www.pomais.com/product/mancozeb/?utm_source=chatgpt.com'
    },
    'Tomato_Late_blight': {
        'text': 'Metalaxyl-based fungicide is recommended. Example product: Metalaxyl + Chlorothalonil.',
        'button_text': 'Buy Metalaxyl Fungicide',
        'button_url': 'https://linuxcrop.com/public/shop/metalaxy-mancozeb-wp?utm_source=chatgpt.com'
    },
    'Tomato_Leaf_Mold': {
        'text': 'Copper fungicide is recommended. Example products: Hicopper or Gozaru Copper Oxychloride.',
        'button_text': 'Buy Copper Fungicide',
        'button_url': 'https://www.example.com/copper-oxychloride'
    },
    'Tomato_Septoria_leaf_spot': {
        'text': 'Chlorothalonil or Mancozeb fungicide is recommended.',
        'button_text': 'Buy Mancozeb',
        'button_url': 'https://www.pomais.com/product/mancozeb/?utm_source=chatgpt.com'
    },
    'Tomato_Spider_mites_Two_spotted_spider_mite': {
        'text': 'Neem oil or organic pesticide is recommended. Example product: Garden Genie Neem Oil Spray.',
        'button_text': 'Buy Neem Oil',
        'button_url': 'https://www.example.com/neem-oil-spray'
    },
    'Tomato__Target_Spot': {
        'text': 'Mancozeb or Chlorothalonil fungicide is recommended.',
        'button_text': 'Buy Mancozeb',
        'button_url': 'https://www.pomais.com/product/mancozeb/?utm_source=chatgpt.com'
    },
    'Tomato__Tomato_YellowLeaf__Curl_Virus': {
        'text': 'No cure. Control whiteflies with neem oil or imidacloprid.',
        'button_text': 'Whitefly Control',
        'button_url': 'https://www.example.com/whitefly-control'
    },
    'Tomato__Tomato_mosaic_virus': {
        'text': 'No cure. Remove infected plants and control vectors with neem oil.',
        'button_text': 'Whitefly Control',
        'button_url': 'https://www.example.com/whitefly-control'
    },
    'Tomato_healthy': {
        'text': 'No medicine needed. Use preventive sprays like Kocide or neem oil for protection.',
        'button_text': 'Preventive Spray Guide',
        'button_url': 'https://www.example.com/preventive-sprays'
    },
    'PlantVillage': {
        'text': 'This is a dataset category. No treatment applies.',
        'button_text': 'Learn More',
        'button_url': 'https://www.example.com/dataset-info'
    },
}

print("\n" + "="*70)
print("APP READY FOR INFERENCE")
print("="*70 + "\n")


def getResult(image_path):
    """Run inference and return probabilities plus model-attention regions."""
    try:
        with Image.open(image_path) as image:
            image_tensor = inference_transform(image.convert('RGB')).unsqueeze(0)
        image_tensor = image_tensor.to(device)

        captured = {}

        def capture_features(module, inputs, output):
            captured['features'] = output
            output.retain_grad()

        hook = model.features[-1].register_forward_hook(capture_features)
        try:
            model.zero_grad(set_to_none=True)
            logits = model(image_tensor)
            probabilities = torch.softmax(logits / confidence_temperature, dim=1)[0]
            predicted_index = int(probabilities.argmax().item())
            logits[0, predicted_index].backward()

            features = captured['features'][0]
            gradients = captured['features'].grad[0]
            channel_weights = gradients.mean(dim=(1, 2), keepdim=True)
            heatmap = torch.relu((channel_weights * features).sum(dim=0))
            heatmap -= heatmap.min()
            heatmap /= heatmap.max().clamp_min(1e-8)
            heatmap = torch.nn.functional.interpolate(
                heatmap[None, None],
                size=(10, 10),
                mode='bilinear',
                align_corners=False,
            )[0, 0]
            attention_regions = heatmap_to_regions(heatmap.detach().cpu().numpy())
        finally:
            hook.remove()

        return probabilities.detach().cpu().numpy(), attention_regions
    except Exception as e:
        print(f"Error in inference: {e}")
        import traceback
        traceback.print_exc()
        return None, []


INVALID_LABELS = {'PlantVillage'}

REMEDY_LABEL_ALIASES = {
    'Pepper,_bell___Bacterial_spot': 'Pepper__bell___Bacterial_spot',
    'Pepper,_bell___healthy': 'Pepper__bell___healthy',
    'Tomato___Bacterial_spot': 'Tomato_Bacterial_spot',
    'Tomato___Early_blight': 'Tomato_Early_blight',
    'Tomato___Late_blight': 'Tomato_Late_blight',
    'Tomato___Leaf_Mold': 'Tomato_Leaf_Mold',
    'Tomato___Septoria_leaf_spot': 'Tomato_Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato___Target_Spot': 'Tomato__Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato___Tomato_mosaic_virus': 'Tomato__Tomato_mosaic_virus',
    'Tomato___healthy': 'Tomato_healthy',
}


def heatmap_to_regions(heatmap, max_regions=6):
    """Convert the strongest Grad-CAM cells into small display boxes."""
    threshold = max(0.55, float(np.quantile(heatmap, 0.82)))
    candidates = [
        (float(heatmap[row, column]), column, row)
        for row in range(heatmap.shape[0])
        for column in range(heatmap.shape[1])
        if heatmap[row, column] >= threshold
    ]
    candidates.sort(reverse=True)

    selected = []
    for score, column, row in candidates:
        if any(abs(column - old_column) <= 1 and abs(row - old_row) <= 1
               for _, old_column, old_row in selected):
            continue
        selected.append((score, column, row))
        if len(selected) >= max_regions:
            break

    return [
        {
            'x': min(column * 10, 80),
            'y': min(row * 10, 80),
            'width': 20,
            'height': 20,
            'attention': round(score * 100, 1),
        }
        for score, column, row in selected
    ]


def get_top_predictions(predictions, k=3, invalid_labels=None):
    """Return top k predictions with confidence scores, excluding invalid labels."""
    if predictions is None:
        return []
    if invalid_labels is None:
        invalid_labels = set()

    scored = [
        (idx, float(predictions[idx]) * 100)
        for idx in range(len(predictions))
        if labels.get(idx) not in invalid_labels
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in scored[:k]:
        results.append({
            'class': labels.get(idx, 'Unknown'),
            'confidence': score
        })
    return results

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        f = request.files['file']
        if f.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        try:
            file_path = os.path.join(UPLOAD_DIR, secure_filename(f.filename))
            f.save(file_path)
            
            predictions, attention_regions = getResult(file_path)
            if predictions is None:
                return jsonify({'error': 'Failed to process image'}), 500
            
            # Get top 3 predictions excluding PlantVillage
            top_preds = get_top_predictions(predictions, k=3, invalid_labels=INVALID_LABELS)
            
            # Choose highest valid prediction by excluding PlantVillage
            valid_scores = [
                (idx, float(predictions[idx]) * 100)
                for idx in range(len(predictions))
                if labels.get(idx) not in INVALID_LABELS
            ]
            if valid_scores:
                valid_scores.sort(key=lambda x: x[1], reverse=True)
                best_idx, best_conf = valid_scores[0]
                predicted_label = labels[best_idx]
                confidence = best_conf
            else:
                best_idx = np.argmax(predictions)
                predicted_label = labels[best_idx]
                confidence = float(predictions[best_idx]) * 100

            suspect_regions = [] if 'healthy' in predicted_label.lower() else attention_regions
            
            # Clean up uploaded file
            try:
                os.remove(file_path)
            except:
                pass
            
            remedy_label = REMEDY_LABEL_ALIASES.get(predicted_label, predicted_label)
            remedy_text = REMEDY_MAP.get(
                remedy_label,
                'No specific remedy available. Maintain good plant care and monitor closely.'
            )
            medicine = MEDICINE_MAP.get(remedy_label, {
                'text': 'No specific medicine recommendation available. Follow good plant care and consult a local supplier.',
                'button_text': 'View plant care guide',
                'button_url': 'https://www.example.com/preventive-sprays'
            })
            # Return only the primary prediction, confidence, remedy guidance, and medicine recommendation
            return jsonify({
                'primary_prediction': predicted_label,
                'confidence': confidence,
                'remedy': remedy_text,
                'medicine': medicine['text'],
                'medicine_button_text': medicine['button_text'],
                'medicine_button_url': medicine['button_url'],
                'suspect_regions': suspect_regions,
            })
        except Exception as e:
            print(f"Error in prediction: {e}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    return None


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
