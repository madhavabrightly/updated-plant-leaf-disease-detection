# Plant Disease Classification

A PyTorch plant-leaf disease classifier trained on the PlantVillage dataset.
The project uses EfficientNet-B0 to classify 38 crop and disease categories,
serves predictions through Flask, and displays Grad-CAM attention regions that
show which parts of an image most influenced a prediction.

## Current Model

- Framework: PyTorch
- Architecture: EfficientNet-B0 with ImageNet pretrained weights
- Input size: 224 x 224 RGB
- Classes: 38 PlantVillage categories
- Dataset: `abdallahalidev/plantvillage-dataset` from Kaggle
- Saved model: `model.pt`
- Training backend: AMD ROCm, NVIDIA CUDA, or CPU
- Latest untouched-test accuracy: approximately 99.64%
- Confidence calibration: temperature scaling stored in `model.pt`

PlantVillage contains mostly controlled leaf photographs. Accuracy on phone
photos, complex backgrounds, unusual lighting, or diseases outside the trained
classes may be lower than the reported test accuracy.

## Features

- Automatic PlantVillage download and caching through KaggleHub
- Automatic discovery of the dataset's correct `color` class directory
- Two-phase transfer learning:
  - Phase 1 trains the classification head
  - Phase 2 fine-tunes later EfficientNet feature blocks
- Separate training, validation, and untouched test splits
- Class balancing, label smoothing, early stopping, and mixed precision
- Confidence calibration for more meaningful prediction percentages
- Flask interface with remedy and product guidance
- Grad-CAM attention boxes for model explainability

Grad-CAM boxes are not exact lesion detections. Precise disease localization
requires a separately annotated detection or segmentation dataset.

## Project Files

```text
app.py                     Flask server and PyTorch inference
train.py                   Standalone PyTorch training pipeline
model.pt                   Final calibrated model checkpoint
best_phase1.pt             Best phase-1 checkpoint
best_phase2.pt             Best phase-2 checkpoint
classes.json               Class names in model-output order
classification_report.txt  Untouched-test precision, recall, and F1 results
requirements.txt           Local application dependencies
templates/                 Flask HTML templates
static/                    Frontend CSS and JavaScript
testimages/                Example leaf images
```

The old `model.h5` file is a legacy TensorFlow model and is not used by the
current application.

## Run The Application

Create and activate a Python environment, then install the local dependencies:

```bash
pip install -r requirements.txt
```

Start the Flask server:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

The app automatically loads `model.pt` and verifies that `classes.json`
matches the class order embedded in the checkpoint.

## Deploy On Render

The repository includes `render.yaml` for a production deployment.

1. Sign in to [Render](https://render.com/).
2. Choose **New > Blueprint**.
3. Connect this GitHub repository:
   `madhavabrightly/updated-plant-leaf-disease-detection`
4. Select the repository and apply the blueprint.
5. Wait for the build and deployment to finish.

Render installs `requirements.txt` and starts the app with:

```bash
gunicorn app:app --workers 1 --threads 2 --timeout 120
```

The free Render service sleeps after inactivity, so its first request after
sleeping can take about one minute. A paid instance avoids this cold start.

## Prediction API

Send an image to the `/predict` endpoint:

```bash
curl -F "file=@/path/to/leaf.jpg" http://127.0.0.1:5000/predict
```

Example response:

```json
{
  "primary_prediction": "Tomato___Late_blight",
  "confidence": 98.42,
  "remedy": "Treatment guidance...",
  "medicine": "Recommended product guidance...",
  "suspect_regions": [
    {
      "x": 40,
      "y": 30,
      "width": 20,
      "height": 20,
      "attention": 91.2
    }
  ]
}
```

## Train On AMD ROCm Cloud

Use a cloud image that already includes a ROCm-compatible PyTorch and
Torchvision installation. Do not replace its PyTorch packages with generic
CPU wheels.

Confirm that PyTorch can access the AMD GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

Upload `train.py`, then run:

```bash
python train.py \
  --require-gpu \
  --epochs1 5 \
  --epochs2 15 \
  --batch-size 64 \
  --label-smoothing 0.1
```

PyTorch uses the `cuda` device API for both NVIDIA CUDA and AMD ROCm. Seeing
`torch.cuda.is_available() == True` on an AMD ROCm system is expected.

The script downloads and caches the approximately 2 GB PlantVillage dataset.
Later training runs reuse the cache.

If batch size 64 causes an out-of-memory error, use:

```bash
python train.py --require-gpu --epochs1 5 --epochs2 15 --batch-size 32
```

## Train With A Local Dataset

The supplied path must contain one folder per class, with images directly
inside each class folder:

```text
Dataset/
  Tomato___healthy/
    image1.jpg
  Tomato___Late_blight/
    image2.jpg
  Potato___Early_blight/
    image3.jpg
```

Run:

```bash
python train.py --data Dataset --epochs1 5 --epochs2 15 --batch-size 32
```

## Training Outputs

Training creates:

- `model.pt`: final model used by `app.py`
- `classes.json`: required class-name mapping
- `classification_report.txt`: untouched-test metrics for every class
- `best_phase1.pt`: best classification-head checkpoint
- `best_phase2.pt`: best fine-tuning checkpoint

At minimum, copy `model.pt` and `classes.json` into the application folder.
Restart `app.py` after replacing the model so the new checkpoint is loaded.

## Important Accuracy Notes

The test split is separate from training and validation, but all images still
come from PlantVillage. For dependable real-world use:

1. Test with phone photos that were not sourced from PlantVillage.
2. Collect examples with natural backgrounds, shadows, blur, and early disease.
3. Add an unknown or unsupported-image rejection workflow.
4. Consult agricultural professionals before acting on treatment guidance.

The app is an educational decision-support tool, not a guaranteed medical or
agricultural diagnosis.

## Exact Disease Localization

The classifier learns:

```text
image -> disease class
```

It does not learn exact lesion boxes or pixel masks. Grad-CAM only visualizes
regions that influenced the prediction. To train accurate lesion locations,
create a dataset with bounding-box or segmentation-mask annotations and train
a dedicated model such as YOLO detection, YOLO segmentation, or U-Net.

## Common Problems

### `TensorFlow is not installed`

The current project does not use TensorFlow. Use the PyTorch `train.py` and
`model.pt`.

### `No PyTorch GPU is visible`

Select a ROCm-compatible AMD image or CUDA-compatible NVIDIA image. Verify the
GPU with the command in the AMD ROCm section.

### `torch.linalg.lstsq ... requires LAPACK`

Upload the latest `train.py`. It uses `RandomAffine` instead of the
LAPACK-dependent `RandomPerspective` transform.

### App still uses an old model

Stop and restart the Flask server after replacing `model.pt`.

### Prediction confidence is unexpectedly high

Use a model produced by the latest `train.py`. It stores a learned confidence
temperature that `app.py` applies during inference.

## Dataset Attribution

Training data is downloaded through KaggleHub from:

[PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
using the identifier `abdallahalidev/plantvillage-dataset`.

Review the dataset page for its source, citation, and license requirements
before redistribution or commercial use.
