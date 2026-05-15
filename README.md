#  Pakistani Politician Image Classifier

A deep learning web application that identifies **16 prominent Pakistani politicians** from photos using a fine-tuned **ResNet-50** CNN model, served through a **Flask** frontend.

---

## 📸 Demo

Upload any photo of a Pakistani politician and the app will return:
- The **top predicted politician** with confidence score
- A **Top-5 ranked list** with animated confidence bars
- A **bar chart** visualization of the predictions

---

## 🧠 Model

| Detail | Value |
|---|---|
| Architecture | ResNet-50 (fine-tuned) |
| Input Size | 224 × 224 px |
| Classes | 16 |
| Training Split | 75% train / 15% val / 10% test |
| Optimizer | Adam (lr=1e-4, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Epochs | 30 |
| Augmentations | Random crop, flip, rotation, color jitter |

### 👥 Supported Politicians (16 Classes)

| | | | |
|---|---|---|---|
| Ahsan Iqbal | Asif Ali Zardari | Benazir Bhutto | Bilawal Bhutto Zardari |
| Hamza Shehbaz | Imran Khan | Ishaq Dar | Khawaja Asif |
| Maryam Nawaz | Mohsin Naqvi | Murad Ali Shah | Nawaz Sharif |
| Pervez Musharraf | Rana Sanaullah | Shehbaz Sharif | Yousef Raza Gillani |

---

## 📁 Project Structure

```
project/
├── app.py                  # Flask web application
├── politician_model.pth    # Trained model weights
├── pakistani-politician-image-classification-using-cn.ipynb  # Training notebook
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone / Download the project

Place all files in the same folder:
```
my_project/
├── app.py
└── politician_model.pth
```

### 2. Install dependencies

```bash
pip install flask torch torchvision pillow matplotlib numpy
```

> **Using a virtual environment (recommended):**
> ```bash
> python -m venv venv
> venv\Scripts\activate        # Windows
> source venv/bin/activate     # macOS / Linux
> pip install flask torch torchvision pillow matplotlib numpy
> ```

### 3. Run the app

```bash
python app.py
```

### 4. Open in browser

```
http://localhost:5000
```

---

## 🚀 Usage

1. Open `http://localhost:5000` in your browser
2. Click the upload box (or drag & drop) to select a politician photo
3. Click **🔍 Classify**
4. View the predicted politician, confidence score, and Top-5 chart

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Main web UI |
| `/predict` | POST | Upload image → get predictions |
| `/health` | GET | Check model load status |

### `/predict` — Example with curl

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@photo.jpg"
```

**Response:**
```json
{
  "results": [
    { "label": "imran_khan", "name": "Imran Khan", "prob": 0.9823, "pct": 98.23 },
    { "label": "shehbaz_sharif", "name": "Shehbaz Sharif", "prob": 0.0091, "pct": 0.91 },
    ...
  ],
  "chart": "<base64-encoded PNG>"
}
```

### `/health` — Example response

```json
{
  "status": "ok",
  "device": "cpu",
  "model_path": "politician_model.pth",
  "model_loaded": true
}
```

---

## 🖥️ Hardware

The app automatically uses **GPU (CUDA)** if available, otherwise falls back to **CPU**.

```
✅ Model loaded from politician_model.pth  |  Device: cuda
```

---

## 🗃️ Dataset

Training used the [Pakistani Politicians Images Dataset](https://www.kaggle.com/datasets/abdullahsaood/pakistani-politicians-images-dataset) from Kaggle, containing labelled face images for all 16 classes.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `flask` | Web server & API |
| `torch` | Deep learning inference |
| `torchvision` | ResNet-50 architecture & transforms |
| `pillow` | Image loading & preprocessing |
| `matplotlib` | Confidence bar chart generation |
| `numpy` | Probability array operations |

---

## ❗ Troubleshooting

**`ModuleNotFoundError`** — Install all dependencies:
```bash
pip install flask torch torchvision pillow matplotlib numpy
```

**`Model checkpoint not found`** — Make sure `politician_model.pth` is in the **same folder** as `app.py`.

**Slow on first run** — The model loads once at startup; subsequent predictions are fast.

**Port already in use** — Change the port at the bottom of `app.py`:
```python
app.run(host='0.0.0.0', port=5001, debug=False)
```

---

## 👤 Author

**Abdullah Saood**  
BS Artificial Intelligence — Artificial Neural Networks, Final Project
