"""
Pakistani Politician Image Classifier — Flask App
Run: python app.py
Then open: http://localhost:5000
"""

import io
import json
import base64
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

from flask import Flask, request, jsonify, render_template_string

warnings.filterwarnings('ignore')

# ─── Config ────────────────────────────────────────────────────────────────────

NUM_CLASSES    = 16
IMG_SIZE       = 224
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Update this path to wherever you placed politician_model.pth
MODEL_PATH = Path('politician_model.pth')

CLASS_LABELS = [
    'ahsan_iqbal', 'asif_zardari', 'benazir_bhutto', 'bilawal_bhutto',
    'hamza_shehbaz', 'imran_khan', 'ishaq_dar', 'khawaja_asif',
    'maryam_nawaz', 'mohsin_naqvi', 'murad_ali_shah', 'nawaz_sharif',
    'pervez_musharraf', 'rana_sanaullah', 'shehbaz_sharif', 'yousef_raza_gillani',
]

DISPLAY_NAMES = {
    'ahsan_iqbal':         'Ahsan Iqbal',
    'asif_zardari':        'Asif Ali Zardari',
    'benazir_bhutto':      'Benazir Bhutto',
    'bilawal_bhutto':      'Bilawal Bhutto Zardari',
    'hamza_shehbaz':       'Hamza Shehbaz',
    'imran_khan':          'Imran Khan',
    'ishaq_dar':           'Ishaq Dar',
    'khawaja_asif':        'Khawaja Asif',
    'maryam_nawaz':        'Maryam Nawaz',
    'mohsin_naqvi':        'Mohsin Naqvi',
    'murad_ali_shah':      'Murad Ali Shah',
    'nawaz_sharif':        'Nawaz Sharif',
    'pervez_musharraf':    'Pervez Musharraf',
    'rana_sanaullah':      'Rana Sanaullah',
    'shehbaz_sharif':      'Shehbaz Sharif',
    'yousef_raza_gillani': 'Yousef Raza Gillani',
}

# ─── Model ─────────────────────────────────────────────────────────────────────

def build_resnet50(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return model


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'Model checkpoint not found at: {path}')
    model = build_resnet50(NUM_CLASSES)
    checkpoint = torch.load(path, map_location=DEVICE)
    # Support both raw state_dict and wrapped checkpoints
    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval().to(DEVICE)
    return model


infer_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Load model once at startup
model = None
model_error = None
try:
    model = load_model(MODEL_PATH)
    print(f'✅ Model loaded from {MODEL_PATH}  |  Device: {DEVICE}')
except Exception as e:
    model_error = str(e)
    print(f'⚠️  Model not loaded: {e}')

# ─── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_image(pil_image: Image.Image):
    if model is None:
        raise RuntimeError(model_error or 'Model not loaded.')

    tensor = infer_transform(pil_image.convert('RGB')).unsqueeze(0).to(DEVICE)
    probs  = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    top5   = np.argsort(-probs)[:5]

    results = [
        {
            'label': CLASS_LABELS[i],
            'name':  DISPLAY_NAMES.get(CLASS_LABELS[i], CLASS_LABELS[i]),
            'prob':  float(probs[i]),
            'pct':   round(float(probs[i]) * 100, 2),
        }
        for i in top5
    ]

    # Build bar chart and encode as base64
    fig, ax = plt.subplots(figsize=(7, 4))
    names  = [r['name'].replace(' ', '\n') for r in reversed(results)]
    values = [r['pct'] for r in reversed(results)]
    colors = ['#2563EB'] + ['#93C5FD'] * 4
    bars   = ax.barh(names, values, color=list(reversed(colors)), edgecolor='white')
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Top-5 Predictions', fontweight='bold')
    ax.set_xlim(0, 100)
    for bar, val in zip(bars, values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    plt.close(fig)
    chart_b64 = base64.b64encode(buf.getvalue()).decode()

    return results, chart_b64


# ─── Flask App ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Pakistani Politician Classifier</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
      min-height: 100vh;
      color: #f1f5f9;
      padding: 2rem 1rem;
    }

    .container { max-width: 960px; margin: 0 auto; }

    header { text-align: center; margin-bottom: 2.5rem; }
    header h1 { font-size: 2rem; font-weight: 800; color: #60a5fa; letter-spacing: -0.5px; }
    header p  { margin-top: .5rem; color: #94a3b8; font-size: .95rem; }

    .card {
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 1rem;
      padding: 1.75rem;
      backdrop-filter: blur(8px);
    }

    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
    @media (max-width: 640px) { .grid { grid-template-columns: 1fr; } }

    /* Upload zone */
    .upload-zone {
      border: 2px dashed #3b82f6;
      border-radius: .75rem;
      overflow: hidden;
      text-align: center;
      cursor: pointer;
      transition: background .2s;
      position: relative;
      height: 260px;
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      gap: .75rem;
    }
    .upload-zone:hover { background: rgba(59,130,246,.08); }
    .upload-zone input[type=file] {
      position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%; z-index: 2;
    }
    .upload-zone svg { color: #3b82f6; }
    .upload-zone span { color: #94a3b8; font-size: .9rem; }
    #preview {
      position: absolute; inset: 0;
      width: 100%; height: 100%;
      object-fit: cover;
      border-radius: .65rem;
      display: none;
      z-index: 1;
    }
    .upload-zone.has-image svg,
    .upload-zone.has-image span { display: none; }

    /* Classify button */
    #classifyBtn {
      margin-top: 1rem; width: 100%;
      background: #2563eb; color: #fff;
      border: none; border-radius: .6rem;
      padding: .8rem 1.5rem; font-size: 1rem; font-weight: 600;
      cursor: pointer; transition: background .2s, transform .1s;
      display: flex; align-items: center; justify-content: center; gap: .5rem;
    }
    #classifyBtn:hover:not(:disabled) { background: #1d4ed8; transform: translateY(-1px); }
    #classifyBtn:disabled { opacity: .5; cursor: not-allowed; }

    /* Spinner */
    .spinner {
      width: 18px; height: 18px;
      border: 2px solid rgba(255,255,255,.3);
      border-top-color: #fff;
      border-radius: 50%;
      animation: spin .7s linear infinite;
      display: none;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* Results */
    #results { display: none; }
    .top-prediction {
      background: linear-gradient(135deg, #1e40af, #1d4ed8);
      border-radius: .75rem; padding: 1.25rem; margin-bottom: 1rem;
    }
    .top-prediction .name { font-size: 1.5rem; font-weight: 700; color: #bfdbfe; }
    .top-prediction .conf { font-size: 1rem; color: #93c5fd; margin-top: .25rem; }

    .bar-list { display: flex; flex-direction: column; gap: .6rem; }
    .bar-row { }
    .bar-meta { display: flex; justify-content: space-between; margin-bottom: .2rem; font-size: .85rem; }
    .bar-name { color: #cbd5e1; }
    .bar-pct  { color: #60a5fa; font-weight: 600; }
    .bar-track { background: rgba(255,255,255,.1); border-radius: 999px; height: 8px; }
    .bar-fill  { background: #3b82f6; border-radius: 999px; height: 8px; transition: width .6s ease; }
    .bar-fill.first { background: #2563eb; }

    #chartImg { width: 100%; border-radius: .5rem; margin-top: 1rem; display: none; }

    /* Error banner */
    #errorBanner {
      display: none; background: rgba(220,38,38,.15);
      border: 1px solid #ef4444; border-radius: .6rem;
      padding: .75rem 1rem; color: #fca5a5; margin-top: 1rem; font-size: .9rem;
    }

    .classes-note {
      margin-top: 2rem; text-align: center;
      color: #64748b; font-size: .82rem; line-height: 1.7;
    }
  </style>
</head>
<body>
<div class="container">
  <header>
    <h1>Pakistani Politician Classifier</h1>
    <p>ResNet-50 · 16 Politicians · Deep Learning</p>
  </header>

  <div class="card grid">
    <!-- Left: upload -->
    <div>
      <label class="upload-zone" id="dropzone">
        <input type="file" id="fileInput" accept="image/*"/>
        <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" fill="none"
             viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
          <path stroke-linecap="round" stroke-linejoin="round"
                d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/>
        </svg>
        <span>Click or drag & drop an image</span>
        <img id="preview" alt="preview"/>
      </label>
      <button id="classifyBtn" disabled>
        <div class="spinner" id="spinner"></div>
        <span id="btnText">🔍 Classify</span>
      </button>
      <div id="errorBanner"></div>
    </div>

    <!-- Right: results -->
    <div id="results">
      <div class="top-prediction">
        <div class="name" id="topName">—</div>
        <div class="conf" id="topConf">—</div>
      </div>
      <div class="bar-list" id="barList"></div>
    </div>
  </div>

  <img id="chartImg" alt="confidence chart"/>

  <p class="classes-note">
    <strong>16 classes:</strong>
    Ahsan Iqbal · Asif Zardari · Benazir Bhutto · Bilawal Bhutto · Hamza Shehbaz ·
    Imran Khan · Ishaq Dar · Khawaja Asif · Maryam Nawaz · Mohsin Naqvi ·
    Murad Ali Shah · Nawaz Sharif · Pervez Musharraf · Rana Sanaullah ·
    Shehbaz Sharif · Yousef Raza Gillani
  </p>
</div>

<script>
  const fileInput   = document.getElementById('fileInput');
  const preview     = document.getElementById('preview');
  const classifyBtn = document.getElementById('classifyBtn');
  const spinner     = document.getElementById('spinner');
  const btnText     = document.getElementById('btnText');
  const results     = document.getElementById('results');
  const topName     = document.getElementById('topName');
  const topConf     = document.getElementById('topConf');
  const barList     = document.getElementById('barList');
  const chartImg    = document.getElementById('chartImg');
  const errorBanner = document.getElementById('errorBanner');

  let selectedFile = null;

  fileInput.addEventListener('change', e => {
    selectedFile = e.target.files[0];
    if (!selectedFile) return;
    preview.src = URL.createObjectURL(selectedFile);
    preview.style.display = 'block';
    document.getElementById('dropzone').classList.add('has-image');
    classifyBtn.disabled  = false;
    results.style.display = 'none';
    chartImg.style.display = 'none';
    errorBanner.style.display = 'none';
  });

  classifyBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    classifyBtn.disabled = true;
    spinner.style.display = 'block';
    btnText.textContent  = 'Classifying…';
    errorBanner.style.display = 'none';

    try {
      const fd = new FormData();
      fd.append('image', selectedFile);

      const res  = await fetch('/predict', { method: 'POST', body: fd });
      const data = await res.json();

      if (!res.ok || data.error) {
        throw new Error(data.error || 'Server error');
      }

      const top = data.results[0];
      topName.textContent = top.name;
      topConf.textContent = `Confidence: ${top.pct.toFixed(2)}%`;

      barList.innerHTML = '';
      data.results.forEach((r, i) => {
        barList.innerHTML += `
          <div class="bar-row">
            <div class="bar-meta">
              <span class="bar-name">${r.name}</span>
              <span class="bar-pct">${r.pct.toFixed(1)}%</span>
            </div>
            <div class="bar-track">
              <div class="bar-fill ${i === 0 ? 'first' : ''}" style="width:${r.pct}%"></div>
            </div>
          </div>`;
      });

      results.style.display = 'block';

      if (data.chart) {
        chartImg.src = 'data:image/png;base64,' + data.chart;
        chartImg.style.display = 'block';
      }

    } catch (err) {
      errorBanner.textContent  = '⚠️ ' + err.message;
      errorBanner.style.display = 'block';
    } finally {
      classifyBtn.disabled = false;
      spinner.style.display = 'none';
      btnText.textContent  = '🔍 Classify';
    }
  });
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    try:
        img = Image.open(io.BytesIO(file.read()))
        results, chart_b64 = predict_image(img)
        return jsonify({'results': results, 'chart': chart_b64})
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok' if model else 'model_missing',
        'device': str(DEVICE),
        'model_path': str(MODEL_PATH),
        'model_loaded': model is not None,
    })


# ─── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f'\n Pakistani Politician Classifier')
    print(f'   Model  : {MODEL_PATH}  ({"loaded ✅" if model else "MISSING ⚠️"})')
    print(f'   Device : {DEVICE}')
    print(f'   URL    : http://localhost:5000\n')
    app.run(host='0.0.0.0', port=5000, debug=False)
