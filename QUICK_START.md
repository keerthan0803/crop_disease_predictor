# Quick Start Guide - Crop Disease Predictor

## For Windows Users

### Step 1: Install Dependencies
Open Command Prompt or PowerShell in the project folder and run:
```bash
pip install -r requirements.txt
```

Or simply double-click `setup.bat` which will install everything automatically.

### Step 2: Train the Model
Run the training script (takes 2-4 hours on GPU, 8-12 hours on CPU):
```bash
python train_model.py
```

This generates:
- `crop_disease_model.h5` - The trained model
- `class_names.pkl` - List of all 38 disease classes
- `disease_info.json` - Plant and disease information

### Step 3: Launch the Web App
```bash
python app.py
```

You should see:
```
* Running on http://127.0.0.1:5000
```

### Step 4: Use the Application
1. Open your web browser
2. Go to: `http://localhost:5000`
3. Upload a plant leaf image
4. Click "Predict Disease"
5. View results!

## For GPU Acceleration (Recommended)

Install CUDA-enabled TensorFlow for faster training:
```bash
pip install tensorflow[and-cuda]
```

## Minimal Setup (Advanced)

If you want to use the model without retraining:
1. Train once: `python train_model.py`
2. Keep the 3 generated files (.h5, .pkl, .json)
3. Only run: `python app.py` in the future

## Troubleshooting

**"No module named tensorflow"**
→ Run: `pip install -r requirements.txt`

**"Model not loaded"**
→ Ensure you ran `python train_model.py` first

**Slow training on CPU**
→ Install GPU support or reduce BATCH_SIZE in train_model.py

**Port 5000 already in use**
→ Change port in app.py line: `app.run(debug=True, port=5001)`

## System Requirements

- **Minimum**: Intel i5/Ryzen 5, 8GB RAM
- **Recommended**: NVIDIA GPU (GTX 1060+), 16GB RAM
- **Storage**: 5GB for dataset + models

## What Gets Trained?

The model learns from 38 classes including:
- ✅ Apple (4 varieties)
- ✅ Tomato (7 varieties)
- ✅ Grape (4 varieties)
- ✅ Corn/Maize (4 varieties)
- ✅ Potato (3 varieties)
- ✅ And 9 more plant types!

---

**You're all set! Start with:** `python train_model.py`
