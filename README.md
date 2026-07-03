# 🎓 Student Performance Predictor — Streamlit App

Artificial Intelligence Model Using Machine Learning for Student Performance Prediction

## 📁 Project Files

```
project-folder/
├── app.py              # Streamlit web application (already created)
├── requirements.txt    # Python dependencies (already created)
├── model.pkl           # Your trained model — COPY THIS FROM YOUR COLAB PROJECT
└── student.csv          # Original dataset (optional, not required to run the app)
```

## ⚠️ Important — model.pkl

I don't have access to your Google Colab session, so I can't generate
`model.pkl` myself — only you have the trained model file.

**Steps to get it:**
1. In your Colab notebook, after training, run:
   ```python
   import pickle
   with open("model.pkl", "wb") as f:
       pickle.dump(model, f)
   ```
2. Download `model.pkl` from Colab's file panel (left sidebar → Files → right-click → Download).
3. Place it in the **same folder** as `app.py`.

## ▶️ How to Run

1. Put `app.py`, `requirements.txt`, and `model.pkl` in one folder.
2. Open a terminal in that folder.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the app:
   ```
   streamlit run app.py
   ```
5. Your browser will open automatically at `http://localhost:8501`.

## 🔧 If you get an encoding/feature-mismatch error

Open `app.py` and check the `encoding_maps` dictionary near the top
(around line 165). It currently assumes standard alphabetical
`LabelEncoder` order (e.g. `no=0, yes=1`). If your notebook encoded
categorical columns differently, update the numbers in that dictionary
to match your notebook exactly — nothing else needs to change.

## ✅ Notes

- The model itself is never modified or retrained by this app.
- Feature order sent to the model exactly matches your specified order (30 features, G1/G2 excluded).
- `student.csv` is included for reference only; the app does not read it at runtime.
