"""
=========================================================================
 Artificial Intelligence Model Using Machine Learning
 for Student Performance Prediction
=========================================================================
 This Streamlit application loads a PRE-TRAINED machine learning model
 (model.pkl) and uses it to:
   1) Predict a student's Final Grade (G3) from user input
   2) Show model performance analysis (R2 score, correlation matrix,
      actual vs predicted plot, feature importance) using student.csv

 IMPORTANT:
 - This app does NOT retrain or modify the model in any way.
 - The model is loaded exactly as it was saved from the notebook.
 - The 30 input features are collected from the user, encoded, and
   passed to the model in the EXACT SAME ORDER used during training.
=========================================================================
"""

# -------------------------------------------------------------------
# 1. IMPORT REQUIRED LIBRARIES
# -------------------------------------------------------------------
import streamlit as st                 # Build the web app UI
import numpy as np                     # Numeric feature arrays
import pandas as pd                    # Load & process student.csv
import pickle                          # Load the saved model.pkl file
import os                              # Check if files exist
import matplotlib.pyplot as plt        # Plotting
import seaborn as sns                  # Correlation heatmap styling
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# -------------------------------------------------------------------
# 2. PAGE CONFIGURATION (must be the first Streamlit command)
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# -------------------------------------------------------------------
# 3. CUSTOM CSS FOR A PROFESSIONAL LOOK
# -------------------------------------------------------------------
st.markdown("""
    <style>
        .main { background-color: #f5f7fa; }

        .main-title {
            text-align: center;
            font-size: 2.6rem;
            font-weight: 800;
            color: #d90b45;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            text-align: center;
            font-size: 1.05rem;
            color: #6b7280;
            margin-bottom: 1.8rem;
        }

        .section-header {
            font-size: 1.25rem;
            font-weight: 700;
            color: #ffffff;
            background: linear-gradient(90deg, #4f46e5, #6366f1);
            padding: 10px 16px;
            border-radius: 10px;
            margin-top: 1.2rem;
            margin-bottom: 1rem;
        }

        div.stButton > button {
            width: 100%;
            background: linear-gradient(90deg, #4f46e5, #6366f1);
            color: white;
            font-size: 1.1rem;
            font-weight: 700;
            padding: 0.7rem;
            border-radius: 10px;
            border: none;
            margin-top: 1.5rem;
        }
        div.stButton > button:hover {
            background: linear-gradient(90deg, #4338ca, #4f46e5);
            color: white;
        }

        .result-card {
            padding: 2rem;
            border-radius: 16px;
            text-align: center;
            margin-top: 1.5rem;
            box-shadow: 0 4px 14px rgba(0,0,0,0.12);
        }
        .result-score {
            font-size: 3rem;
            font-weight: 800;
            margin: 0.3rem 0;
        }
        .result-category {
            font-size: 1.4rem;
            font-weight: 700;
            margin-top: 0.3rem;
        }

        .metric-card {
            background: #ffffff;
            border-radius: 14px;
            padding: 1.2rem;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            border: 1px solid #e5e7eb;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: #4f46e5;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #6b7280;
            font-weight: 600;
        }

        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# 4. LOAD THE PRE-TRAINED MODEL (model.pkl)
# -------------------------------------------------------------------
MODEL_PATH = "model.pkl"
DATA_PATH = "student.csv"

@st.cache_resource
def load_model():
    """
    Loads the trained model from model.pkl.
    Cached so the model is loaded only ONCE.

    Some notebooks save models using joblib (common with sklearn)
    instead of plain pickle. joblib-saved files can fail with errors
    like "STACK_GLOBAL requires str" if opened with plain pickle.load(),
    so we try joblib first and fall back to pickle.
    """
    if not os.path.exists(MODEL_PATH):
        return None

    try:
        import joblib
        return joblib.load(MODEL_PATH)
    except Exception:
        pass

    with open(MODEL_PATH, "rb") as file:
        loaded_model = pickle.load(file)
    return loaded_model


model = load_model()

if model is None:
    st.error(
        f"⚠️ Model file '{MODEL_PATH}' was not found. "
        "Please place your trained model.pkl file in the same folder as app.py."
    )
    st.stop()


# -------------------------------------------------------------------
# 5. TITLE SECTION
# -------------------------------------------------------------------
st.markdown('<div class="main-title">🎓 Student Performance Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Artificial Intelligence Model Using Machine Learning '
    'for Student Performance Prediction</div>',
    unsafe_allow_html=True
)


# -------------------------------------------------------------------
# 6. ENCODING MAPS FOR CATEGORICAL FEATURES
# -------------------------------------------------------------------
# These mappings convert text categories into numbers, exactly the way
# scikit-learn's LabelEncoder encodes values (alphabetical order = 0,1,2...).
# If your notebook used a DIFFERENT encoding order, update these
# dictionaries so they match your training encoding exactly.
# -------------------------------------------------------------------

binary_yes_no = {"no": 0, "yes": 1}  # alphabetical: no=0, yes=1

encoding_maps = {
    "school":     {"GP": 0, "MS": 1},
    "sex":        {"F": 0, "M": 1},
    "address":    {"R": 0, "U": 1},
    "famsize":    {"GT3": 0, "LE3": 1},
    "Pstatus":    {"A": 0, "T": 1},
    "Mjob":       {"at_home": 0, "health": 1, "other": 2, "services": 3, "teacher": 4},
    "Fjob":       {"at_home": 0, "health": 1, "other": 2, "services": 3, "teacher": 4},
    "reason":     {"course": 0, "home": 1, "other": 2, "reputation": 3},
    "guardian":   {"father": 0, "mother": 1, "other": 2},
    "schoolsup":  binary_yes_no,
    "famsup":     binary_yes_no,
    "paid":       binary_yes_no,
    "activities": binary_yes_no,
    "nursery":    binary_yes_no,
    "higher":     binary_yes_no,
    "internet":   binary_yes_no,
    "romantic":   binary_yes_no,
}

# The EXACT feature order the model was trained on. DO NOT CHANGE.
FEATURE_ORDER = [
    'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
    'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
    'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
    'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout',
    'Dalc', 'Walc', 'health', 'absences'
]


def encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the same encoding_maps used for single predictions to an
    entire DataFrame, so the dataset can be run through the model for
    performance analysis (R2 score, correlation matrix, etc).
    """
    encoded = df.copy()
    for col, mapping in encoding_maps.items():
        if col in encoded.columns:
            encoded[col] = encoded[col].map(mapping)
    return encoded


# -------------------------------------------------------------------
# 7. TABS: PREDICT  |  MODEL ANALYSIS
# -------------------------------------------------------------------
tab1, tab2 = st.tabs(["🔮 Predict", "📊 Model Analysis"])


# =====================================================================
# TAB 1 — PREDICTION FORM
# =====================================================================
with tab1:

    user_input = {}

    with st.form("student_form"):

        # ---------------- STUDENT INFORMATION ----------------
        st.markdown('<div class="section-header">👤 Student Information</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            user_input["school"] = st.selectbox("School", ["GP", "MS"],
                                                 help="GP = Gabriel Pereira, MS = Mousinho da Silveira")
            user_input["sex"] = st.selectbox("Sex", ["F", "M"])
        with c2:
            user_input["age"] = st.number_input("Age", min_value=15, max_value=22, value=17, step=1)
            user_input["address"] = st.selectbox("Home Address Type", ["U", "R"],
                                                  help="U = Urban, R = Rural")
        with c3:
            user_input["famsize"] = st.selectbox("Family Size", ["LE3", "GT3"],
                                                  help="LE3 = <=3, GT3 = >3")
            user_input["Pstatus"] = st.selectbox("Parent's Cohabitation Status", ["T", "A"],
                                                  help="T = Living Together, A = Apart")

        # ---------------- FAMILY INFORMATION ----------------
        st.markdown('<div class="section-header">👨‍👩‍👧 Family Information</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            user_input["Medu"] = st.selectbox(
                "Mother's Education", [0, 1, 2, 3, 4],
                format_func=lambda x: ["None", "Primary (4th)", "5th-9th grade", "Secondary", "Higher"][x]
            )
            user_input["Mjob"] = st.selectbox("Mother's Job", ["at_home", "health", "other", "services", "teacher"])
        with c2:
            user_input["Fedu"] = st.selectbox(
                "Father's Education", [0, 1, 2, 3, 4],
                format_func=lambda x: ["None", "Primary (4th)", "5th-9th grade", "Secondary", "Higher"][x]
            )
            user_input["Fjob"] = st.selectbox("Father's Job", ["at_home", "health", "other", "services", "teacher"])
        with c3:
            user_input["guardian"] = st.selectbox("Guardian", ["mother", "father", "other"])
            user_input["famrel"] = st.slider("Quality of Family Relationship", 1, 5, 4,
                                              help="1 = Very Bad, 5 = Excellent")

        # ---------------- ACADEMIC INFORMATION ----------------
        st.markdown('<div class="section-header">📚 Academic Information</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            user_input["reason"] = st.selectbox("Reason to Choose School",
                                                 ["course", "home", "other", "reputation"])
            user_input["traveltime"] = st.selectbox(
                "Home to School Travel Time", [1, 2, 3, 4],
                format_func=lambda x: ["<15 min", "15-30 min", "30-60 min", ">60 min"][x - 1]
            )
        with c2:
            user_input["studytime"] = st.selectbox(
                "Weekly Study Time", [1, 2, 3, 4],
                format_func=lambda x: ["<2 hrs", "2-5 hrs", "5-10 hrs", ">10 hrs"][x - 1]
            )
            user_input["failures"] = st.selectbox("Past Class Failures", [0, 1, 2, 3])
        with c3:
            user_input["schoolsup"] = st.selectbox("Extra Educational Support", ["yes", "no"])
            user_input["famsup"] = st.selectbox("Family Educational Support", ["yes", "no"])

        c1, c2, c3 = st.columns(3)
        with c1:
            user_input["paid"] = st.selectbox("Extra Paid Classes", ["yes", "no"])
        with c2:
            user_input["activities"] = st.selectbox("Extra-Curricular Activities", ["yes", "no"])
        with c3:
            user_input["higher"] = st.selectbox("Wants Higher Education", ["yes", "no"])

        c1, c2 = st.columns(2)
        with c1:
            user_input["nursery"] = st.selectbox("Attended Nursery School", ["yes", "no"])
        with c2:
            user_input["absences"] = st.number_input("Number of School Absences",
                                                       min_value=0, max_value=100, value=0, step=1)

        # ---------------- LIFESTYLE ----------------
        st.markdown('<div class="section-header">🌿 Lifestyle Information</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            user_input["internet"] = st.selectbox("Internet Access at Home", ["yes", "no"])
            user_input["romantic"] = st.selectbox("In a Romantic Relationship", ["yes", "no"])
        with c2:
            user_input["freetime"] = st.slider("Free Time After School", 1, 5, 3,
                                                help="1 = Very Low, 5 = Very High")
            user_input["goout"] = st.slider("Going Out with Friends", 1, 5, 3,
                                             help="1 = Very Low, 5 = Very High")
        with c3:
            user_input["Dalc"] = st.slider("Workday Alcohol Consumption", 1, 5, 1,
                                            help="1 = Very Low, 5 = Very High")
            user_input["Walc"] = st.slider("Weekend Alcohol Consumption", 1, 5, 1,
                                            help="1 = Very Low, 5 = Very High")

        user_input["health"] = st.slider("Current Health Status", 1, 5, 3,
                                          help="1 = Very Bad, 5 = Very Good")

        # ---------------- SUBMIT BUTTON ----------------
        submitted = st.form_submit_button("🔮 Predict Final Grade")


    # -------------------------------------------------------------------
    # INPUT VALIDATION + PREDICTION LOGIC
    # -------------------------------------------------------------------
    if submitted:

        errors = []
        if user_input["age"] < 15 or user_input["age"] > 22:
            errors.append("Age must be between 15 and 22.")
        if user_input["absences"] < 0:
            errors.append("Absences cannot be negative.")

        if errors:
            for err in errors:
                st.error(f"❌ {err}")
            st.stop()

        try:
            feature_values = []
            for feature_name in FEATURE_ORDER:
                raw_value = user_input[feature_name]
                if feature_name in encoding_maps:
                    feature_values.append(encoding_maps[feature_name][raw_value])
                else:
                    feature_values.append(raw_value)

            final_input = np.array(feature_values).reshape(1, -1)
            prediction = model.predict(final_input)
            predicted_g3 = float(prediction[0])
            predicted_g3_display = max(0, min(20, predicted_g3))

            if predicted_g3_display >= 16:
                category, color, emoji = "Excellent", "#16a34a", "🌟"
            elif predicted_g3_display >= 12:
                category, color, emoji = "Good", "#2563eb", "👍"
            elif predicted_g3_display >= 8:
                category, color, emoji = "Average", "#f59e0b", "📘"
            else:
                category, color, emoji = "Needs Improvement", "#dc2626", "⚠️"

            st.markdown(f"""
                <div class="result-card" style="background-color:{color}15; border: 2px solid {color};">
                    <div style="font-size:2.5rem;">{emoji}</div>
                    <div style="font-size:1.1rem; color:#374151; font-weight:600;">
                        Predicted Final Grade (G3)
                    </div>
                    <div class="result-score" style="color:{color};">
                        {predicted_g3_display:.2f} / 20
                    </div>
                    <div class="result-category" style="color:{color};">
                        {category}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.progress(min(1.0, max(0.0, predicted_g3_display / 20)))

        except KeyError as e:
            st.error(f"❌ Encoding error: missing mapping for value {e}. Please check your inputs.")
        except Exception as e:
            st.error(f"❌ An error occurred while making the prediction: {e}")


# =====================================================================
# TAB 2 — MODEL PERFORMANCE ANALYSIS (R2 score, correlation, plots)
# =====================================================================
with tab2:

    st.markdown('<div class="section-header">📊 Model Performance & Data Analysis</div>', unsafe_allow_html=True)

    if not os.path.exists(DATA_PATH):
        st.warning(
            f"⚠️ '{DATA_PATH}' not found in the project folder. "
            "Place your student.csv here to see R2 score, correlation matrix, "
            "and other analysis charts."
        )
    else:
        @st.cache_data
        def load_dataset():
            """Loads student.csv once and caches it."""
            # The UCI dataset commonly uses ';' as separator — this handles
            # both ',' and ';' automatically.
            return pd.read_csv(DATA_PATH, sep=None, engine="python")

        raw_df = load_dataset()

        # Basic sanity check — target column must exist
        if "G3" not in raw_df.columns:
            st.error("❌ 'G3' column not found in student.csv. Cannot run analysis.")
        else:
            # -----------------------------------------------------------
            # Prepare data: drop G1, G2 (same as training), encode
            # categorical columns, and align feature order with the model.
            # -----------------------------------------------------------
            df = raw_df.copy()
            df = df.drop(columns=[c for c in ["G1", "G2"] if c in df.columns])

            encoded_df = encode_dataframe(df)

            missing_cols = [c for c in FEATURE_ORDER if c not in encoded_df.columns]
            if missing_cols:
                st.error(f"❌ student.csv is missing required columns: {missing_cols}")
            else:
                X = encoded_df[FEATURE_ORDER]
                y_actual = encoded_df["G3"]

                # Drop rows with any missing/unmapped values so the model
                # doesn't break on NaNs from unexpected category values.
                valid_mask = X.notna().all(axis=1) & y_actual.notna()
                X_valid = X[valid_mask]
                y_valid = y_actual[valid_mask]

                try:
                    y_pred = model.predict(X_valid)

                    r2 = r2_score(y_valid, y_pred)
                    mae = mean_absolute_error(y_valid, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

                    # ---------------------------------------------------
                    # METRIC CARDS
                    # ---------------------------------------------------
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{r2:.3f}</div>
                                <div class="metric-label">R² Score</div>
                            </div>""", unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{mae:.2f}</div>
                                <div class="metric-label">Mean Absolute Error</div>
                            </div>""", unsafe_allow_html=True)
                    with m3:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{rmse:.2f}</div>
                                <div class="metric-label">RMSE</div>
                            </div>""", unsafe_allow_html=True)
                    with m4:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{len(X_valid)}</div>
                                <div class="metric-label">Records Evaluated</div>
                            </div>""", unsafe_allow_html=True)

                    st.caption(
                        "R² Score shows how well the model's predictions match actual "
                        "G3 grades in the dataset (closer to 1.0 = better fit). "
                        "These metrics are computed by running the FULL student.csv "
                        "through your existing model.pkl — the model itself is not retrained."
                    )

                    st.markdown("---")

                    # ---------------------------------------------------
                    # ACTUAL vs PREDICTED SCATTER PLOT
                    # ---------------------------------------------------
                    col_left, col_right = st.columns(2)

                    with col_left:
                        st.markdown("#### 🎯 Actual vs Predicted G3")
                        fig1, ax1 = plt.subplots(figsize=(5, 4.2))
                        ax1.scatter(y_valid, y_pred, alpha=0.5, color="#4f46e5", edgecolor="white", s=40)
                        min_val = min(y_valid.min(), y_pred.min())
                        max_val = max(y_valid.max(), y_pred.max())
                        ax1.plot([min_val, max_val], [min_val, max_val], color="#dc2626",
                                 linestyle="--", linewidth=1.5, label="Perfect Prediction")
                        ax1.set_xlabel("Actual G3")
                        ax1.set_ylabel("Predicted G3")
                        ax1.set_title("Actual vs Predicted Final Grade")
                        ax1.legend()
                        fig1.tight_layout()
                        st.pyplot(fig1)

                    # ---------------------------------------------------
                    # RESIDUAL DISTRIBUTION PLOT
                    # ---------------------------------------------------
                    with col_right:
                        st.markdown("#### 📉 Prediction Error Distribution")
                        residuals = y_valid.values - y_pred
                        fig2, ax2 = plt.subplots(figsize=(5, 4.2))
                        sns.histplot(residuals, kde=True, color="#6366f1", ax=ax2)
                        ax2.axvline(0, color="#dc2626", linestyle="--", linewidth=1.5)
                        ax2.set_xlabel("Residual (Actual - Predicted)")
                        ax2.set_title("Distribution of Prediction Errors")
                        fig2.tight_layout()
                        st.pyplot(fig2)

                except Exception as e:
                    st.error(f"❌ Could not compute predictions for analysis: {e}")

                st.markdown("---")

                # -------------------------------------------------------
                # CORRELATION MATRIX HEATMAP
                # -------------------------------------------------------
                st.markdown("#### 🔗 Correlation Matrix (Features vs G3)")

                corr_df = encoded_df[FEATURE_ORDER + ["G3"]].dropna()
                corr_matrix = corr_df.corr()

                fig3, ax3 = plt.subplots(figsize=(10, 4))
                sns.heatmap(
                    corr_matrix,
                    annot=False,
                    cmap="coolwarm",
                    center=0,
                    linewidths=0.4,
                    ax=ax3,
                    cbar_kws={"shrink": 0.7}
                )
                ax3.set_title("Correlation Matrix — All Features vs Final Grade (G3)")
                fig3.tight_layout()
                st.pyplot(fig3)

                # -------------------------------------------------------
                # TOP CORRELATIONS WITH G3 (bar chart)
                # -------------------------------------------------------
                st.markdown("#### 📈 Top Features Correlated with G3")

                g3_corr = corr_matrix["G3"].drop("G3").sort_values(key=abs, ascending=False).head(10)

                fig4, ax4 = plt.subplots(figsize=(6, 2.5))
                colors = ["#16a34a" if v > 0 else "#dc2626" for v in g3_corr.values]
                ax4.barh(g3_corr.index[::-1], g3_corr.values[::-1], color=colors[::-1])
                ax4.set_xlabel("Correlation with G3")
                ax4.set_title("Top 10 Features Correlated with Final Grade")
                ax4.axvline(0, color="#374151", linewidth=0.8)
                fig4.tight_layout()
                st.pyplot(fig4)

                # -------------------------------------------------------
                # FEATURE IMPORTANCE (only if the model supports it)
                # -------------------------------------------------------
                if hasattr(model, "feature_importances_"):
                    st.markdown("---")
                    st.markdown("#### 🌳 Feature Importance (from the trained model)")

                    importances = pd.Series(model.feature_importances_, index=FEATURE_ORDER)
                    importances = importances.sort_values(ascending=False).head(15)

                    fig5, ax5 = plt.subplots(figsize=(8,4))
                    ax5.barh(importances.index[::-1], importances.values[::-1], color="#4f46e5")
                    ax5.set_xlabel("Importance")
                    ax5.set_title("Top 15 Most Important Features")
                    fig5.tight_layout()
                    st.pyplot(fig5)


# -------------------------------------------------------------------
# 8. FOOTER
# -------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#9ca3af; font-size:0.85rem;'>"
    "</div>",
    unsafe_allow_html=True
)