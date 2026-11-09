
import os
import io
import warnings
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import qrcode

from scipy.interpolate import CubicSpline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Thermal Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Color Palette
NAVY = "#102A43"
BLUE = "#1D4E89"
ACCENT = "#168AAD"
CYAN = "#16A6C7"
GREEN = "#159D85"
RED = "#DC5252"
ORANGE = "#D97706"

BACKGROUND = "#F4F7FB"
WHITE = "#FFFFFF"
TEXT = "#243B53"
MUTED = "#718096"
BORDER = "#E1E8F0"


# ==========================================================
# 2. CUSTOM CSS
# ==========================================================

st.markdown(f"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {{
    background-color: {BACKGROUND};
    color: {TEXT};
    font-family: 'Inter', 'Segoe UI', sans-serif;
}}

.block-container {{
    max-width: 1500px;
    padding-top: 1.6rem;
    padding-bottom: 2.5rem;
}}

#MainMenu, footer {{
    visibility: hidden;
}}

[data-testid="stHeader"] {{
    background: transparent;
}}

/* SIDEBAR */

[data-testid="stSidebar"] {{
    background: {NAVY};
    border-right: 1px solid #1D405D;
}}

[data-testid="stSidebar"] * {{
    color: #EAF2F8;
}}

[data-testid="stSidebar"] hr {{
    border-color: #36536B;
}}

[data-testid="stSidebar"] input {{
    background: white;
    color: {TEXT};
    border-radius: 8px;
}}

[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
    color: #D5E5F2;
    font-weight: 600;
}}

/* BUTTONS */

.stButton > button {{
    background: {ACCENT};
    color: white;
    border: none;
    border-radius: 9px;
    min-height: 42px;
    font-weight: 700;
    transition: all 0.2s ease;
}}

.stButton > button:hover {{
    background: #116B8B;
    color: white;
    border: none;
    transform: translateY(-1px);
}}

/* TABS */

.stTabs [data-baseweb="tab-list"] {{
    background: #E8EEF5;
    padding: 7px;
    gap: 5px;
    border-radius: 11px;
    border: 1px solid #DCE5EE;
}}

.stTabs [data-baseweb="tab"] {{
    background: transparent;
    border-radius: 8px;
    color: #52677D;
    font-weight: 650;
    padding: 10px 14px;
    border: none;
}}

.stTabs [aria-selected="true"] {{
    background: white !important;
    color: #12618A !important;
    box-shadow: 0 2px 7px rgba(16,42,67,0.09);
}}

/* KPI CARDS */

[data-testid="stMetric"] {{
    background: white;
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 17px 18px;
    box-shadow: 0 2px 8px rgba(16,42,67,0.04);
}}

[data-testid="stMetricLabel"] {{
    color: {MUTED};
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

[data-testid="stMetricValue"] {{
    color: {NAVY};
    font-weight: 800;
    font-size: 26px;
}}

/* DATAFRAMES */

[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    overflow: hidden;
}}

/* HEADINGS */

h1, h2, h3 {{
    color: {NAVY} !important;
    font-weight: 750 !important;
    letter-spacing: -0.35px;
}}

/* INPUTS */

.stSelectbox > div > div,
.stMultiSelect > div > div,
.stTextInput > div > div {{
    border-radius: 8px;
}}

div[data-testid="stAlert"] {{
    border-radius: 10px;
}}

/* SECTION DESCRIPTION */

.section-caption {{
    color: {MUTED};
    font-size: 0.91rem;
    margin-top: -8px;
    margin-bottom: 18px;
}}

/* FOOTER */

.app-footer {{
    margin-top: 32px;
    padding: 15px;
    border-top: 1px solid {BORDER};
    color: {MUTED};
    text-align: center;
    font-size: 12px;
}}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. APPLICATION HEADER
# ==========================================================

st.markdown("""
<div style="
    background:linear-gradient(115deg,#102A43,#1D4E89);
    padding:28px 32px;
    border-radius:15px;
    margin-bottom:25px;
    box-shadow:0 5px 18px rgba(16,42,67,0.12);
">

    <div style="
        color:#7DD3FC;
        font-size:11px;
        font-weight:800;
        letter-spacing:2px;
        margin-bottom:10px;">
        ENGINEERING DATA SCIENCE PLATFORM
    </div>

    <div style="
        color:white;
        font-size:29px;
        font-weight:800;
        margin-bottom:9px;">
        Thermal Properties Analytics
    </div>

    <div style="
        color:#D5E5F2;
        font-size:14px;">
        Critical Properties Prediction |
        Thermodynamic Evaluation |
        Phase Envelope Modeling
    </div>

    <div style="margin-top:18px;">

        <span style="
            background:#24618A;
            color:white;
            padding:6px 11px;
            border-radius:5px;
            font-size:10px;">
            MACHINE LEARNING
        </span>

        <span style="
            background:#24618A;
            color:white;
            padding:6px 11px;
            border-radius:5px;
            font-size:10px;
            margin-left:6px;">
            PVT ANALYTICS
        </span>

        <span style="
            background:#24618A;
            color:white;
            padding:6px 11px;
            border-radius:5px;
            font-size:10px;
            margin-left:6px;">
            PHASE BEHAVIOR
        </span>

    </div>

</div>
""", unsafe_allow_html=True)


# ==========================================================
# 4. SIDEBAR
# ==========================================================

st.sidebar.markdown("""
<div style="padding:8px 0 18px 0;">

    <div style="
        font-size:11px;
        color:#7DD3FC;
        font-weight:800;
        letter-spacing:2px;">
        THERMAL ANALYTICS
    </div>

    <div style="
        font-size:21px;
        color:white;
        font-weight:800;
        margin-top:5px;">
        Control Center
    </div>

    <div style="
        font-size:11px;
        color:#AFC4D6;
        margin-top:5px;">
        Data & Model Configuration
    </div>

</div>
""", unsafe_allow_html=True)

csv_file_path = st.sidebar.text_input(
    "CSV File Path",
    value="critprop_merged_final.csv"
)

run_button = st.sidebar.button(
    "⚡ Start & Analyze",
    type="primary",
    use_container_width=True
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 📱 Application Access")

APP_URL = "https://critical-point-prediction-dashboard-lyecr5liams58tpxqqcub3.streamlit.app"

qr = qrcode.QRCode(
    version=1,
    box_size=3,
    border=1
)

qr.add_data(APP_URL)
qr.make(fit=True)

qr_image = qr.make_image(
    fill_color="black",
    back_color="white"
)

qr_buffer = io.BytesIO()
qr_image.save(qr_buffer, format="PNG")

st.sidebar.image(
    qr_buffer.getvalue(),
    caption="Scan to open dashboard",
    width=115
)

st.sidebar.markdown("---")
st.sidebar.caption("THERMAL ANALYTICS | ENGINEERING DATA SCIENCE")


# ==========================================================
# 5. DATA LOADING
# ==========================================================

@st.cache_data(show_spinner=False)
def load_data(file_path, file_mtime_ns, file_size):

    if not os.path.exists(file_path):
        return None

    df = pd.read_csv(file_path)

    required_columns = [
        "name",
        "Vc",
        "Mw",
        "Zc",
        "Tc mean (K)",
        "Pc mean (bar)"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    numeric_columns = [
        "Vc",
        "Mw",
        "Zc",
        "Tc mean (K)",
        "Pc mean (bar)"
    ]

    if "mol weight (g/mol)" in df.columns:
        numeric_columns.append("mol weight (g/mol)")

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=required_columns
    ).copy()

    df = df[
        np.isfinite(df[numeric_columns]).all(axis=1)
    ].copy()

    if "mol weight (g/mol)" not in df.columns:
        df["mol weight (g/mol)"] = df["Mw"]

    return df.reset_index(drop=True)


# ==========================================================
# 6. MACHINE LEARNING TRAINING
# ==========================================================

@st.cache_resource(show_spinner=False)
def get_cached_training(file_path, file_mtime_ns, file_size):

    clean_df = load_data(
        file_path,
        file_mtime_ns,
        file_size
    )

    if clean_df is None or len(clean_df) < 10:
        return None

    train_df, test_df = train_test_split(
        clean_df,
        test_size=0.2,
        random_state=42
    )

    features = ["Vc", "Mw", "Zc"]

    X_train_raw = train_df[features]
    X_test_raw = test_df[features]

    y_train_tc = train_df["Tc mean (K)"]
    y_test_tc = test_df["Tc mean (K)"]

    y_train_pc = train_df["Pc mean (bar)"]
    y_test_pc = test_df["Pc mean (bar)"]

    scaler_X = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(
        X_train_raw
    )

    X_test_scaled = scaler_X.transform(
        X_test_raw
    )

    models_list = [
        "random_forest",
        "decision_tree",
        "mlp",
        "xgboost"
    ]

    trained_models = {}
    metrics_results = []
    loss_curves = {}

    for model_name in models_list:

        X_tr = (
            X_train_scaled
            if model_name == "mlp"
            else X_train_raw
        )

        X_te = (
            X_test_scaled
            if model_name == "mlp"
            else X_test_raw
        )

        train_losses_tc = []
        test_losses_tc = []

        train_losses_pc = []
        test_losses_pc = []

        # RANDOM FOREST

        if model_name == "random_forest":

            model_tc = RandomForestRegressor(
                n_estimators=10,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features=0.8,
                random_state=42,
                warm_start=True,
                n_jobs=-1
            )

            model_pc = RandomForestRegressor(
                n_estimators=10,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features=0.8,
                random_state=42,
                warm_start=True,
                n_jobs=-1
            )

            for n in [10, 20, 30, 40]:

                model_tc.n_estimators = n
                model_pc.n_estimators = n

                model_tc.fit(X_tr, y_train_tc)
                model_pc.fit(X_tr, y_train_pc)

                train_losses_tc.append(
                    mean_squared_error(
                        y_train_tc,
                        model_tc.predict(X_tr)
                    )
                )

                test_losses_tc.append(
                    mean_squared_error(
                        y_test_tc,
                        model_tc.predict(X_te)
                    )
                )

                train_losses_pc.append(
                    mean_squared_error(
                        y_train_pc,
                        model_pc.predict(X_tr)
                    )
                )

                test_losses_pc.append(
                    mean_squared_error(
                        y_test_pc,
                        model_pc.predict(X_te)
                    )
                )

        # DECISION TREE

        elif model_name == "decision_tree":

            for depth in range(1, 6):

                model_tc = DecisionTreeRegressor(
                    max_depth=depth,
                    min_samples_split=15,
                    min_samples_leaf=8,
                    random_state=42
                )

                model_pc = DecisionTreeRegressor(
                    max_depth=depth,
                    min_samples_split=15,
                    min_samples_leaf=8,
                    random_state=42
                )

                model_tc.fit(X_tr, y_train_tc)
                model_pc.fit(X_tr, y_train_pc)

                train_losses_tc.append(
                    mean_squared_error(
                        y_train_tc,
                        model_tc.predict(X_tr)
                    )
                )

                test_losses_tc.append(
                    mean_squared_error(
                        y_test_tc,
                        model_tc.predict(X_te)
                    )
                )

                train_losses_pc.append(
                    mean_squared_error(
                        y_train_pc,
                        model_pc.predict(X_tr)
                    )
                )

                test_losses_pc.append(
                    mean_squared_error(
                        y_test_pc,
                        model_pc.predict(X_te)
                    )
                )

        # MLP

        elif model_name == "mlp":

            model_tc = MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                alpha=0.05,
                max_iter=1,
                warm_start=True,
                random_state=42
            )

            model_pc = MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                alpha=0.05,
                max_iter=1,
                warm_start=True,
                random_state=42
            )

            with warnings.catch_warnings():

                warnings.simplefilter("ignore")

                for _ in range(25):

                    model_tc.fit(X_tr, y_train_tc)
                    model_pc.fit(X_tr, y_train_pc)

                    train_losses_tc.append(
                        mean_squared_error(
                            y_train_tc,
                            model_tc.predict(X_tr)
                        )
                    )

                    test_losses_tc.append(
                        mean_squared_error(
                            y_test_tc,
                            model_tc.predict(X_te)
                        )
                    )

                    train_losses_pc.append(
                        mean_squared_error(
                            y_train_pc,
                            model_pc.predict(X_tr)
                        )
                    )

                    test_losses_pc.append(
                        mean_squared_error(
                            y_test_pc,
                            model_pc.predict(X_te)
                        )
                    )

        # XGBOOST

        elif model_name == "xgboost":

            model_tc = XGBRegressor(
                n_estimators=40,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective="reg:squarederror",
                n_jobs=-1,
                verbosity=0
            )

            model_pc = XGBRegressor(
                n_estimators=40,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective="reg:squarederror",
                n_jobs=-1,
                verbosity=0
            )

            for n in [10, 20, 30, 40]:

                model_tc.set_params(n_estimators=n)
                model_pc.set_params(n_estimators=n)

                model_tc.fit(X_tr, y_train_tc)
                model_pc.fit(X_tr, y_train_pc)

                train_losses_tc.append(
                    mean_squared_error(
                        y_train_tc,
                        model_tc.predict(X_tr)
                    )
                )

                test_losses_tc.append(
                    mean_squared_error(
                        y_test_tc,
                        model_tc.predict(X_te)
                    )
                )

                train_losses_pc.append(
                    mean_squared_error(
                        y_train_pc,
                        model_pc.predict(X_tr)
                    )
                )

                test_losses_pc.append(
                    mean_squared_error(
                        y_test_pc,
                        model_pc.predict(X_te)
                    )
                )

        # MODEL PREDICTIONS

        pred_train_tc = model_tc.predict(X_tr)
        pred_test_tc = model_tc.predict(X_te)

        pred_train_pc = model_pc.predict(X_tr)
        pred_test_pc = model_pc.predict(X_te)

        trained_models[model_name] = {
            "model_tc": model_tc,
            "model_pc": model_pc
        }

        # TRAIN METRICS

        metrics_results.append({

            "Model": model_name.upper(),
            "Dataset": "Train",

            "RMSE_Tc": np.sqrt(
                mean_squared_error(
                    y_train_tc,
                    pred_train_tc
                )
            ),

            "MAE_Tc": mean_absolute_error(
                y_train_tc,
                pred_train_tc
            ),

            "R2_Tc": r2_score(
                y_train_tc,
                pred_train_tc
            ),

            "RMSE_Pc": np.sqrt(
                mean_squared_error(
                    y_train_pc,
                    pred_train_pc
                )
            ),

            "MAE_Pc": mean_absolute_error(
                y_train_pc,
                pred_train_pc
            ),

            "R2_Pc": r2_score(
                y_train_pc,
                pred_train_pc
            )

        })

        # TEST METRICS

        metrics_results.append({

            "Model": model_name.upper(),
            "Dataset": "Test",

            "RMSE_Tc": np.sqrt(
                mean_squared_error(
                    y_test_tc,
                    pred_test_tc
                )
            ),

            "MAE_Tc": mean_absolute_error(
                y_test_tc,
                pred_test_tc
            ),

            "R2_Tc": r2_score(
                y_test_tc,
                pred_test_tc
            ),

            "RMSE_Pc": np.sqrt(
                mean_squared_error(
                    y_test_pc,
                    pred_test_pc
                )
            ),

            "MAE_Pc": mean_absolute_error(
                y_test_pc,
                pred_test_pc
            ),

            "R2_Pc": r2_score(
                y_test_pc,
                pred_test_pc
            )

        })

        loss_curves[model_name] = {

            "train_tc": train_losses_tc,
            "test_tc": test_losses_tc,

            "train_pc": train_losses_pc,
            "test_pc": test_losses_pc

        }

    metrics_df = pd.DataFrame(metrics_results)

    return (
        clean_df,
        trained_models,
        scaler_X,
        metrics_df,
        loss_curves,
        models_list
    )


# ==========================================================
# 7. PHASE ENVELOPE FUNCTION
# ==========================================================

def generate_envelope_plot(
    df,
    mixture_composition,
    model_key,
    is_real_data,
    scaler_X,
    model_obj_dict=None
):

    comp_names = list(mixture_composition.keys())

    fractions = np.array(
        list(mixture_composition.values()),
        dtype=float
    )

    if (
        len(comp_names) == 0
        or np.any(fractions < 0)
        or fractions.sum() <= 0
    ):
        raise ValueError("Invalid mixture composition.")

    fractions = fractions / fractions.sum()

    mix_data = []

    for component in comp_names:

        match = df[
            df["name"].astype(str).str.strip().str.lower()
            == str(component).strip().lower()
        ]

        if match.empty:
            raise ValueError(
                f"Component {component} not found."
            )

        row = match.iloc[0]

        if is_real_data:

            pred_tc = float(row["Tc mean (K)"])
            pred_pc = float(row["Pc mean (bar)"])

        else:

            features = np.array([[
                float(row["Vc"]),
                float(row["Mw"]),
                float(row["Zc"])
            ]])

            model_tc = model_obj_dict["model_tc"]
            model_pc = model_obj_dict["model_pc"]

            if model_key == "mlp":
                features = scaler_X.transform(features)

            pred_tc = float(
                model_tc.predict(features)[0]
            )

            pred_pc = float(
                model_pc.predict(features)[0]
            )

        mix_data.append({

            "name": row["name"],
            "Tc": pred_tc,
            "Pc": pred_pc

        })

    df_mix = pd.DataFrame(mix_data)

    df_mix["zi"] = fractions

    Tc_mix = np.sum(
        df_mix["zi"] * df_mix["Tc"]
    )

    Pc_mix = np.sum(
        df_mix["zi"] * df_mix["Pc"]
    )

    volatility_ratio = (

        np.max(df_mix["Tc"]) / np.min(df_mix["Tc"])

        if len(df_mix) > 1 and np.min(df_mix["Tc"]) > 0

        else 1.1

    )

    p_mult = 1.12 + 0.08 * (
        volatility_ratio - 1.0
    )

    t_mult = 1.15 + 0.12 * (
        volatility_ratio - 1.0
    )

    Pcc = p_mult * Pc_mix
    T_cb = 0.76 * Tc_mix
    Tcc = t_mult * Tc_mix
    P_ct = 0.48 * Pc_mix

    points = np.array([

        [0.28 * Tc_mix, 0.26 * Pc_mix],
        [0.44 * Tc_mix, 0.68 * Pc_mix],
        [0.58 * Tc_mix, 0.95 * Pc_mix],
        [T_cb, Pcc],
        [0.88 * Tc_mix, 1.10 * Pc_mix],
        [Tc_mix, Pc_mix],
        [1.10 * Tc_mix, 0.85 * Pc_mix],
        [Tcc, P_ct],
        [1.16 * Tc_mix, 0.24 * Pc_mix],
        [0.98 * Tc_mix, 0.12 * Pc_mix],
        [0.68 * Tc_mix, 0.08 * Pc_mix]

    ])

    s = np.linspace(
        0, 1, len(points)
    )

    s_fine = np.linspace(
        0, 1, 500
    )

    cs_T = CubicSpline(
        s,
        points[:, 0],
        bc_type="natural"
    )

    cs_P = CubicSpline(
        s,
        points[:, 1],
        bc_type="natural"
    )

    T_vals = cs_T(s_fine)
    P_vals = cs_P(s_fine)

    fig, ax = plt.subplots(
        figsize=(7.5, 5.2),
        dpi=110,
        facecolor="white"
    )

    ax.set_facecolor("white")

    ax.plot(
        T_vals,
        P_vals,
        color=BLUE,
        linewidth=2.2
    )

    ax.scatter(
        [Tc_mix],
        [Pc_mix],
        color=RED,
        s=55,
        zorder=5,
        label="Mixture critical-property estimate"
    )

    comp_str = ", ".join([

        f"{row['zi']*100:.0f}% {row['name']}"

        for _, row in df_mix.iterrows()

    ])

    display_name = (
        "REAL DATA"
        if is_real_data
        else model_key.upper()
    )

    ax.set_title(
        f"Estimated P-T Envelope: {display_name}\n"
        f"{comp_str}\n"
        f"Tc = {Tc_mix:.1f} K | Pc = {Pc_mix:.1f} bar",
        fontsize=9,
        fontweight="bold",
        color=NAVY
    )

    ax.set_xlabel(
        "Temperature (K)",
        fontsize=10,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Pressure (bar)",
        fontsize=10,
        fontweight="bold"
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.35
    )

    ax.legend(
        fontsize=8,
        frameon=False
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    return fig, T_vals, P_vals, Tc_mix, Pc_mix


# ==========================================================
# 8. MAIN APPLICATION
# ==========================================================

if run_button:
    st.session_state["trained"] = True

if st.session_state.get("trained", False):

    if not os.path.exists(csv_file_path):

        st.error(
            f"CSV file not found: {csv_file_path}"
        )

        st.stop()

    file_stat = os.stat(csv_file_path)

    try:

        with st.spinner(
            "Loading dataset and training machine-learning models..."
        ):

            package = get_cached_training(
                csv_file_path,
                file_stat.st_mtime_ns,
                file_stat.st_size
            )

        if package is None:

            st.error(
                "The dataset must contain at least 10 valid records."
            )

            st.stop()

        (
            clean_df,
            trained_models,
            scaler_X,
            metrics_df,
            loss_curves,
            models_list
        ) = package

    except Exception as error:

        st.error(
            f"Unable to prepare analysis: {error}"
        )

        st.stop()

    model_colors = {

        "real_data": NAVY,
        "random_forest": "#2563EB",
        "decision_tree": GREEN,
        "mlp": ORANGE,
        "xgboost": RED

    }

    # ======================================================
    # WORKSPACE TABS
    # ======================================================

    tab_eda, tab_metrics, tab_loss, tab_predictions, tab_envelope = st.tabs([

        "🔍 Data Overview",
        "📊 Model Performance",
        "📉 Training Analytics",
        "🌐 Predictions",
        "📈 Phase Envelopes"

    ])

    # ======================================================
    # TAB 1: EDA
    # ======================================================

    with tab_eda:

        st.subheader(
            "Exploratory Data Analysis"
        )

        st.markdown(
            '<div class="section-caption">'
            'Dataset quality, descriptive statistics and '
            'critical-property distributions.'
            '</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Components",
            f"{len(clean_df):,}"
        )

        col2.metric(
            "Average Tc",
            f"{clean_df['Tc mean (K)'].mean():.2f} K"
        )

        col3.metric(
            "Average Pc",
            f"{clean_df['Pc mean (bar)'].mean():.2f} bar"
        )

        col4.metric(
            "ML Algorithms",
            f"{len(models_list):02d}"
        )

        st.markdown("### Dataset Preview")

        st.dataframe(
            clean_df.head(10),
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            "### Data Distributions & Relationships"
        )

        col_a, col_b = st.columns(2)

        with col_a:

            fig, ax = plt.subplots(
                figsize=(6, 4),
                dpi=110
            )

            ax.scatter(
                clean_df["mol weight (g/mol)"],
                clean_df["Tc mean (K)"],
                color="#2563EB",
                alpha=0.58
            )

            ax.set_title(
                "Molecular Weight vs Critical Temperature",
                fontweight="bold"
            )

            ax.set_xlabel("Molecular Weight (g/mol)")
            ax.set_ylabel("Tc (K)")

            ax.grid(
                True,
                linestyle="--",
                alpha=0.3
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

        with col_b:

            fig, ax = plt.subplots(
                figsize=(6, 4),
                dpi=110
            )

            ax.scatter(
                clean_df["Vc"],
                clean_df["Pc mean (bar)"],
                color=GREEN,
                alpha=0.58
            )

            ax.set_title(
                "Critical Volume vs Critical Pressure",
                fontweight="bold"
            )

            ax.set_xlabel("Critical Volume (Vc)")
            ax.set_ylabel("Pc (bar)")

            ax.grid(
                True,
                linestyle="--",
                alpha=0.3
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

        col_c, col_d = st.columns(2)

        with col_c:

            fig, ax = plt.subplots(
                figsize=(6, 4),
                dpi=110
            )

            ax.hist(
                clean_df["Tc mean (K)"],
                bins=30,
                color="#2563EB",
                edgecolor="white",
                alpha=0.82
            )

            ax.set_title(
                "Critical Temperature Distribution",
                fontweight="bold"
            )

            ax.set_xlabel("Tc (K)")
            ax.set_ylabel("Frequency")

            ax.grid(
                True,
                axis="y",
                linestyle="--",
                alpha=0.3
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

        with col_d:

            fig, ax = plt.subplots(
                figsize=(6, 4),
                dpi=110
            )

            ax.hist(
                clean_df["Pc mean (bar)"],
                bins=30,
                color=ORANGE,
                edgecolor="white",
                alpha=0.82
            )

            ax.set_title(
                "Critical Pressure Distribution",
                fontweight="bold"
            )

            ax.set_xlabel("Pc (bar)")
            ax.set_ylabel("Frequency")

            ax.grid(
                True,
                axis="y",
                linestyle="--",
                alpha=0.3
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

        with st.expander("Descriptive Statistics"):

            st.dataframe(
                clean_df.describe().T,
                use_container_width=True
            )

    # ======================================================
    # TAB 2: MODEL PERFORMANCE
    # ======================================================

    with tab_metrics:

        st.subheader(
            "Model Performance Comparison"
        )

        st.markdown(
            '<div class="section-caption">'
            'Training and testing metrics for critical temperature '
            '(Tc) and critical pressure (Pc).'
            '</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            metrics_df.round(4),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Test Set Comparison")

        test_metrics = metrics_df[
            metrics_df["Dataset"] == "Test"
        ].copy()

        col_tc, col_pc = st.columns(2)

        with col_tc:

            fig, ax = plt.subplots(
                figsize=(6, 4),
                dpi=110
            )

            ax.bar(
                test_metrics["Model"],
                test_metrics["RMSE_Tc"],
                color="#2563EB",
                width=0.62
            )

            ax.set_title(
                "Test RMSE — Tc",
                fontweight="bold"
            )

            ax.set_ylabel("RMSE (K)")

            ax.grid(
                True,
                axis="y",
                linestyle="--",
                alpha=0.3
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

        with col_pc:

            fig, ax = plt.subplots(
                figsize=(6, 4),
                dpi=110
            )

            ax.bar(
                test_metrics["Model"],
                test_metrics["RMSE_Pc"],
                color=GREEN,
                width=0.62
            )

            ax.set_title(
                "Test RMSE — Pc",
                fontweight="bold"
            )

            ax.set_ylabel("RMSE (bar)")

            ax.grid(
                True,
                axis="y",
                linestyle="--",
                alpha=0.3
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

        best_tc = test_metrics.loc[
            test_metrics["RMSE_Tc"].idxmin()
        ]

        best_pc = test_metrics.loc[
            test_metrics["RMSE_Pc"].idxmin()
        ]

        best1, best2 = st.columns(2)

        best1.metric(
            "Lowest Test RMSE — Tc",
            best_tc["Model"],
            f"{best_tc['RMSE_Tc']:.4f} K"
        )

        best2.metric(
            "Lowest Test RMSE — Pc",
            best_pc["Model"],
            f"{best_pc['RMSE_Pc']:.4f} bar"
        )

    # ======================================================
    # TAB 3: LOSS CURVES
    # ======================================================

    with tab_loss:

        st.subheader(
            "Training & Testing Loss Curves"
        )

        st.markdown(
            '<div class="section-caption">'
            'Mean Squared Error across model training stages.'
            '</div>',
            unsafe_allow_html=True
        )

        chart_columns = st.columns(2)

        for index, model_name in enumerate(models_list):

            curve = loss_curves[model_name]

            with chart_columns[index % 2]:

                fig, axes = plt.subplots(
                    1,
                    2,
                    figsize=(9, 3.8),
                    dpi=105
                )

                axes[0].plot(
                    curve["train_tc"],
                    label="Train",
                    color="#2563EB",
                    linewidth=2
                )

                axes[0].plot(
                    curve["test_tc"],
                    label="Test",
                    color=RED,
                    linestyle="--",
                    linewidth=2
                )

                axes[0].set_title(
                    f"{model_name.upper()} — Tc",
                    fontweight="bold",
                    fontsize=9
                )

                axes[0].set_ylabel("MSE")
                axes[0].set_xlabel("Training Stage")
                axes[0].legend(fontsize=8)

                axes[0].grid(
                    True,
                    linestyle="--",
                    alpha=0.3
                )

                axes[1].plot(
                    curve["train_pc"],
                    label="Train",
                    color="#2563EB",
                    linewidth=2
                )

                axes[1].plot(
                    curve["test_pc"],
                    label="Test",
                    color=RED,
                    linestyle="--",
                    linewidth=2
                )

                axes[1].set_title(
                    f"{model_name.upper()} — Pc",
                    fontweight="bold",
                    fontsize=9
                )

                axes[1].set_ylabel("MSE")
                axes[1].set_xlabel("Training Stage")
                axes[1].legend(fontsize=8)

                axes[1].grid(
                    True,
                    linestyle="--",
                    alpha=0.3
                )

                fig.tight_layout()

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)

    # ======================================================
    # TAB 4: PREDICTIONS
    # ======================================================

    with tab_predictions:

        st.subheader(
            "Critical Properties Prediction"
        )

        st.markdown(
            '<div class="section-caption">'
            'Actual and predicted critical properties for all '
            'valid components in the dataset.'
            '</div>',
            unsafe_allow_html=True
        )

        prediction_rows = []

        for _, row in clean_df.iterrows():

            features = np.array([[
                float(row["Vc"]),
                float(row["Mw"]),
                float(row["Zc"])
            ]])

            features_scaled = scaler_X.transform(features)

            prediction = {

                "Component": row["name"],

                "Actual Tc (K)": round(
                    float(row["Tc mean (K)"]), 2
                ),

                "Actual Pc (bar)": round(
                    float(row["Pc mean (bar)"]), 2
                )

            }

            for model_name in models_list:

                model_features = (
                    features_scaled
                    if model_name == "mlp"
                    else features
                )

                predicted_tc = trained_models[
                    model_name
                ]["model_tc"].predict(
                    model_features
                )[0]

                predicted_pc = trained_models[
                    model_name
                ]["model_pc"].predict(
                    model_features
                )[0]

                prediction[
                    f"{model_name.upper()} Tc (K)"
                ] = round(
                    float(predicted_tc), 2
                )

                prediction[
                    f"{model_name.upper()} Pc (bar)"
                ] = round(
                    float(predicted_pc), 2
                )

            prediction_rows.append(prediction)

        predictions_df = pd.DataFrame(
            prediction_rows
        )

        search_component = st.text_input(
            "Search Component",
            placeholder="Enter component name..."
        )

        display_predictions = predictions_df.copy()

        if search_component.strip():

            display_predictions = display_predictions[
                display_predictions["Component"]
                .astype(str)
                .str.contains(
                    search_component.strip(),
                    case=False,
                    na=False
                )
            ]

        st.dataframe(
            display_predictions,
            use_container_width=True,
            hide_index=True
        )

        csv_output = predictions_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Predictions CSV",
            data=csv_output,
            file_name="critical_properties_predictions.csv",
            mime="text/csv"
        )

        st.caption(
            "Note: predictions for all components include training "
            "and test records. Test-set metrics above should be used "
            "to evaluate model generalization."
        )

    # ======================================================
    # TAB 5: PHASE ENVELOPE
    # ======================================================

    with tab_envelope:

        st.subheader(
            "P-T Phase Envelope Workspace"
        )

        st.markdown(
            '<div class="section-caption">'
            'Compare estimated mixture envelopes using actual or '
            'machine-learning-predicted critical properties.'
            '</div>',
            unsafe_allow_html=True
        )

        st.info(
            "Scientific note: The current envelope is a heuristic "
            "visualization based on estimated mixture properties. "
            "It is not an EOS-based phase-equilibrium calculation."
        )

        available_names = (
            clean_df["name"]
            .dropna()
            .unique()
            .tolist()
        )

        system_category = st.selectbox(
            "Select System Category",

            [
                "Pure Component",
                "Ternary Mixture (3 Components)",
                "Pentary Mixture (up to 5 Components)",
                "Custom Mixture"
            ],

            key="system_category_select"
        )

        mix_dict = {}

        # PURE COMPONENT

        if system_category == "Pure Component":

            selected_pure = st.selectbox(
                "Select Pure Component",
                options=available_names,
                key="pure_component_select"
            )

            if selected_pure:
                mix_dict = {
                    selected_pure: 1.0
                }

        # TERNARY MIXTURE

        elif system_category == "Ternary Mixture (3 Components)":

            selected_ternary = st.multiselect(
                "Select exactly 3 components",

                options=available_names,

                default=available_names[:3],

                max_selections=3,

                key="ternary_mixture_select"
            )

            if len(selected_ternary) == 3:

                fraction = 1.0 / 3

                mix_dict = {
                    comp: fraction
                    for comp in selected_ternary
                }

            elif selected_ternary:

                st.warning(
                    "Please select exactly 3 components."
                )

        # PENTARY MIXTURE

        elif system_category == "Pentary Mixture (up to 5 Components)":

            selected_pentary = st.multiselect(
                "Select up to 5 components",

                options=available_names,

                default=available_names[:5],

                max_selections=5,

                key="pentary_mixture_select"
            )

            if selected_pentary:

                fraction = 1.0 / len(selected_pentary)

                mix_dict = {
                    comp: fraction
                    for comp in selected_pentary
                }

        # CUSTOM MIXTURE

        else:

            selected_custom = st.multiselect(
                "Select Custom Components",

                options=available_names,

                default=available_names[:2],

                key="custom_mixture_select"
            )

            if selected_custom:

                fraction = 1.0 / len(selected_custom)

                mix_dict = {
                    comp: fraction
                    for comp in selected_custom
                }

        # PLOT

        if mix_dict:

            st.markdown(
                "### Mixture Composition"
            )

            composition_df = pd.DataFrame({

                "Component": list(mix_dict.keys()),

                "Mole Fraction (zi)": list(mix_dict.values()),

                "Mole Percent (%)": [
                    value * 100
                    for value in mix_dict.values()
                ]

            })

            st.dataframe(
                composition_df,
                use_container_width=True,
                hide_index=True
            )

            try:

                fig_comb, ax = plt.subplots(
                    figsize=(10, 6),
                    dpi=115,
                    facecolor="white"
                )

                ax.set_facecolor("white")

                # REAL DATA

                fig_real, T_real, P_real, Tc_real, Pc_real = (
                    generate_envelope_plot(
                        clean_df,
                        mix_dict,
                        "real_data",
                        True,
                        scaler_X
                    )
                )

                plt.close(fig_real)

                ax.plot(
                    T_real,
                    P_real,
                    color=model_colors["real_data"],
                    linewidth=2.5,
                    label="Actual Critical Properties"
                )

                # MACHINE LEARNING MODELS

                for model_name in models_list:

                    fig_model, T_model, P_model, _, _ = (
                        generate_envelope_plot(
                            clean_df,
                            mix_dict,
                            model_name,
                            False,
                            scaler_X,
                            trained_models[model_name]
                        )
                    )

                    plt.close(fig_model)

                    ax.plot(
                        T_model,
                        P_model,
                        color=model_colors[model_name],
                        linewidth=1.8,
                        label=f"{model_name.upper()} Prediction"
                    )

                ax.set_xlabel(
                    "Temperature (K)",
                    fontsize=10,
                    fontweight="bold"
                )

                ax.set_ylabel(
                    "Pressure (bar)",
                    fontsize=10,
                    fontweight="bold"
                )

                ax.set_title(
                    "Estimated P-T Phase Envelope Comparison",
                    fontsize=12,
                    fontweight="bold",
                    color=NAVY
                )

                ax.legend(
                    frameon=True,
                    facecolor="white",
                    edgecolor=BORDER,
                    fontsize=8
                )

                ax.grid(
                    True,
                    linestyle="--",
                    alpha=0.3
                )

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                fig_comb.tight_layout()

                st.pyplot(
                    fig_comb,
                    use_container_width=True
                )

                plt.close(fig_comb)

                result1, result2 = st.columns(2)

                result1.metric(
                    "Mixture Tc (Actual Data)",
                    f"{Tc_real:.2f} K"
                )

                result2.metric(
                    "Mixture Pc (Actual Data)",
                    f"{Pc_real:.2f} bar"
                )

            except Exception as error:

                st.error(
                    f"Unable to generate phase envelope: {error}"
                )

    # ======================================================
    # FOOTER
    # ======================================================

    st.markdown("""
    <div class="app-footer">

        THERMAL ANALYTICS | ENGINEERING DATA SCIENCE

        <br>

        Critical Properties Prediction &
        Thermodynamic Analysis

    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# 9. INITIAL STATE
# ==========================================================

else:

    st.markdown("""
    <div style="
        background:white;
        border:1px solid #E1E8F0;
        border-radius:14px;
        padding:35px;
        text-align:center;
        margin-top:25px;
    ">

        <div style="font-size:36px;margin-bottom:12px;">
            📂
        </div>

        <div style="
            font-size:21px;
            font-weight:800;
            color:#102A43;">
            Ready to Analyze
        </div>

        <div style="
            font-size:13px;
            color:#718096;
            margin-top:9px;">
            Confirm the CSV file path in the sidebar,
            then select Start & Analyze.
        </div>

    </div>
    """, unsafe_allow_html=True)

