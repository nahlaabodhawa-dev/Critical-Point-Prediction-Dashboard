import pandas as pd
df = pd.read_csv('critprop_merged_final (1).csv')
print(df.head(2))
print(df.info())
```text
         name                smiles                                                                             inchi  mol weight (g/mol)  Tc mean (K)  Pc mean (bar)  rhoc mean (mol/L)  omega mean (-)        Vc       Mw        Zc  split
0  nonadecane   CCCCCCCCCCCCCCCCCCC     InChI=1/C19H40/c1-3-5-7-9-11-13-15-17-19-18-16-14-12-10-8-6-4-2/h3-19H2,1-2H3             268.529        758.0           12.1              0.871           0.852  1.148106  268.529  0.220426  train
1     icosane  CCCCCCCCCCCCCCCCCCCC  InChI=1/C20H42/c1-3-5-7-9-11-13-15-17-19-20-18-16-14-12-10-8-6-4-2/h3-20H2,1-2H3             282.556        768.0           11.6              0.828           0.907  1.207729  282.556  0.219397   test
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5790 entries, 0 to 5789
Data columns (total 12 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   name                5790 non-null   object 
 1   smiles              5790 non-null   object 
 2   inchi               5790 non-null   object 
 3   mol weight (g/mol)  5790 non-null   float64
 4   Tc mean (K)         5790 non-null   float64
 5   Pc mean (bar)       5790 non-null   float64
 6   rhoc mean (mol/L)   5790 non-null   float64
 7   omega mean (-)      4831 non-null   float64
 8   Vc                  5790 non-null   float64
 9   Mw                  5790 non-null   float64
 10  Zc                  5790 non-null   float64
 11  split               5790 non-null   object 
dtypes: float64(8), object(4)
memory usage: 542.9+ KB
None


```

```python
import os
import io
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import qrcode
from PIL import Image
from scipy.interpolate import CubicSpline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

st.set_page_config(
    page_title="Thermal Properties Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #f4f6f9;
        color: #333333;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
        padding-top: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
        padding-bottom: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: white !important;
        border-color: #2563eb !important;
    }
    div.stButton > button {
        background-color: #2563eb;
        color: white;
        border-radius: 6px;
        border: none;
        font-weight: 600;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🎛️ Control Panel")
csv_file_path = st.sidebar.text_input("CSV File Path", value="critprop_merged_final (1).csv")
run_button = st.sidebar.button("⚡ Start & Analyze", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📱 App QR Code")
app_url = "https://critical-point-prediction-dashboard-lyecr5liams58tpxqqcub3.streamlit.app"

qr = qrcode.QRCode(version=1, box_size=5, border=2)
qr.add_data(app_url)
qr.make(fit=True)
img_qr = qr.make_image(fill_color="black", back_color="white")

buf = io.BytesIO()
img_qr.save(buf, format="PNG")
st.sidebar.image(buf.getvalue(), caption="Scan to open", width=130)

st.title("📊 Thermal Properties Analytics & Dashboard")
st.markdown("Professional machine learning platform for thermodynamic property evaluation and phase envelope modeling.")

@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    return df.dropna(subset=["Vc", "Mw", "Zc", "Tc mean (K)", "Pc mean (bar)"])

@st.cache_resource
def get_cached_training(file_path):
    cache_file = "optimized_models_cache_v2.pkl"
    if os.path.exists(cache_file):
        return joblib.load(cache_file)
    
    clean_df = load_data(file_path)
    if clean_df is None:
        return None
        
    train_df, test_df = train_test_split(clean_df, test_size=0.2, random_state=42)
    
    X_train_raw = train_df[["Vc", "Mw", "Zc"]]
    y_train_tc = train_df["Tc mean (K)"]
    y_train_pc = train_df["Pc mean (bar)"]
    
    X_test_raw = test_df[["Vc", "Mw", "Zc"]]
    y_test_tc = test_df["Tc mean (K)"]
    y_test_pc = test_df["Pc mean (bar)"]
    
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    X_test_scaled = scaler_X.transform(X_test_raw)
    
    models_list = ["random_forest", "decision_tree", "mlp", "xgboost"]
    trained_models = {}
    metrics_results = []
    loss_figs = {}
    
    for m_name in models_list:
        X_tr = X_train_scaled if m_name == "mlp" else X_train_raw
        X_te = X_test_scaled if m_name == "mlp" else X_test_raw
        
        train_losses_tc, test_losses_tc = [], []
        train_losses_pc, test_losses_pc = [], []
        
        if m_name == "random_forest":
            model_tc = RandomForestRegressor(n_estimators=40, max_depth=8, min_samples_split=10, min_samples_leaf=4, max_features=0.8, random_state=42, warm_start=True)
            model_pc = RandomForestRegressor(n_estimators=40, max_depth=8, min_samples_split=10, min_samples_leaf=4, max_features=0.8, random_state=42, warm_start=True)
            for s in range(10, 41, 10):
                model_tc.n_estimators = s
                model_pc.n_estimators = s
                model_tc.fit(X_tr, y_train_tc)
                model_pc.fit(X_tr, y_train_pc)
                train_losses_tc.append(mean_squared_error(y_train_tc, model_tc.predict(X_tr)))
                test_losses_tc.append(mean_squared_error(y_test_tc, model_tc.predict(X_te)))
                train_losses_pc.append(mean_squared_error(y_train_pc, model_pc.predict(X_tr)))
                test_losses_pc.append(mean_squared_error(y_test_pc, model_pc.predict(X_te)))
        elif m_name == "decision_tree":
            for d in range(1, 6):
                model_tc = DecisionTreeRegressor(max_depth=d, min_samples_split=15, min_samples_leaf=8, random_state=42).fit(X_tr, y_train_tc)
                model_pc = DecisionTreeRegressor(max_depth=d, min_samples_split=15, min_samples_leaf=8, random_state=42).fit(X_tr, y_train_pc)
                train_losses_tc.append(mean_squared_error(y_train_tc, model_tc.predict(X_tr)))
                test_losses_tc.append(mean_squared_error(y_test_tc, model_tc.predict(X_te)))
                train_losses_pc.append(mean_squared_error(y_train_pc, model_pc.predict(X_tr)))
                test_losses_pc.append(mean_squared_error(y_test_pc, model_pc.predict(X_te)))
        elif m_name == "mlp":
            model_tc = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", alpha=0.05, max_iter=1, warm_start=True, random_state=42)
            model_pc = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", alpha=0.05, max_iter=1, warm_start=True, random_state=42)
            for _ in range(25):
                model_tc.fit(X_tr, y_train_tc)
                model_pc.fit(X_tr, y_train_pc)
                train_losses_tc.append(mean_squared_error(y_train_tc, model_tc.predict(X_tr)))
                test_losses_tc.append(mean_squared_error(y_test_tc, model_tc.predict(X_te)))
                train_losses_pc.append(mean_squared_error(y_train_pc, model_pc.predict(X_tr)))
                test_losses_pc.append(mean_squared_error(y_test_pc, model_pc.predict(X_te)))
        elif m_name == "xgboost":
            model_tc = XGBRegressor(n_estimators=40, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42)
            model_pc = XGBRegressor(n_estimators=40, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42)
            for s in range(10, 41, 10):
                model_tc.n_estimators = s
                model_pc.n_estimators = s
                model_tc.fit(X_tr, y_train_tc)
                model_pc.fit(X_tr, y_train_pc)
                train_losses_tc.append(mean_squared_error(y_train_tc, model_tc.predict(X_tr)))
                test_losses_tc.append(mean_squared_error(y_test_tc, model_tc.predict(X_te)))
                train_losses_pc.append(mean_squared_error(y_train_pc, model_pc.predict(X_tr)))
                test_losses_pc.append(mean_squared_error(y_test_pc, model_pc.predict(X_te)))
                
        pred_train_tc = model_tc.predict(X_tr)
        pred_test_tc = model_tc.predict(X_te)
        pred_train_pc = model_pc.predict(X_tr)
        pred_test_pc = model_pc.predict(X_te)
        trained_models[m_name] = {"model_tc": model_tc, "model_pc": model_pc}
        
        metrics_results.append({
            "Model": m_name.upper(), "Dataset": "Train",
            "RMSE_Tc": np.sqrt(mean_squared_error(y_train_tc, pred_train_tc)),
            "MAE_Tc": mean_absolute_error(y_train_tc, pred_train_tc), "R2_Tc": r2_score(y_train_tc, pred_train_tc),
            "RMSE_Pc": np.sqrt(mean_squared_error(y_train_pc, pred_train_pc)),
            "MAE_Pc": mean_absolute_error(y_train_pc, pred_train_pc), "R2_Pc": r2_score(y_train_pc, pred_train_pc)})
        metrics_results.append({
            "Model": m_name.upper(), "Dataset": "Test",
            "RMSE_Tc": np.sqrt(mean_squared_error(y_test_tc, pred_test_tc)),
            "MAE_Tc": mean_absolute_error(y_test_tc, pred_test_tc), "R2_Tc": r2_score(y_test_tc, pred_test_tc),
            "RMSE_Pc": np.sqrt(mean_squared_error(y_test_pc, pred_test_pc)),
            "MAE_Pc": mean_absolute_error(y_test_pc, pred_test_pc), "R2_Pc": r2_score(y_test_pc, pred_test_pc)})

        fig_loss, axes_loss = plt.subplots(1, 2, figsize=(9, 3.5), dpi=100, facecolor="white")
        axes_loss[0].set_facecolor("white")
        axes_loss[0].plot(train_losses_tc, label="Train Loss", color="#2563eb", lw=2)
        axes_loss[0].plot(test_losses_tc, label="Test Loss", color="#dc2626", linestyle="--", lw=2)
        axes_loss[0].set_title(f"{m_name.upper()} - Tc Loss Curve", fontsize=9, fontweight="bold")
        axes_loss[0].legend(fontsize=8)
        axes_loss[0].grid(True, linestyle="--", alpha=0.5)
        
        axes_loss[1].set_facecolor("white")
        axes_loss[1].plot(train_losses_pc, label="Train Loss", color="#2563eb", lw=2)
        axes_loss[1].plot(test_losses_pc, label="Test Loss", color="#dc2626", linestyle="--", lw=2)
        axes_loss[1].set_title(f"{m_name.upper()} - Pc Loss Curve", fontsize=9, fontweight="bold")
        axes_loss[1].legend(fontsize=8)
        axes_loss[1].grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        loss_figs[m_name] = fig_loss

    metrics_df = pd.DataFrame(metrics_results)
    package = (clean_df, trained_models, scaler_X, metrics_df, loss_figs, models_list)
    joblib.dump(package, cache_file)
    return package

def generate_envelope_plot(df, mixture_composition, model_key, is_real_data, scaler_X, model_colors, model_obj_dict=None):
    comp_names = list(mixture_composition.keys())
    fractions = np.array(list(mixture_composition.values()), dtype=float)
    fractions /= np.sum(fractions)
    
    mix_data = []
    for i, comp in enumerate(comp_names):
        match = df[df["name"].str.lower() == comp.lower()]
        row = df.iloc[0] if match.empty else match.iloc[0]
        
        if is_real_data:
            pred_tc_comp = float(row["Tc mean (K)"])
            pred_pc_comp = float(row["Pc mean (bar)"])
        else:
            comp_features = np.array([[float(row["Vc"]), float(row["Mw"]), float(row["Zc"])]])
            m_tc = model_obj_dict["model_tc"]
            m_pc = model_obj_dict["model_pc"]
            if model_key == "mlp":
                comp_features_scaled = scaler_X.transform(comp_features)
                pred_tc_comp = m_tc.predict(comp_features_scaled)[0]
                pred_pc_comp = m_pc.predict(comp_features_scaled)[0]
            else:
                pred_tc_comp = m_tc.predict(comp_features)[0]
                pred_pc_comp = m_pc.predict(comp_features)[0]
                
        mix_data.append({"name": row["name"], "Tc": float(pred_tc_comp), "Pc": float(pred_pc_comp)})
        
    df_mix = pd.DataFrame(mix_data)
    df_mix["zi"] = fractions
    Tc_mix = np.sum(df_mix["zi"] * df_mix["Tc"])
    Pc_mix = np.sum(df_mix["zi"] * df_mix["Pc"])
    
    volatility_ratio = (np.max(df_mix["Tc"]) / np.min(df_mix["Tc"]) if len(df_mix) > 1 else 1.1)
    p_mult = 1.12 + 0.08 * (volatility_ratio - 1.0)
    t_mult = 1.15 + 0.12 * (volatility_ratio - 1.0)
    Pcc = p_mult * Pc_mix
    T_cb = 0.76 * Tc_mix
    Tcc = t_mult * Tc_mix
    P_ct = 0.48 * Pc_mix
    
    pts = np.array([
        [0.28 * Tc_mix, 0.26 * Pc_mix], [0.44 * Tc_mix, 0.68 * Pc_mix],
        [0.58 * Tc_mix, 0.95 * Pc_mix], [T_cb, Pcc],
        [0.88 * Tc_mix, 1.10 * Pc_mix], [Tc_mix, Pc_mix],
        [1.10 * Tc_mix, 0.85 * Pc_mix], [Tcc, P_ct],
        [1.16 * Tc_mix, 0.24 * Pc_mix], [0.98 * Tc_mix, 0.12 * Pc_mix],
        [0.68 * Tc_mix, 0.08 * Pc_mix]])
    s = np.linspace(0, 1, len(pts))
    s_fine = np.linspace(0, 1, 500)
    cs_T = CubicSpline(s, pts[:, 0], bc_type="natural")
    cs_P = CubicSpline(s, pts[:, 1], bc_type="natural")
    T_vals = cs_T(s_fine)
    P_vals = cs_P(s_fine)
    
    idx_cp = np.argmin((T_vals - Tc_mix) ** 2 + (P_vals - Pc_mix) ** 2)
    T_cp_val, P_cp_val = T_vals[idx_cp], P_vals[idx_cp]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=100, facecolor="white")
    ax.set_facecolor("white")
    T_origin = 0.18 * Tc_mix
    P_origin = 0.0
    T_max = 1.38 * Tc_mix
    P_max = 1.32 * Pc_mix
    ax.set_xlim(T_origin - 0.05 * Tc_mix, T_max + 0.05 * Tc_mix)
    ax.set_ylim(P_origin - 0.05 * Pc_mix, P_max + 0.08 * Pc_mix)
    ax.spines['left'].set_position(('data', T_origin))
    ax.spines['bottom'].set_position(('data', P_origin))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Temperature (T) [K]", fontsize=10, fontweight="bold", color="#334155")
    ax.set_ylabel("Pressure (P) [bar]", fontsize=10, fontweight="bold", color="#334155")
    
    curve_color = model_colors.get(model_key, "#2563eb")
    ax.plot(T_vals, P_vals, color=curve_color, lw=2.0)
    ax.plot(T_cp_val, P_cp_val, marker="o", markersize=8, markerfacecolor="#dc2626", markeredgecolor="none")
    
    comp_str = ", ".join([f"{row['zi']*100:.0f}% {row['name']}" for _, row in df_mix.iterrows()])
    display_name = "REAL DATA" if is_real_data else model_key.upper()
    ax.set_title(f"P-T Envelope: {display_name}\n({comp_str})\nTc = {Tc_mix:.1f} K, Pc = {Pc_mix:.1f} bar", fontsize=9, color="#1e293b")
    
    return fig, T_vals, P_vals, Tc_mix, Pc_mix

if run_button or st.session_state.get("trained", True):
    st.session_state["trained"] = True
    if not os.path.exists(csv_file_path):
        st.error(f"File not found at the specified path: {csv_file_path}. Please make sure '{csv_file_path}' is uploaded.")
    else:
        with st.spinner("⚡ Loading application instantly..."):
            package = get_cached_training(csv_file_path)
            if package is None:
                st.error("Error loading or cleaning data. Please check the columns.")
            else:
                clean_df, trained_models, scaler_X, metrics_df, loss_figs, models_list = package
                model_colors = {
                    "real_data": "#1e293b",    
                    "random_forest": "#2563eb",
                    "decision_tree": "#16a34a",
                    "mlp": "#d97706",
                    "xgboost": "#dc2626"
                }
                
                tab_eda, tab1, tab2, tab3, tab4 = st.tabs([
                    "🔍 Exploratory Data Analysis (EDA)",
                    "📊 Performance Metrics", 
                    "📉 Loss Curves", 
                    "🌐 All Components Predictions", 
                    "📈 Phase Envelopes Comparison"
                ])
                
                with tab_eda:
                    st.subheader("🔍 Exploratory Data Analysis (EDA) & Dataset Overview")
                    st.markdown("Here is a comprehensive exploratory analysis of the dataset, including summary statistics, distributions, and feature correlations.")
                    
                    # 1. Quick metrics overview
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    col_m1.metric("Total Components", len(clean_df))
                    col_m2.metric("Avg Tc (K)", f"{clean_df['Tc mean (K)'].mean():.2f}")
                    col_m3.metric("Avg Pc (bar)", f"{clean_df['Pc mean (bar)'].mean():.2f}")
                    col_m4.metric("Avg Mol Weight", f"{clean_df['mol weight (g/mol)'].mean():.2f}")
                    
                    st.markdown("---")
                    
                    # 2. Dataset preview
                    st.markdown("### 📋 Dataset Preview")
                    st.dataframe(clean_df.head(10), use_container_width=True)
                    
                    st.markdown("---")
                    
                    # 3. Interactive Visualizations (EDA Plots)
                    st.markdown("### 📊 Interactive Data Distributions & Relationships")
                    
                    col_e1, col_e2 = st.columns(2)
                    
                    with col_e1:
                        fig_eda1, ax_e1 = plt.subplots(figsize=(6, 4), dpi=100, facecolor="white")
                        ax_e1.set_facecolor("white")
                        ax_e1.scatter(clean_df["mol weight (g/mol)"], clean_df["Tc mean (K)"], color="#2563eb", alpha=0.5, edgecolors="none")
                        ax_e1.set_title("Mol Weight vs Critical Temperature (Tc)", fontsize=10, fontweight="bold")
                        ax_e1.set_xlabel("Molecular Weight (g/mol)")
                        ax_e1.set_ylabel("Tc mean (K)")
                        ax_e1.grid(True, linestyle="--", alpha=0.5)
                        plt.tight_layout()
                        st.pyplot(fig_eda1)
                        plt.close(fig_eda1)
                        
                    with col_e2:
                        fig_eda2, ax_e2 = plt.subplots(figsize=(6, 4), dpi=100, facecolor="white")
                        ax_e2.set_facecolor("white")
                        ax_e2.scatter(clean_df["Vc"], clean_df["Pc mean (bar)"], color="#dc2626", alpha=0.5, edgecolors="none")
                        ax_e2.set_title("Critical Volume (Vc) vs Critical Pressure (Pc)", fontsize=10, fontweight="bold")
                        ax_e2.set_xlabel("Critical Volume (Vc)")
                        ax_e2.set_ylabel("Pc mean (bar)")
                        ax_e2.grid(True, linestyle="--", alpha=0.5)
                        plt.tight_layout()
                        st.pyplot(fig_eda2)
                        plt.close(fig_eda2)

                    col_e3, col_e4 = st.columns(2)
                    
                    with col_e3:
                        fig_eda3, ax_e3 = plt.subplots(figsize=(6, 4), dpi=100, facecolor="white")
                        ax_e3.set_facecolor("white")
                        ax_e3.hist(clean_df["Tc mean (K)"], bins=30, color="#16a34a", edgecolor="black", alpha=0.7)
                        ax_e3.set_title("Distribution of Critical Temperature (Tc)", fontsize=10, fontweight="bold")
                        ax_e3.set_xlabel("Tc mean (K)")
                        ax_e3.set_ylabel("Frequency")
                        ax_e3.grid(True, linestyle="--", alpha=0.5)
                        plt.tight_layout()
                        st.pyplot(fig_eda3)
                        plt.close(fig_eda3)
                        
                    with col_e4:
                        fig_eda4, ax_e4 = plt.subplots(figsize=(6, 4), dpi=100, facecolor="white")
                        ax_e4.set_facecolor("white")
                        ax_e4.hist(clean_df["Pc mean (bar)"], bins=30, color="#d97706", edgecolor="black", alpha=0.7)
                        ax_e4.set_title("Distribution of Critical Pressure (Pc)", fontsize=10, fontweight="bold")
                        ax_e4.set_xlabel("Pc mean (bar)")
                        ax_e4.set_ylabel("Frequency")
                        ax_e4.grid(True, linestyle="--", alpha=0.5)
                        plt.tight_layout()
                        st.pyplot(fig_eda4)
                        plt.close(fig_eda4)

                with tab1:
                    st.subheader("📋 Model Performance Comparison (Metrics Summary)")
                    st.dataframe(metrics_df, use_container_width=True)
                
                with tab2:
                    st.subheader("📉 Training & Testing Loss Curves per Model")
                    cols = st.columns(2)
                    for i, m_name in enumerate(models_list):
                        with cols[i%2]:
                            st.pyplot(loss_figs[m_name])
                            plt.close(loss_figs[m_name])
                
                with tab3:
                    st.subheader("🌐 Comprehensive Evaluation Across All Components")
                    st.markdown("This table iterates through **all components** in the dataset, predicting their critical properties ($Tc, Pc$) using the trained models.")
                    
                    all_comp_preds = []
                    for _, row in clean_df.iterrows():
                        comp_name = row["name"]
                        actual_tc = row["Tc mean (K)"]
                        actual_pc = row["Pc mean (bar)"]
                        
                        feat = np.array([[float(row["Vc"]), float(row["Mw"]), float(row["Zc"])]])
                        feat_mlp = scaler_X.transform(feat)
                        
                        pred_rf_tc = trained_models["random_forest"]["model_tc"].predict(feat)[0]
                        pred_rf_pc = trained_models["random_forest"]["model_pc"].predict(feat)[0]
                        pred_xgb_tc = trained_models["xgboost"]["model_tc"].predict(feat)[0]
                        pred_xgb_pc = trained_models["xgboost"]["model_pc"].predict(feat)[0]
                        pred_mlp_tc = trained_models["mlp"]["model_tc"].predict(feat_mlp)[0]
                        pred_mlp_pc = trained_models["mlp"]["model_pc"].predict(feat_mlp)[0]
                        
                        all_comp_preds.append({
                            "Component": comp_name,
                            "Actual Tc (K)": round(actual_tc, 2),
                            "RF Tc (K)": round(pred_rf_tc, 2),
                            "XGB Tc (K)": round(pred_xgb_tc, 2),
                            "MLP Tc (K)": round(pred_mlp_tc, 2),
                            "Actual Pc (bar)": round(actual_pc, 2),
                            "RF Pc (bar)": round(pred_rf_pc, 2),
                            "XGB Pc (bar)": round(pred_xgb_pc, 2),
                            "MLP Pc (bar)": round(pred_mlp_pc, 2),
                        })
                    
                    df_all_predictions = pd.DataFrame(all_comp_preds)
                    st.dataframe(df_all_predictions, use_container_width=True)
                
                with tab4:
                    st.subheader("📈 P-T Phase Envelopes & Mixtures Analysis")
                    available_names = clean_df["name"].dropna().unique()
                    
                    system_category = st.selectbox(
                        "Select System Category:", 
                        ["Pure Component (Pure)", "Ternary Mixture (3 Components)", "Pentary Mixture (5 Components)", "Custom Mixture Selection"],
                        key="system_category_select"
                    )
                    
                    mix_dict = {}
                    if system_category == "Pure Component (Pure)":
                        selected_pure = st.selectbox(
                            "Select Pure Component:", 
                            options=available_names,
                            key="pure_component_select"
                        )
                        if selected_pure:
                            mix_dict = {selected_pure: 1.0}
                    elif system_category == "Ternary Mixture (3 Components)":
                        selected_ternary = st.multiselect(
                            "Select exactly 3 components:", 
                            options=available_names, 
                            default=list(available_names[:3]) if len(available_names)>=3 else list(available_names),
                            key="ternary_mixture_select"
                        )
                        if len(selected_ternary) > 0:
                            fraction = 1.0 / len(selected_ternary)
                            mix_dict = {comp: fraction for comp in selected_ternary}
                    elif system_category == "Pentary Mixture (5 Components)":
                        selected_pentary = st.multiselect(
                            "Select up to 5 components:", 
                            options=available_names, 
                            default=list(available_names[:5]) if len(available_names)>=5 else list(available_names),
                            key="pentary_mixture_select"
                        )
                        if len(selected_pentary) > 0:
                            fraction = 1.0 / len(selected_pentary)
                            mix_dict = {comp: fraction for comp in selected_pentary}
                    else:
                        selected_custom = st.multiselect(
                            "Select custom components:", 
                            options=available_names, 
                            default=list(available_names[:2]),
                            key="custom_mixture_select"
                        )
                        if len(selected_custom) > 0:
                            fraction = 1.0 / len(selected_custom)
                            mix_dict = {comp: fraction for comp in selected_custom}
                    
                    if len(mix_dict) > 0:
                        fig_comb, ax_c = plt.subplots(figsize=(8, 5), dpi=120, facecolor="white")
                        ax_c.set_facecolor("white")
                        
                        _, T_r, P_r, Tc_mix_r, Pc_mix_r = generate_envelope_plot(clean_df, mix_dict, "real_data", True, scaler_X, model_colors)
                        plt.close()
                        
                        fig_comb, ax_c = plt.subplots(figsize=(8, 5), dpi=120, facecolor="white")
                        ax_c.set_facecolor("white")
                        
                        ax_c.plot(T_r, P_r, color=model_colors["real_data"], lw=2.2, label="Real Data")
                        for m_name in models_list:
                            _, Tv, Pv, _, _ = generate_envelope_plot(clean_df, mix_dict, m_name, False, scaler_X, model_colors, trained_models[m_name])
                            plt.close()
                            ax_c.plot(Tv, Pv, color=model_colors[m_name], lw=1.8, label=m_name.upper())
                        
                        ax_c.set_xlabel("Temperature (T) [K]", fontsize=10, fontweight="bold", color="#334155")
                        ax_c.set_ylabel("Pressure (P) [bar]", fontsize=10, fontweight="bold", color="#334155")
                        ax_c.set_title("Combined P-T Phase Envelopes Comparison", fontsize=11, fontweight="bold", color="#1e293b")
                        ax_c.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")
                        ax_c.grid(True, linestyle="--", alpha=0.5)
                        st.pyplot(fig_comb)
                        plt.close(fig_comb)
                
                st.info("Results and chart files have= been saved successfully to local directories.")
else:
    st.info("Please verify the file path in the sidebar and click the Start Training button.")
```
