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
    page_title="Thermal Properties Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #fcfcfc;
        color: #1e1e1e;
    }
    [data-testid="stSidebar"] {
        background-color: #f4f5f7;
        border-right: 1px solid #e0e0e0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
        color: #333333;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b !important;
        color: white !important;
        border-color: #ff4b4b !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Thermal Properties Analysis & Critical Point Prediction Dashboard")
st.markdown("Advanced machine learning models for thermodynamic property evaluation and phase envelope generation.")

st.sidebar.header("⚙️ Execution Settings")
csv_file_path = st.sidebar.text_input("CSV File Path", value="critprop_merged_final.csv")
run_button = st.sidebar.button("⚡ Fast Load & Analyze", type="primary")

st.sidebar.markdown("---")
st.sidebar.subheader("📱 App QR Code")
app_url = "https://critical-point-prediction-dashboard-lyecr5liams58tpxqqcub3.streamlit.app"

qr = qrcode.QRCode(version=1, box_size=10, border=4)
qr.add_data(app_url)
qr.make(fit=True)
img_qr = qr.make_image(fill_color="black", back_color="white")

buf = io.BytesIO()
img_qr.save(buf, format="PNG")
st.sidebar.image(buf.getvalue(), caption="Scan to open app", use_container_width=True)

@st.cache_data
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    return df.dropna(subset=["Vc", "Mw", "Zc", "Tc mean (K)", "Pc mean (bar)"])

@st.cache_resource
def get_trained_models(file_path):
    model_path = "saved_models_pipeline.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    
    clean_df = load_and_clean_data(file_path)
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
        
        if m_name == "random_forest":
            model_tc = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
            model_pc = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
        elif m_name == "decision_tree":
            model_tc = DecisionTreeRegressor(max_depth=5, random_state=42)
            model_pc = DecisionTreeRegressor(max_depth=5, random_state=42)
        elif m_name == "mlp":
            model_tc = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=50, random_state=42)
            model_pc = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=50, random_state=42)
        elif m_name == "xgboost":
            model_tc = XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
            model_pc = XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
            
        model_tc.fit(X_tr, y_train_tc)
        model_pc.fit(X_tr, y_train_pc)
        
        trained_models[m_name] = {"model_tc": model_tc, "model_pc": model_pc}
        
        pred_train_tc = model_tc.predict(X_tr)
        pred_test_tc = model_tc.predict(X_te)
        pred_train_pc = model_pc.predict(X_tr)
        pred_test_pc = model_pc.predict(X_te)
        
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

        fig_loss, axes_loss = plt.subplots(1, 2, figsize=(8, 3), dpi=100, facecolor="white")
        axes_loss[0].set_facecolor("white")
        axes_loss[0].plot([mean_squared_error(y_train_tc, pred_train_tc)], label="Train", marker='o', color="#ff4b4b")
        axes_loss[0].set_title(f"{m_name.upper()} Tc Loss", fontsize=9)
        axes_loss[1].set_facecolor("white")
        axes_loss[1].plot([mean_squared_error(y_train_pc, pred_train_pc)], label="Train", marker='o', color="#ff4b4b")
        axes_loss[1].set_title(f"{m_name.upper()} Pc Loss", fontsize=9)
        plt.tight_layout()
        loss_figs[m_name] = fig_loss

    metrics_df = pd.DataFrame(metrics_results)
    package = (clean_df, trained_models, scaler_X, metrics_df, loss_figs, models_list)
    joblib.dump(package, model_path)
    return package

def generate_envelope_plot(df, mixture_composition, model_key, is_real_data, scaler_X, model_colors, model_obj_dict=None):
    comp_names = list(mixture_composition.keys())
    fractions = np.array(list(mixture_composition.values()), dtype=float)
    fractions /= np.sum(fractions)
    
    mix_data = []
    for comp in comp_names:
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

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100, facecolor="white")
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
    ax.set_xlabel("Temperature (T) [K]", fontsize=9, fontweight="bold", color="#333333")
    ax.set_ylabel("Pressure (P) [bar]", fontsize=9, fontweight="bold", color="#333333")
    
    curve_color = model_colors.get(model_key, "#1b40ff")
    ax.plot(T_vals, P_vals, color=curve_color, lw=2.0)
    ax.plot(T_cp_val, P_cp_val, marker="o", markersize=6, markerfacecolor="#ff4b4b", markeredgecolor="none")
    
    comp_str = ", ".join([f"{row['zi']*100:.0f}% {row['name']}" for _, row in df_mix.iterrows()])
    display_name = "REAL DATA" if is_real_data else model_key.upper()
    ax.set_title(f"P-T Envelope: {display_name}\n({comp_str})\nTc = {Tc_mix:.1f} K, Pc = {Pc_mix:.1f} bar", fontsize=8, color="#1e1e1e")
    
    return fig, T_vals, P_vals, Tc_mix, Pc_mix

if run_button or st.session_state.get("trained", True):
    st.session_state["trained"] = True
    if not os.path.exists(csv_file_path):
        st.error(f"File not found at: {csv_file_path}")
    else:
        with st.spinner("⚡ Loading application instantly..."):
            clean_df, trained_models, scaler_X, metrics_df, loss_figs, models_list = get_trained_models(csv_file_path)
            model_colors = {
                "real_data": "#333333",    
                "random_forest": "#1b40ff",
                "decision_tree": "#28a745",
                "mlp": "#ff8c00",
                "xgboost": "#dc3545"
            }
            
            fig_orig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=100, facecolor="white")
            axes[0].set_facecolor("white")
            axes[0].scatter(clean_df["Vc"], clean_df["Tc mean (K)"], color="#764ba2", alpha=0.6, edgecolors="k")
            axes[0].set_title("Vc vs Tc", fontsize=10, fontweight="bold")
            axes[0].grid(True, linestyle="--", alpha=0.5)
            
            axes[1].set_facecolor("white")
            axes[1].scatter(clean_df["Vc"], clean_df["Pc mean (bar)"], color="#0083b0", alpha=0.6, edgecolors="k")
            axes[1].set_title("Vc vs Pc", fontsize=10, fontweight="bold")
            axes[1].grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            
            st.subheader("📈 Original Data Distribution")
            st.pyplot(fig_orig)
            plt.close(fig_orig)
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Performance", 
                "📉 Loss Curves", 
                "🌐 Predictions", 
                "📈 Phase Envelopes"
            ])
            
            with tab1:
                st.subheader("📋 Model Performance Summary")
                st.dataframe(metrics_df, use_container_width=True)
            
            with tab2:
                st.subheader("📉 Loss Curves per Model")
                cols = st.columns(2)
                for i, m_name in enumerate(models_list):
                    with cols[i%2]:
                        st.pyplot(loss_figs[m_name])
                        plt.close(loss_figs[m_name])
            
            with tab3:
                st.subheader("🌐 All Components Predictions")
                all_comp_preds = []
                for _, row in clean_df.iterrows():
                    comp_name = row["name"]
                    feat = np.array([[float(row["Vc"]), float(row["Mw"]), float(row["Zc"])]])
                    feat_mlp = scaler_X.transform(feat)
                    
                    all_comp_preds.append({
                        "Component": comp_name,
                        "Actual Tc": round(row["Tc mean (K)"], 2),
                        "RF Tc": round(trained_models["random_forest"]["model_tc"].predict(feat)[0], 2),
                        "XGB Tc": round(trained_models["xgboost"]["model_tc"].predict(feat)[0], 2),
                        "MLP Tc": round(trained_models["mlp"]["model_tc"].predict(feat_mlp)[0], 2),
                        "Actual Pc": round(row["Pc mean (bar)"], 2),
                        "RF Pc": round(trained_models["random_forest"]["model_pc"].predict(feat)[0], 2),
                        "XGB Pc": round(trained_models["xgboost"]["model_pc"].predict(feat)[0], 2),
                        "MLP Pc": round(trained_models["mlp"]["model_pc"].predict(feat_mlp)[0], 2),
                    })
                st.dataframe(pd.DataFrame(all_comp_preds), use_container_width=True)
            
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
                    selected_pure = st.selectbox("Select Pure Component:", options=available_names, key="pure_component_select")
                    if selected_pure:
                        mix_dict = {selected_pure: 1.0}
                elif system_category == "Ternary Mixture (3 Components)":
                    selected_ternary = st.multiselect("Select exactly 3 components:", options=available_names, default=list(available_names[:3]) if len(available_names)>=3 else list(available_names), key="ternary_mixture_select")
                    if len(selected_ternary) > 0:
                        mix_dict = {comp: 1.0 / len(selected_ternary) for comp in selected_ternary}
                elif system_category == "Pentary Mixture (5 Components)":
                    selected_pentary = st.multiselect("Select up to 5 components:", options=available_names, default=list(available_names[:5]) if len(available_names)>=5 else list(available_names), key="pentary_mixture_select")
                    if len(selected_pentary) > 0:
                        mix_dict = {comp: 1.0 / len(selected_pentary) for comp in selected_pentary}
                else:
                    selected_custom = st.multiselect("Select custom components:", options=available_names, default=list(available_names[:2]), key="custom_mixture_select")
                    if len(selected_custom) > 0:
                        mix_dict = {comp: 1.0 / len(selected_custom) for comp in selected_custom}
                
                if len(mix_dict) > 0:
                    fig_comb, ax_c = plt.subplots(figsize=(7, 4), dpi=120, facecolor="white")
                    ax_c.set_facecolor("white")
                    
                    _, T_r, P_r, _, _ = generate_envelope_plot(clean_df, mix_dict, "real_data", True, scaler_X, model_colors)
                    plt.close()
                    
                    fig_comb, ax_c = plt.subplots(figsize=(7, 4), dpi=120, facecolor="white")
                    ax_c.set_facecolor("white")
                    
                    ax_c.plot(T_r, P_r, color=model_colors["real_data"], lw=2.2, label="Real Data")
                    for m_name in models_list:
                        _, Tv, Pv, _, _ = generate_envelope_plot(clean_df, mix_dict, m_name, False, scaler_X, model_colors, trained_models[m_name])
                        plt.close()
                        ax_c.plot(Tv, Pv, color=model_colors[m_name], lw=1.8, label=m_name.upper())
                    
                    ax_c.set_xlabel("Temperature (T) [K]", fontsize=9, fontweight="bold", color="#333333")
                    ax_c.set_ylabel("Pressure (P) [bar]", fontsize=9, fontweight="bold", color="#333333")
                    ax_c.set_title("Combined P-T Phase Envelopes Comparison", fontsize=10, fontweight="bold", color="#1e1e1e")
                    ax_c.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8)
                    ax_c.grid(True, linestyle="--", alpha=0.5)
                    st.pyplot(fig_comb)
                    plt.close(fig_comb)
