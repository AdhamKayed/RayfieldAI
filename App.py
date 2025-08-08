import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# === Import your pipeline bits from your code file ===
# Save your long script as core.py next to this app.
from core import (
    FEATURES,
    clean_turbine_data,
    add_engineered_features,
    inject_synthetic_anomalies,
    split_data,
    grid_search_isolation_forest,
    train_final_model,
    compute_feature_anomaly_correlation,
    plot_top_feature_relationships,
    plot_feature_correlation_heatmap,
    get_training_stats,
    prepare_input_data,
    predict_single_turbine,
    summarize_prediction,
    check_design_violations,
)

st.set_page_config(page_title="Turbine AI Dashboard", layout="wide")
st.title("🌀 Turbine Anomaly Detection Dashboard")

# ---------------- Sidebar Controls ----------------
with st.sidebar:
    st.header("⚙️ Controls")

    st.markdown("**Data Source**")
    uploaded = st.file_uploader("Upload raw CSV (e.g., wind_turbines.csv)", type=["csv"])

    st.markdown("---")
    st.markdown("**Preprocessing**")
    use_cleaner = st.checkbox("Run cleaning (drop & rename columns)", value=True)
    do_feature_eng = st.checkbox("Add engineered features", value=True)

    st.markdown("---")
    st.markdown("**Anomaly Injection (for testing & tuning)**")
    do_inject = st.checkbox("Inject synthetic anomalies", value=False)
    anomaly_ratio = st.slider("Anomaly ratio", 0.0, 0.25, 0.05, 0.01)

    st.markdown("---")
    st.markdown("**Model & Tuning**")
    do_tune = st.checkbox("Grid search tuning", value=True)
    contamination = st.slider("Default contamination (if not tuning)", 0.01, 0.2, 0.05, 0.01)

    st.markdown("---")
    st.markdown("**Exports**")
    export_alerts = st.checkbox("Export alerts_today.csv on anomalies", value=True)

# ---------------- Helpers ----------------
def _mock_summary(df):
    avg_capacity = df["capacity_kw"].mean() if "capacity_kw" in df.columns else df.get("turbine_capacity_mw", pd.Series([np.nan])).mean()
    peak_rotor = df["rotor_diameter_m"].max() if "rotor_diameter_m" in df.columns else np.nan
    anomalies = df[df["is_anomaly"].eq(1) | df["anomaly"].eq(True)] if set(["is_anomaly","anomaly"]).issubset(df.columns.union(["is_anomaly","anomaly"])) else pd.DataFrame()
    dates = None
    if "date" in df.columns and not anomalies.empty:
        try:
            dates = pd.to_datetime(anomalies["date"]).dt.strftime("%b %d").tolist()
        except Exception:
            dates = anomalies["date"].astype(str).tolist()
    return (
        f"Avg Capacity ~ {avg_capacity:.2f} | Peak Rotor ~ {peak_rotor:.2f} m | "
        f"Anomalies: {', '.join(dates) if dates else 'none'}"
    )

def _safe_display_df(df, label="Preview"):
    st.markdown(f"### {label}")
    st.dataframe(df.head(20))

# ---------------- Main Flow ----------------
tabs = st.tabs(["📥 Data", "🔎 Explore", "🧠 Train & Tune", "🧪 Single Turbine", "🚨 Alerts & Export"])

if uploaded:
    # Read uploaded bytes -> CSV
    raw = pd.read_csv(uploaded)
    with tabs[0]:
        _safe_display_df(raw, "Raw Data (top 20)")

    df = raw.copy()

    # Normalize your expected column names: your cleaner expects specific headers.
    if use_cleaner:
        # If the uploaded file is the original dataset with dots in names
        # write to a temp file so your cleaner can read it
        tmp_path = "uploaded_input.csv"
        raw.to_csv(tmp_path, index=False)
        df = clean_turbine_data(filepath=tmp_path, save_cleaned=False)

    # Feature engineering (adds ratios/zscores/logs)
    if do_feature_eng:
        df = add_engineered_features(df, dropna=True)

    # Optional: inject synthetic anomalies for evaluation
    if do_inject:
        df = inject_synthetic_anomalies(df, anomaly_ratio=anomaly_ratio, verbose=False)

    # Keep a preview
    with tabs[0]:
        _safe_display_df(df, "Processed Data (top 20)")
        st.success("Data processing complete.")

    # ---------------- EDA ----------------
    with tabs[1]:
        st.subheader("Distributions")
        fig1, ax = plt.subplots(figsize=(10, 6))
        # plot numeric hist quickly
        df.select_dtypes(include=[np.number]).hist(ax=ax)
        st.pyplot(fig1)
        plt.close(fig1)

        # seaborn pairplot can be heavy; show a heatmap of correlations instead
        st.subheader("Feature Correlation (numeric)")
        if not df.select_dtypes(include=[np.number]).empty:
            corr = df.select_dtypes(include=[np.number]).corr()
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax2)
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.info("No numeric columns found for correlation heatmap.")

    # ---------------- Train & Tune ----------------
    with tabs[2]:
        st.subheader("Training")
        # Must have FEATURES present
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing required feature columns: {missing}. Make sure cleaning + feature engineering ran.")
        else:
            # Split
            try:
                X_train, X_test, y_train, y_test = split_data(df, features=FEATURES, test_size=0.2, random_state=42)
            except ValueError:
                # If no labels (is_anomaly) present, create a dummy normal label to allow tuning preview
                df["is_anomaly"] = 0
                X_train, X_test, y_train, y_test = split_data(df, features=FEATURES, test_size=0.2, random_state=42)

            # Tuning
            if do_tune:
                st.write("Running grid search (this may take a few seconds)...")
                param_grid = {
                    "n_estimators": [100],
                    "max_samples": ["auto"],
                    "contamination": [anomaly_ratio if do_inject else contamination],
                    "max_features": [1.0]
                }
                best_params = grid_search_isolation_forest(X_train, X_test, y_test, param_grid)
            else:
                best_params = {
                    "n_estimators": 100,
                    "max_samples": "auto",
                    "contamination": contamination,
                    "max_features": 1.0
                }

            st.success(f"Best params: {best_params}")

            # Train final
            model = train_final_model(X_train, best_params)
            st.session_state["model"] = model  # stash for later

            # Correlation with anomaly score
            st.subheader("Feature Importance (correlation vs anomaly score)")
            corrs = compute_feature_anomaly_correlation(model, X_train)
            st.write(corrs.sort_values(ascending=False).to_frame("abs_corr").head(10))

            # Plots: top relationships
            try:
                scores = model.decision_function(X_train)
                figs = plot_top_feature_relationships(X_train, scores, corrs, top_n=3, use_streamlit=False)
                for f in figs:
                    st.pyplot(f)
                    plt.close(f)
            except Exception as e:
                st.info(f"Could not plot top relationships: {e}")

            # Heatmap
            try:
                fig_hm = plot_feature_correlation_heatmap(corrs, use_streamlit=False)
                st.pyplot(fig_hm)
                plt.close(fig_hm)
            except Exception as e:
                st.info(f"Could not plot heatmap: {e}")

    # ---------------- Single Turbine What‑If ----------------
    with tabs[3]:
        st.subheader("Single Turbine Prediction")

        # Training stats for z-score
        mean, std = get_training_stats(df, feature="capacity_kw" if "capacity_kw" in df.columns else FEATURES[0])

        c1, c2, c3 = st.columns(3)
        with c1:
            capacity = st.number_input("Capacity (kW)", min_value=0.0, value=2000.0, step=100.0)
            hub_height = st.number_input("Hub Height (m)", min_value=0.0, value=90.0, step=1.0)
            rotor_diameter = st.number_input("Rotor Diameter (m)", min_value=0.0, value=100.0, step=1.0)
        with c2:
            swept_area = st.number_input("Swept Area (m²)", min_value=0.0, value=7850.0, step=10.0)
            total_height = st.number_input("Total Height (m)", min_value=0.0, value=140.0, step=1.0)
        with c3:
            include_project = st.checkbox("Include project fields in report", value=False)

        if st.button("Predict"):
            row = prepare_input_data(
                capacity=capacity,
                hub_height=hub_height,
                rotor_diameter=rotor_diameter,
                swept_area=swept_area,
                total_height=total_height,
                training_mean=mean,
                training_std=std
            )

            model = st.session_state.get("model")
            if model is None:
                st.error("Train a model first in the 'Train & Tune' tab.")
            else:
                pred = predict_single_turbine(model, row, features=FEATURES)
                score = float(model.decision_function(row[FEATURES])[0])
                violations = check_design_violations(capacity, hub_height, rotor_diameter, swept_area, total_height)
                report = summarize_prediction(row, pred, include_project_fields=include_project)

                st.markdown("**Prediction**")
                st.info("Anomaly" if pred == -1 else "Normal")
                st.markdown(f"**Anomaly score:** {score:.4f}")
                if violations:
                    st.warning("Engineering rule flags:\n- " + "\n- ".join(violations))
                st.text(report)

                # Mock summary paragraph (no API dependency)
                st.markdown("**Summary Paragraph**")
                st.write(_mock_summary(pd.concat([df.head(1), row], ignore_index=True)))

    # ---------------- Alerts & Export ----------------
    with tabs[4]:
        st.subheader("Alerts")
        # If you injected anomalies, you'll have 'is_anomaly'; if not, you may not have predictions on full df.
        # We’ll just export rows where your pipeline says anomaly==1/True if present
        alert_df = None
        if "is_anomaly" in df.columns:
            alert_df = df[df["is_anomaly"] == 1].copy()
        elif "anomaly" in df.columns:
            alert_df = df[df["anomaly"] == True].copy()

        if alert_df is not None and not alert_df.empty:
            keep_cols = [c for c in ["date", "capacity_kw", "turbine_capacity_mw"] if c in alert_df.columns]
            if not keep_cols:
                keep_cols = alert_df.columns.tolist()[:5]
            st.dataframe(alert_df[keep_cols].head(50))
            if export_alerts:
                alert_df.to_csv("alerts_today.csv", index=False)
                st.success("Exported alerts_today.csv")
                st.download_button("Download alerts_today.csv", data=alert_df.to_csv(index=False), file_name="alerts_today.csv", mime="text/csv")
        else:
            st.info("No anomalies found to export yet.")

else:
    st.info("Upload a CSV in the sidebar to begin.")
