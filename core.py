import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from itertools import product
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
import streamlit as st
import openai
from openai import OpenAI

client = openai.OpenAI(api_key="API-KEY")

FEATURES = [
        'capacity_kw',
        'hub_height_m',
        'rotor_diameter_m',
        'swept_area_m2',
        'total_height_m',
        'rotor_to_height_ratio',
        'capacity_to_rotor_ratio',
        'capacity_zscore',
        'log_rotor_diameter'
    ]

# Cleaning
def clean_turbine_data(filepath='wind_turbines.csv', save_cleaned=True, save_path='cleaned_data.csv'):

    # Load the raw dataset
    df = pd.read_csv(filepath)
    print(f"Original dataset size: {len(df)} rows")

    # Drop unnecessary columns (e.g., coordinates)
    df = df.drop(columns=['Site.Latitude', 'Site.Longitude', 'Site.State', 'Site.County', 'Year'], errors='ignore')

    # Drop rows where key turbine data is missing
    df = df.dropna(subset=[
        'Turbine.Capacity',
        'Turbine.Hub_Height',
        'Turbine.Rotor_Diameter',
        'Turbine.Swept_Area',
        'Turbine.Total_Height'
    ])

    # Convert data types
    numeric_columns = [
        'Turbine.Capacity',
        'Turbine.Hub_Height',
        'Turbine.Rotor_Diameter',
        'Turbine.Swept_Area',
        'Turbine.Total_Height'
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Rename columns to snake_case
    df = df.rename(columns={
        'Turbine.Capacity': 'capacity_kw',
        'Turbine.Hub_Height': 'hub_height_m',
        'Turbine.Rotor_Diameter': 'rotor_diameter_m',
        'Turbine.Swept_Area': 'swept_area_m2',
        'Turbine.Total_Height': 'total_height_m',
        'Project.Capacity': 'project_capacity_kw',
        'Project.Number_Turbines': 'project_number_turbines'
    })


    # Save cleaned dataset
    if save_cleaned:
        df.to_csv(save_path, index=False)
        print(f"Cleaned dataset saved to '{save_path}'. Final size: {len(df)} rows")

    return df
# Feature engineering
def add_engineered_features(df, dropna: bool = True):
    df = df.copy()

    df["rotor_to_height_ratio"] = df["rotor_diameter_m"] / df["hub_height_m"]
    df["capacity_to_rotor_ratio"] = df["capacity_kw"] / df["rotor_diameter_m"]
    df["capacity_zscore"] = (df["capacity_kw"] - df["capacity_kw"].mean()) / (df["capacity_kw"].std() if df["capacity_kw"].std() != 0 else 1.0)
    df["log_rotor_diameter"] = np.log1p(df["rotor_diameter_m"])
    
    # Remove rows w/ missing values in those columns
    if dropna:
        df = df.dropna(subset=FEATURES)


    return df

# Anomaly injection
def inject_synthetic_anomalies(df: pd.DataFrame, anomaly_ratio: float = 0.05, verbose: bool = False) -> pd.DataFrame:

    df = df.copy()

    # Optional prints for data handling & checking columns
    if verbose:
        print(df.info())
        print(df.describe())

    # Add 5% synthetic anomalies
    num_anomalies = max(1, int(anomaly_ratio * len(df)))

    # Create copies of random valid rows to perturb
    anomalies = df.sample(n=num_anomalies, random_state=42).copy()

    # Tip height limit = 152m (set total_height_m > 152)
    anomalies['total_height_m'] = anomalies['total_height_m'].apply(
    lambda x: x + np.random.uniform(1, 5) + (152 - x) if x <= 152 else x
    )

    # Hub height limit ~107m
    anomalies['hub_height_m'] = anomalies['hub_height_m'].apply(
        lambda x: x + np.random.uniform(1, 5) + (107 - x) if x <= 107 else x
    )

    # Rotor diameter >150m as an anomaly
    anomalies['rotor_diameter_m'] = anomalies['rotor_diameter_m'].apply(
        lambda x: x + np.random.uniform(1, 5) + (150- x) if x <= 150 else x
    )

    # Inconsistent swept area: break the πD²/4 rule slightly
    anomalies['swept_area_m2'] = anomalies['rotor_diameter_m'] ** 2 * np.random.uniform(0.7, 0.9)

    # Label anomalies and normal data
    df['is_anomaly'] = 0
    anomalies['is_anomaly'] = 1
    
     # Combine both, then shuffle
    df_combined = pd.concat([df, anomalies], ignore_index=True).sample(frac=1, random_state=42)

    return df_combined

# Split dataset into train & test sets
def split_data(df: pd.DataFrame, features: list = FEATURES, test_size: float = 0.2, random_state: int = 42):
    X = df[features].copy()
    y = df["is_anomaly"] if "is_anomaly" in df.columns else None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test

# Visualize distributions
def visualize_feature_distributions(X: pd.DataFrame, use_streamlit: bool = False):
    
    # Histogram of feature distributions
    fig_hist = plt.figure(figsize=(12, 10))
    X.hist(figsize=(12, 10), bins=30)
    plt.suptitle("Distribution of Input Features", fontsize=16)
    plt.tight_layout()

    fig_box = plt.figure(figsize=(14, 6))
    sns.boxplot(data=X)
    plt.xticks(rotation=45)
    plt.title("Boxplot of Input Features")
    plt.tight_layout()

    # If requested, show in streamlit
    if use_streamlit:
        st.pyplot(fig_hist)
        st.pyplot(fig_box)
        plt.close(fig_hist)
        plt.close(fig_box)

    return fig_hist, fig_box


# Hyperparameter tuning & training 
def grid_search_isolation_forest(X_train, X_val, y_val, param_grid):
    param_combos = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())

    best_f1 = -1
    best_params = None


    # Try every combo of params
    for combo in param_combos:
        params = dict(zip(param_names, combo))

        model = IsolationForest(**params, random_state=42)
        model.fit(X_train)

        y_pred = model.predict(X_val)
        y_pred = (y_pred == -1)

        f1 = f1_score(y_val, y_pred)
        print(f"Params: {params}, F1 Score: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            
    print("\n Best params found:")
    print(best_params)
    print(f"Best F1 score: {best_f1:.4f}")

    return best_params
# Train final model w/ best params
def train_final_model(X_train, best_params):
    model = IsolationForest(**best_params, random_state=42)
    model.fit(X_train)

    return model

# Compute correlation btwn featurs & anomaly scores
def compute_feature_anomaly_correlation(model, X):

    scores = model.decision_function(X)
    score_series = pd.Series(scores, name="anomaly_score")
    correlations = X.corrwith(score_series)

    return correlations.abs().sort_values(ascending=False)

def show_top_correlated_features(correlations: pd.Series, n: int = 10, use_streamlit: bool = False):
    top = correlations.head(n).to_frame(name='abs_corr')
    if use_streamlit:
        st.write("### Top correlated features with anomaly score:")
        st.dataframe(top)
    else:
        print(top)
    return top

   

def plot_top_feature_relationships(X: pd.DataFrame, scores: np.ndarray, correlations: pd.Series, top_n: int = 3, use_streamlit: bool = False):
    # Direct relationship btwn top 3 features and anomaly score
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
    X_scaled["anomaly_score"] = scores

    figs = []
    for feature in correlations.head(top_n).index:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.regplot(data=X_scaled, x=feature, y="anomaly_score", lowess=True,
                    scatter_kws={"alpha": 0.3}, line_kws={"color": "red"}, ax=ax)
        ax.set_title(f"{feature} vs Anomaly Score")
        ax.set_xlabel(feature)
        ax.set_ylabel("Anomaly Score")
        plt.tight_layout()
        figs.append(fig)

        if use_streamlit:
            st.pyplot(fig)
            plt.close(fig)

    return figs


def plot_feature_correlation_heatmap(correlations: pd.Series, use_streamlit: bool = False):
    fig, ax = plt.subplots(figsize=(10, 2))
    sns.heatmap(correlations.to_frame().T, cmap="coolwarm", annot=True, ax=ax)
    ax.set_title("Correlation of Features with Anomaly Score")
    ax.set_yticks([])
    plt.tight_layout()

    if use_streamlit:
        st.pyplot(fig)
        plt.close(fig)

    return fig

# Compute mean and std for a numeric feature from training data
def get_training_stats(training_df: pd.DataFrame, feature: str = "capacity_kw"):
    mean = training_df[feature].mean()
    std = training_df[feature].std() if training_df[feature].std() != 0 else 1.0
    return mean, std

# Takes raw turbine metrics and returns a cleaned, feature-engineered DataFrame
def prepare_input_data(capacity: float,
                       hub_height: float,
                       rotor_diameter: float,
                       swept_area: float,
                       total_height: float,
                       training_mean: float = None,
                       training_std: float = None,
                       project_number_turbines: int = None) -> pd.DataFrame:


    if training_mean is None or training_std is None:
         warnings.warn("training_mean/training_std not provided — capacity_zscore set to 0. For best results, pass training stats.", UserWarning)

    rotor_to_height_ratio = rotor_diameter / hub_height if hub_height != 0 else np.nan
    capacity_to_rotor_ratio = capacity / rotor_diameter if rotor_diameter != 0 else np.nan
    capacity_zscore = ((capacity - training_mean) / training_std) if (training_mean is not None and training_std is not None) else 0.0
    log_rotor_diameter = np.log1p(rotor_diameter)

    data = {
        "capacity_kw": [capacity],  
        "hub_height_m": [hub_height],
        "rotor_diameter_m": [rotor_diameter],
        "swept_area_m2": [swept_area],
        "total_height_m": [total_height],
        "rotor_to_height_ratio": [rotor_to_height_ratio],
        "capacity_to_rotor_ratio": [capacity_to_rotor_ratio],
        "capacity_zscore": [capacity_zscore],
        "log_rotor_diameter": [log_rotor_diameter]
    }

    row = pd.DataFrame(data)

    # Ensure column order matches FEATURES exactly
    return row.reindex(columns=FEATURES)

# Runs anomaly model & returns prediction (1 = normal, -1 = anomaly)
def predict_single_turbine(model: IsolationForest, turbine_df: pd.DataFrame, features: list = FEATURES) -> int:
    preds = model.predict(turbine_df[features])
    # If single row, return scalar
    if len(preds) == 1:
        return int(preds[0])
    return preds


# Formats a report
def summarize_prediction(turbine_df: pd.DataFrame, prediction: int, include_project_fields: bool = False) -> str:
    pred_label = "Normal" if int(prediction) == 1 else "Anomaly"
    lines = [
        f"Wind Turbine Prediction Report:",
        f"- Capacity: {turbine_df.get('capacity_kw', pd.Series([np.nan]))[0]} kW",
        f"- Hub Height: {turbine_df.get('hub_height_m', pd.Series([np.nan]))[0]} m",
        f"- Rotor Diameter: {turbine_df.get('rotor_diameter_m', pd.Series([np.nan]))[0]} m",
        f"- Swept Area: {turbine_df.get('swept_area_m2', pd.Series([np.nan]))[0]} m²",
        f"- Total Height: {turbine_df.get('total_height_m', pd.Series([np.nan]))[0]} m",
    ]

    if include_project_fields:
        if 'project_capacity_kw' in turbine_df.columns:
            lines.append(f"- Project Capacity: {turbine_df['project_capacity_kw'][0]} kW")
        if 'project_number_turbines' in turbine_df.columns:
            lines.append(f"- Num Turbines: {turbine_df['project_number_turbines'][0]}")

    lines += [
        f"- Rotor/Height Ratio: {turbine_df.get('rotor_to_height_ratio', pd.Series([np.nan]))[0]:.2f}",
        f"- Capacity/Rotor Ratio: {turbine_df.get('capacity_to_rotor_ratio', pd.Series([np.nan]))[0]:.2f}",
        f"- Log Rotor Diameter: {turbine_df.get('log_rotor_diameter', pd.Series([np.nan]))[0]:.2f}",
        f"- Prediction: {pred_label}"
    ]
    return "\n".join(lines)

def evaluate_predictions(y_true, y_pred):
    """
    Compute precision, recall, f1 and confusion matrix for binary anomaly labels.
    y_pred should be binary (True anomaly / False normal) or 0/1; convert if needed.
    """
    # Accept y_pred as -1/1 or boolean or 0/1
    # Convert to boolean anomaly flag
    y_pred_bool = None
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # If predictions are -1/1
    if set(np.unique(y_pred_arr)).issubset({-1, 1}):
        y_pred_bool = (y_pred_arr == -1)
    else:
        y_pred_bool = y_pred_arr.astype(bool)

    precision = precision_score(y_true_arr, y_pred_bool)
    recall = recall_score(y_true_arr, y_pred_bool)
    f1 = f1_score(y_true_arr, y_pred_bool)
    cm = confusion_matrix(y_true_arr, y_pred_bool)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }

# Manual checks
def check_design_violations(capacity_kw, hub_height_m, rotor_diameter_m, swept_area_m2, total_height_m):
    violations = []

    # Rule 1: Total height < 152 m
    if total_height_m >= 152:
        violations.append(f"Total height is {total_height_m}m, which exceeds the 152m limit.")

    # Rule 2: Hub height < ~107 m
    if hub_height_m >= 107:
        violations.append(f"Hub height is {hub_height_m}m, which exceeds the recommended maximum of ~107m.")

    # Rule 3: Rotor diameter > 150 m as anomaly
    if rotor_diameter_m > 150:
        violations.append(f"Rotor diameter is {rotor_diameter_m}m, which is unusually large and may be anomalous.")

    # Rule 4: Check swept area consistency (πD²/4)
    expected_swept_area = (3.1416 * rotor_diameter_m**2) / 4
    if abs(swept_area_m2 - expected_swept_area) > 0.1 * expected_swept_area:  # >10% deviation
        violations.append(
            f"Swept area {swept_area_m2}m² is inconsistent with rotor diameter {rotor_diameter_m}m (expected ~{expected_swept_area:.2f}m²)."
        )

    return violations


# Use OpenAI to generate NL summary
def get_chatgpt_summary(input_data, prediction, anomaly_score, violations=None):
    input_dict = input_data.to_dict()
    status = "anomalous" if prediction == -1 else "normal"
    
    violation_text = ""
    if violations:
        violation_text = "\nThe following engineering rule violations were detected:\n" + "\n".join(f"- {v}" for v in violations)

    prompt = f"""
You are an expert in wind turbine analytics. Given the following turbine input data and its model prediction, summarize the result in a human-readable format.

Turbine Input: {input_dict}
Model Prediction: {status.upper()}
Anomaly Score: {anomaly_score:.4f}
{violation_text}

Write a concise and accurate paragraph summarizing the turbine's status. If there are engineering rule violations, mention how they might relate to the prediction.
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You generate clear, expert-level summaries of anomaly detection results in wind turbine data."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()



def run_test_pipeline():
    # 1. Load and clean
    df = clean_turbine_data(filepath="wind_turbines.csv", save_cleaned=False)
    
    # 2. Feature engineering
    df = add_engineered_features(df)

    # 3. Inject anomalies
    df = inject_synthetic_anomalies(df, anomaly_ratio=0.05)

    # 4. Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # 5. Define hyperparameter grid
    param_grid = {
        "n_estimators": [100],
        "max_samples": ["auto"],
        "contamination": [0.05],  # match synthetic anomaly ratio
        "max_features": [1.0]
    }

    # 6. Tune model
    best_params = grid_search_isolation_forest(X_train, X_test, y_test, param_grid)

    # 7. Train final model
    model = train_final_model(X_train, best_params)

    # 8. Evaluate
    y_pred = model.predict(X_test)
    evaluation = evaluate_predictions(y_test, y_pred)
    print("\nEvaluation Metrics:")
    print(evaluation)

    # 9. Feature importance
    correlations = compute_feature_anomaly_correlation(model, X_train)
    show_top_correlated_features(correlations)
    
    # 10. Predict for a single custom turbine
    mean, std = get_training_stats(df)  # needed for z-scoring and log transform

    capacity = 2000
    hub_height = 90
    rotor_diameter = 100
    swept_area = 7850
    total_height = 140


    turbine_input = prepare_input_data(
        capacity=capacity, 
        hub_height=hub_height, 
        rotor_diameter=rotor_diameter, 
        swept_area=swept_area, 
        total_height=total_height,
        training_mean=mean, 
        training_std=std

    )

    prediction = model.predict(turbine_input)[0]
    anomaly_score = model.decision_function(turbine_input)[0]
    print("Prediction:", "Anomaly" if prediction == -1 else "Normal")

    violations = check_design_violations(
        capacity_kw=capacity,
        hub_height_m=hub_height,
        rotor_diameter_m=rotor_diameter,
        swept_area_m2=swept_area,
        total_height_m=total_height
    )


    # Get ChatGPT-style summary
    summary = get_chatgpt_summary(turbine_input, prediction, anomaly_score, violations)

    print("\nAI Summary of Turbine Prediction:")
    print(summary)

# Run the test
run_test_pipeline()
