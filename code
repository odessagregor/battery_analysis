
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80, flush=True)
print("BATTERY CAPACITY PREDICTION - ML ANALYSIS", flush=True)
print("="*80, flush=True)

print("\n[1] Loading and Exploring Data...", flush=True)
df = pd.read_csv("metadata.csv")
print(f"Dataset shape: {df.shape}", flush=True)
print(f"\nTest types distribution:\n{df['type'].value_counts()}", flush=True)

print("\n[2] Preprocessing Data...", flush=True)

#create numeric columns that are stored as objects
df['Capacity'] = pd.to_numeric(df['Capacity'], errors='coerce')
df['Re'] = pd.to_numeric(df['Re'], errors='coerce')
df['Rct'] = pd.to_numeric(df['Rct'], errors='coerce')

#change start_time to datetime
#use test id for ordering (parsing is slow) as it is chronological

print("\n[3] Engineering Features...", flush=True)

#predict capacity using impedance measurements
#aggregate impedance data and merge with capacity data for each test id

#separate the data based on type
impedance_df = df[df['type'] == 'impedance'][['battery_id', 'test_id', 'ambient_temperature',
                                                'Re', 'Rct']].copy()
discharge_df = df[df['type'] == 'discharge'][['battery_id', 'test_id', 'ambient_temperature',
                                                'Capacity']].copy()

#calculate cycle number based on test id order for every battery
discharge_df = discharge_df.sort_values(['battery_id', 'test_id'])
discharge_df['cycle_number'] = discharge_df.groupby('battery_id').cumcount() + 1

impedance_df = impedance_df.sort_values(['battery_id', 'test_id'])
impedance_df['impedance_measurement_num'] = impedance_df.groupby('battery_id').cumcount() + 1

#merge impedance with discharge data
#strategy: merge all data and forward fill impedance values
all_data = pd.concat([
    discharge_df.assign(type='discharge'),
    impedance_df.assign(type='impedance')
], ignore_index=True)

#sort by battery and test id
all_data = all_data.sort_values(['battery_id', 'test_id'])

#forward fill impedance measurements within each battery
all_data[['Re', 'Rct']] = all_data.groupby('battery_id')[['Re', 'Rct']].ffill()

#keep only discharge rows
merged_df = all_data[all_data['Capacity'].notna()].copy()

#drop rows with missing values in key columns
merged_df = merged_df.dropna(subset=['Capacity', 'Re', 'Rct'])

print(f"Merged dataset shape: {merged_df.shape}", flush=True)
print(f"Features available: {merged_df.columns.tolist()}", flush=True)

#create additional features
merged_df['total_impedance'] = merged_df['Re'] + merged_df['Rct']
merged_df['impedance_ratio'] = merged_df['Rct'] / (merged_df['Re'] + 1e-10)

#calculate capacity degradation (capacity loss from first cycle)
merged_df['initial_capacity'] = merged_df.groupby('battery_id')['Capacity'].transform('first')
merged_df['capacity_degradation'] = merged_df['initial_capacity'] - merged_df['Capacity']
merged_df['capacity_retention_pct'] = (merged_df['Capacity'] / merged_df['initial_capacity']) * 100

print(f"\nCapacity statistics:", flush=True)
print(merged_df['Capacity'].describe(), flush=True)
print(f"\nCapacity degradation statistics:", flush=True)
print(merged_df['capacity_degradation'].describe(), flush=True)

print("\n[4] Preparing Features for Machine Learning...", flush=True)

#select features for modeling
feature_cols = ['Re', 'Rct', 'total_impedance', 'impedance_ratio',
                'ambient_temperature', 'cycle_number']
X = merged_df[feature_cols].copy()
y = merged_df['Capacity'].copy()

print(f"Feature matrix shape: {X.shape}", flush=True)
print(f"Target variable shape: {y.shape}", flush=True)
print(f"\nFeature correlations with Capacity:", flush=True)
correlations = merged_df[feature_cols + ['Capacity']].corr()['Capacity'].sort_values(ascending=False)
print(correlations, flush=True)

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#train models
print("\n[5] Training Machine Learning Models...", flush=True)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...", flush=True)

    # Train model
    if name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    #evaluate results 
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        'model': model,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred
    }

    print(f"  RMSE: {rmse:.4f}", flush=True)
    print(f"  MAE: {mae:.4f}", flush=True)
    print(f"  R² Score: {r2:.4f}", flush=True)

#choose best model
best_model_name = max(results, key=lambda x: results[x]['r2'])
best_model = results[best_model_name]['model']
print(f"\n{'='*80}", flush=True)
print(f"BEST MODEL: {best_model_name} (R² = {results[best_model_name]['r2']:.4f})", flush=True)
print(f"{'='*80}", flush=True)

#feature importance
if hasattr(best_model, 'feature_importances_'):
    print("\nFeature Importance:", flush=True)
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.to_string(index=False), flush=True)

#plots
print("\n[6] Creating Visualizations...", flush=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Battery Capacity Prediction - ML Analysis', fontsize=16, fontweight='bold')

# 1. Actual vs Predicted
ax = axes[0, 0]
y_pred_best = results[best_model_name]['predictions']
ax.scatter(y_test, y_pred_best, alpha=0.6, edgecolors='k', linewidth=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Capacity')
ax.set_ylabel('Predicted Capacity')
ax.set_title(f'Actual vs Predicted Capacity\n{best_model_name} (R²={results[best_model_name]["r2"]:.3f})')
ax.grid(True, alpha=0.3)

# 2. Residuals
ax = axes[0, 1]
residuals = y_test - y_pred_best
ax.scatter(y_pred_best, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Predicted Capacity')
ax.set_ylabel('Residuals')
ax.set_title('Residual Plot')
ax.grid(True, alpha=0.3)

# 3. Model Comparison
ax = axes[0, 2]
model_names = list(results.keys())
r2_scores = [results[m]['r2'] for m in model_names]
colors = ['green' if m == best_model_name else 'skyblue' for m in model_names]
bars = ax.bar(model_names, r2_scores, color=colors, edgecolor='black')
ax.set_ylabel('R² Score')
ax.set_title('Model Performance Comparison')
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')
for bar, score in zip(bars, r2_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

# 4. Capacity vs Cycle Number by Temperature
ax = axes[1, 0]
for temp in sorted(merged_df['ambient_temperature'].unique()):
    temp_data = merged_df[merged_df['ambient_temperature'] == temp]
    ax.scatter(temp_data['cycle_number'], temp_data['Capacity'],
              label=f'{temp}°C', alpha=0.6, s=30)
ax.set_xlabel('Cycle Number')
ax.set_ylabel('Capacity')
ax.set_title('Battery Capacity Degradation Over Cycles')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. Impedance vs Capacity
ax = axes[1, 1]
scatter = ax.scatter(merged_df['total_impedance'], merged_df['Capacity'],
                    c=merged_df['cycle_number'], cmap='viridis',
                    alpha=0.6, edgecolors='k', linewidth=0.5)
ax.set_xlabel('Total Impedance (Re + Rct)')
ax.set_ylabel('Capacity')
ax.set_title('Capacity vs Total Impedance')
plt.colorbar(scatter, ax=ax, label='Cycle Number')
ax.grid(True, alpha=0.3)

# 6. Feature Importance (if available)
ax = axes[1, 2]
if hasattr(best_model, 'feature_importances_'):
    feature_imp_sorted = feature_importance.sort_values('importance')
    ax.barh(feature_imp_sorted['feature'], feature_imp_sorted['importance'],
            color='steelblue', edgecolor='black')
    ax.set_xlabel('Importance')
    ax.set_title(f'Feature Importance - {best_model_name}')
    ax.grid(True, alpha=0.3, axis='x')
else:
    # Show correlation heatmap instead
    corr_matrix = merged_df[feature_cols + ['Capacity']].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Feature Correlation Matrix')

plt.tight_layout()
plt.savefig('battery_ml_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'battery_ml_analysis.png'", flush=True)

#summary
print("\n" + "="*80, flush=True)
print("ANALYSIS SUMMARY", flush=True)
print("="*80, flush=True)
print(f"\nDataset Information:", flush=True)
print(f"  Total records: {len(merged_df)}", flush=True)
print(f"  Number of batteries: {merged_df['battery_id'].nunique()}", flush=True)
print(f"  Temperature conditions: {sorted(merged_df['ambient_temperature'].unique())}°C", flush=True)
print(f"  Cycle range: {merged_df['cycle_number'].min()} to {merged_df['cycle_number'].max()}", flush=True)

print(f"\nCapacity Insights:", flush=True)
print(f"  Initial capacity range: {merged_df['initial_capacity'].min():.2f} - {merged_df['initial_capacity'].max():.2f}", flush=True)
print(f"  Average capacity degradation: {merged_df['capacity_degradation'].mean():.4f}", flush=True)
print(f"  Average capacity retention: {merged_df['capacity_retention_pct'].mean():.2f}%", flush=True)

print(f"\nModel Performance:", flush=True)
for name, res in results.items():
    print(f"  {name}:", flush=True)
    print(f"    RMSE: {res['rmse']:.4f}", flush=True)
    print(f"    MAE: {res['mae']:.4f}", flush=True)
    print(f"    R²: {res['r2']:.4f}", flush=True)

print("\n" + "="*80, flush=True)
print("ANALYSIS COMPLETE", flush=True)
print("="*80, flush=True)
