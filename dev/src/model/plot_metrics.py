import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIG ---
CSV_PATH = "TrainingLogs/picobanana_model/version_9/metrics.csv"  
OUTPUT_DIR = "Metrics_Plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading metrics from {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH)

# Mostrar columnas detectadas
print("Detected Columns:")
print(df.columns)

# Eliminar columnas que están totalmente vacías
df = df.dropna(axis=1, how='all')

# Eliminar filas completamente vacías
df = df.dropna(how='all')

print("\nUsed columns for plots:")
print(df.columns)

# --- FUNCIÓN PARA GRAFICAR UNA COLUMNA ---
def plot_metric(column_name, ylabel=None):
    if column_name not in df.columns:
        return

    series = df[column_name].dropna()
    if len(series) == 0:
        return
    
    plt.figure(figsize=(8, 5))
    plt.plot(series.index, series.values)
    plt.title(column_name)
    plt.xlabel("Index (steps/epochs depending on metric)")
    plt.ylabel(ylabel if ylabel else column_name)
    plt.grid(True)

    out_path = os.path.join(OUTPUT_DIR, f"{column_name}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Generated: {out_path}")

# --- GRAFICAR MÉTRICAS PRINCIPALES ---
for col in ["train_loss_step", "train_loss_epoch", "val_loss"]:
    plot_metric(col)

# --- GRAFICAR TODAS LAS MÉTRICAS NUMÉRICAS ---
numeric_cols = df.select_dtypes(include=["float", "int"]).columns
for col in numeric_cols:
    plot_metric(col)

print("\nPlot phase finished...")

