# Project Pipeline

---

# Modeling and Construction Phase


1. Lectura de datos.

    2. Exploración de imagenes (visualizaciones de input-output y sacar medidas de interés) - Fuera del pipeline secuencial de ejecución
3. Definición de transformaciones (normalización y estandarización)
4. Modulo de Dataset
5. Modulo de Datos Lightning (Dataloaders) (LightningDataModule) (Split de dataset)

**CHECKPOINT 1**

6. LightningModule
    - training, validation y test steps
    - configuración de optimizadores
    - forward
    - init (definir loss y leraning rate)
7. torch nn.Module con arquitectura Unet + DDPM (Unet/Unet.py)
    - Implementar componentes (time embedding.py, downblock.py, midblock.py y upblock.py)
    
**CHECKPOINT 2**

8. Entrenamiento 
9. Evaluación (metricas, curvas de aprendiazje, etc.)
10. Evaluación Preproduction
11. Exportar modelo como objeto serializado

--- 
# Production Phase 

1. Implementar UI
2. Conectar modelo serializado con UI
3. Evaluación del ambiente

