# 🚀 GUÍA RÁPIDA DE USO
## Dashboard de Ventas de Motos en Colombia

---

## 📋 PASO 1: Preparar el Entorno

### Crear estructura de carpetas:
```bash
proyecto_motos_colombia/
├── app.py
├── requirements.txt
├── README.md
├── generar_datos_ejemplo.py
├── data/
│   └── ventas_motos_colombia.csv
├── notebooks/
│   └── analisis_exploratorio.ipynb
└── models/
    └── (se generará automáticamente)
```

### Instalar dependencias:
```bash
pip install -r requirements.txt
```

---

## 📊 PASO 2: Generar o Preparar los Datos

### Opción A: Generar datos de ejemplo
```bash
python -m generar_datos_prueba
```
Esto creará `ventas_motos_colombia.csv` con 2000 registros de ejemplo.

### Opción B: Usar tu propio dataset
Tu archivo CSV debe tener estas columnas (o similares):
- `cilindrada` o `cc` → Cilindrada de la moto
- `ciudad` → Ciudad de venta
- `ventas` o `cantidad` → Unidades vendidas
- `precio` → Precio de venta
- `descuento` → Descuento aplicado (opcional)
- `fecha` → Fecha de venta (opcional)

---

## 🚀 PASO 3: Ejecutar la Aplicación

```bash
python app.py
```

Abre tu navegador en: **http://localhost:8050**

---

## 🎯 PASO 4: Usar el Dashboard

### 1️⃣ Cargar Datos
- Clic en "Seleccionar Archivo CSV"
- Elige tu archivo `ventas_motos_colombia.csv`
- Espera el mensaje de confirmación ✅

### 2️⃣ Explorar la Pestaña "Análisis Exploratorio"

Verás:
- **Gráfico de barras**: Ventas por rango de cilindrada (100-200cc, 200-300cc, etc.)
- **Tabla Top 5**: Ciudades con más ventas (ranking con ciudad #1 destacada)
- **Gráfico circular**: Distribución porcentual por ciudad
- **Mapa de calor**: Relación ciudad vs. rango de cilindrada
- **Gráfico de línea**: Evolución temporal de ventas

### 3️⃣ Entrenar Modelo (Pestaña "Modelo Predictivo")

1. Selecciona un algoritmo:
   - Regresión Lineal (rápido, simple)
   - Árbol de Decisión (no lineal, interpretable)
   - Random Forest (más preciso, más lento)

2. Clic en "Entrenar Modelo"

3. Revisa las métricas:
   - **R²**: Qué tan bien se ajusta (1.0 = perfecto)
   - **MAE**: Error absoluto medio
   - **RMSE**: Raíz del error cuadrático medio

4. Ve los gráficos:
   - Scatter plot: Predicción vs Real
   - Importancia de variables (si aplica)

### 4️⃣ Hacer Predicciones (Pestaña "Hacer Predicción")

1. Ingresa valores:
   - Cilindrada: ej. 150cc
   - Precio: ej. 5000000 COP
   - Descuento: ej. 5%

2. Clic en "Realizar Predicción"

3. Obtén la estimación de ventas

---

## 📸 PASO 5: Capturar Pantallas para el PDF

Toma capturas de:
1. Dashboard principal con datos cargados
2. Gráfico de ventas por rango de cilindrada
3. Tabla de Top 5 ciudades
4. Métricas del modelo entrenado
5. Resultado de una predicción

---

## ❓ SOLUCIÓN DE PROBLEMAS

### Error: "ModuleNotFoundError"
```bash
pip install --upgrade -r requirements.txt
```

### Error: "Puerto en uso"
Cambia el puerto en `app.py`:
```python
app.run_server(debug=True, port=8051)  # Cambiar 8050 por 8051
```

### El gráfico de cilindrada no aparece
Verifica que tu CSV tenga columna `cilindrada` o `cc`

### No aparece la tabla de ciudades
Verifica que tu CSV tenga columna `ciudad`



