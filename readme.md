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

## 🌐 PASO 6: Desplegar en la Nube (OPCIONAL +1 punto)

### Opción A: Render.com (RECOMENDADO)

1. Crea cuenta en [render.com](https://render.com)
2. Conecta tu repositorio de GitHub
3. Crea un "Web Service"
4. Configuración:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:server --bind 0.0.0.0:$PORT`
5. Añade esta línea al final de `app.py`:
   ```python
   server = app.server  # Añadir esta línea antes del if __name__
   ```

### Opción B: Railway.app

1. Crea cuenta en [railway.app](https://railway.app)
2. Sube tu proyecto desde GitHub
3. Railway detectará automáticamente que es Python
4. Configura el comando: `python app.py`

### Opción C: Streamlit Cloud (si cambias a Streamlit)

1. Sube a GitHub
2. Conecta en [streamlit.io/cloud](https://streamlit.io/cloud)
3. Deploy automático

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

### El modelo no entrena
Verifica que tengas columnas numéricas y una columna de ventas/cantidad

---

## 📝 CHECKLIST PARA ENTREGA

- [ ] Código en repositorio público (GitHub/Drive)
- [ ] README.md completo con nombres de integrantes
- [ ] requirements.txt con todas las dependencias
- [ ] app.py funcionando correctamente
- [ ] Dataset incluido en carpeta `data/`
- [ ] PDF tutorial con:
  - [ ] Nombres de todos los integrantes
  - [ ] Explicación del código comentado
  - [ ] Capturas de pantalla de cada funcionalidad
  - [ ] Explicación de gráficos y métricas
  - [ ] (Opcional) Link de aplicación desplegada
- [ ] Notebook de análisis exploratorio (opcional pero recomendado)

---

## 🎓 CRITERIOS DE EVALUACIÓN

| Criterio | Puntos | ¿Cumple? |
|----------|--------|----------|
| Limpieza de datos (loc, iloc, nulos) | 20% | ⬜ |
| EDA con gráficos variados | 20% | ⬜ |
| Modelo ML entrenado y evaluado | 25% | ⬜ |
| Dashboard funcional con Dash | 25% | ⬜ |
| Documentación y código comentado | 10% | ⬜ |
| **Bonus**: Deploy en la nube | +1 punto | ⬜ |

---

## 💡 TIPS PARA OBTENER MÁXIMA CALIFICACIÓN

1. **Comenta tu código**: Explica qué hace cada función
2. **Gráficos variados**: Usa al menos 5 tipos diferentes
3. **Análisis profundo**: No solo muestres gráficos, interpreta los resultados
4. **PDF detallado**: Incluye explicaciones, no solo código
5. **Prueba todo**: Asegúrate que funcione antes de entregar
6. **Deploy**: El +1 punto puede marcar la diferencia

---

## 📞 SOPORTE

Si tienes problemas:
1. Revisa los mensajes de error en la consola
2. Verifica que instalaste todas las dependencias
3. Confirma que el CSV tiene el formato correcto
4. Lee la documentación de Dash: [dash.plotly.com](https://dash.plotly.com)

---

## 🎉 ¡LISTO!

Ahora tienes todo lo necesario para completar el proyecto exitosamente.

**Recuerda**: 
- Grupos de 4 personas: calificación sobre 5.0
- Grupos de 5 personas: calificación sobre 4.0
- Grupos de 6 personas: calificación sobre 3.5

¡Buena suerte! 🍀