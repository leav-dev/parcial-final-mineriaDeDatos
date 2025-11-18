"""
Dashboard de Análisis y Predicción de Ventas de Motos en Colombia
Proyecto Final - Minería de Datos
"""

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import io
import base64
import pickle
from datetime import datetime
import warnings

# Silenciar warnings de sklearn
warnings.filterwarnings('ignore', category=UserWarning)

# Inicializar la aplicación Dash con tema Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Análisis Ventas Motos Colombia"

# Variable global para almacenar el dataset y modelo
data_store = {'df': None, 'modelo': None, 'metricas': None}

# ==================== FUNCIONES AUXILIARES ====================

def crear_rangos_cc(df):
    """Crea columnas de rangos de cilindrada"""
    if 'cilindrada' not in df.columns and 'cc' not in df.columns:
        return df
    
    col_cc = 'cilindrada' if 'cilindrada' in df.columns else 'cc'
    
    def clasificar_cc(cc):
        if pd.isna(cc):
            return 'Desconocido'
        cc = float(cc)
        if cc <= 100:
            return '0-100cc'
        elif cc <= 200:
            return '100-200cc'
        elif cc <= 300:
            return '200-300cc'
        elif cc <= 400:
            return '300-400cc'
        elif cc <= 500:
            return '400-500cc'
        elif cc <= 600:
            return '500-600cc'
        else:
            return '600+cc'
    
    df['rango_cc'] = df[col_cc].apply(clasificar_cc)
    return df

def limpiar_datos(df):
    """Limpia y prepara el dataset"""
    # Eliminar duplicados
    df = df.drop_duplicates()
    
    # Identificar columnas numéricas y eliminar nulos
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df.dropna(subset=numeric_cols)
    
    # Crear rangos de cilindrada si existe la columna
    df = crear_rangos_cc(df)
    
    # Normalizar nombres de ciudades (si existe)
    if 'ciudad' in df.columns:
        df['ciudad'] = df['ciudad'].str.strip().str.title()
    
    return df

def procesar_archivo(contents, filename):
    """Procesa el archivo CSV cargado"""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return None, "Error: Solo se aceptan archivos CSV"
        
        df = limpiar_datos(df)
        return df, f"Archivo '{filename}' cargado exitosamente. {len(df)} registros."
    
    except Exception as e:
        return None, f"Error al procesar el archivo: {str(e)}"

def obtener_top_ciudades(df, n=5):
    """Obtiene las top N ciudades con más ventas"""
    if 'ciudad' not in df.columns:
        return pd.DataFrame()
    
    # Columna de ventas
    col_ventas = None
    for col in ['ventas', 'cantidad', 'unidades_vendidas', 'total_ventas']:
        if col in df.columns:
            col_ventas = col
            break
    
    if col_ventas is None:
        # Si no hay columna explícita, contar registros
        top_ciudades = df['ciudad'].value_counts().head(n).reset_index()
        top_ciudades.columns = ['Ciudad', 'Cantidad de Ventas']
    else:
        top_ciudades = df.groupby('ciudad')[col_ventas].sum().sort_values(ascending=False).head(n).reset_index()
        top_ciudades.columns = ['Ciudad', 'Total Ventas']
    
    return top_ciudades

# ==================== LAYOUT DE LA APLICACIÓN ====================

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🏍️ Dashboard de Ventas de Motos en Colombia", 
                   className="text-center text-primary mb-4 mt-4"),
            html.Hr()
        ])
    ]),
    
    # Sección de carga de datos
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("📁 Cargar Dataset de Ventas", className="card-title"),
                    dcc.Upload(
                        id='upload-data',
                        children=dbc.Button("Seleccionar Archivo CSV", color="primary", className="me-2"),
                        multiple=False
                    ),
                    html.Div(id='output-data-upload', className="mt-3")
                ])
            ], className="mb-4")
        ])
    ]),
    
    # Contenido principal (se muestra después de cargar datos)
    html.Div(id='main-content', children=[
        
        # Tabs para diferentes secciones
        dbc.Tabs([
            # TAB 1: Análisis Exploratorio
            dbc.Tab(label="📊 Análisis Exploratorio", children=[
                dbc.Container([
                    # Insight clave
                    dbc.Row([
                        dbc.Col([
                            dbc.Alert([
                                html.H4("💡 Insight Clave del Mercado", className="alert-heading"),
                                html.P("En Colombia, las motos de baja cilindrada (100-200cc) dominan el mercado. "
                                      "A menor cilindrada, mayor volumen de ventas debido a su precio accesible "
                                      "y menores costos de mantenimiento.", className="mb-0")
                            ], color="info", className="mt-4")
                        ])
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            html.H3("Análisis de Ventas por Rango de Cilindrada", className="mt-4 mb-3"),
                            dcc.Graph(id='grafico-ventas-cc')
                        ], width=12)
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            html.H3("Relación Cilindrada vs Volumen de Ventas", className="mt-4 mb-3"),
                            dcc.Graph(id='grafico-scatter-cc-ventas')
                        ], width=12)
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            html.H3("Top 5 Ciudades con Más Ventas", className="mt-4 mb-3"),
                            html.Div(id='tabla-top-ciudades')
                        ], width=6),
                        
                        dbc.Col([
                            html.H3("Distribución por Ciudad", className="mt-4 mb-3"),
                            dcc.Graph(id='grafico-ciudades')
                        ], width=6)
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            html.H3("Ventas por Rango CC y Ciudad", className="mt-4 mb-3"),
                            dcc.Graph(id='grafico-heatmap')
                        ], width=12)
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            html.H3("Evolución Temporal de Ventas", className="mt-4 mb-3"),
                            dcc.Graph(id='grafico-tiempo')
                        ], width=12)
                    ])
                ], fluid=True)
            ]),
            
            # TAB 2: Modelo Predictivo
            dbc.Tab(label="🤖 Modelo Predictivo", children=[
                dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            html.H3("Entrenamiento del Modelo", className="mt-4 mb-3"),
                            dbc.Card([
                                dbc.CardBody([
                                    html.Label("Seleccione el algoritmo:"),
                                    dcc.Dropdown(
                                        id='selector-modelo',
                                        options=[
                                            {'label': 'Regresión Lineal', 'value': 'linear'},
                                            {'label': 'Árbol de Decisión', 'value': 'tree'},
                                            {'label': 'Random Forest', 'value': 'forest'}
                                        ],
                                        value='linear',
                                        className="mb-3"
                                    ),
                                    dbc.Button("Entrenar Modelo", id="btn-entrenar", color="success", className="me-2"),
                                    dbc.Button("Cargar Modelo Guardado", id="btn-cargar-modelo", color="info", className="me-2"),
                                    html.Div(id='resultado-entrenamiento', className="mt-3")
                                ])
                            ])
                        ], width=12)
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            html.H3("Métricas del Modelo", className="mt-4 mb-3"),
                            html.Div(id='metricas-modelo')
                        ], width=6),
                        
                        dbc.Col([
                            html.H3("Predicción vs Real", className="mt-4 mb-3"),
                            dcc.Graph(id='grafico-prediccion')
                        ], width=6)
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            html.H3("Importancia de Variables", className="mt-4 mb-3"),
                            dcc.Graph(id='grafico-importancia')
                        ], width=12)
                    ])
                ], fluid=True)
            ]),
            
            # TAB 3: Predicción Interactiva
            dbc.Tab(label="🎯 Hacer Predicción", children=[
                dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            html.H3("Predictor de Ventas Interactivo", className="mt-4 mb-3"),
                            dbc.Card([
                                dbc.CardBody([
                                    html.P("Configure los parámetros para predecir las ventas:"),
                                    
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Cilindrada (CC):"),
                                            dcc.Input(id='input-cc', type='number', value=150, 
                                                     className="form-control mb-3")
                                        ], width=4),
                                        
                                        dbc.Col([
                                            html.Label("Precio (COP):"),
                                            dcc.Input(id='input-precio', type='number', value=5000000,
                                                     className="form-control mb-3")
                                        ], width=4),
                                        
                                        dbc.Col([
                                            html.Label("Descuento (%):"),
                                            dcc.Input(id='input-descuento', type='number', value=0,
                                                     className="form-control mb-3")
                                        ], width=4)
                                    ]),
                                    
                                    dbc.Button("Realizar Predicción", id="btn-predecir", 
                                              color="primary", className="mt-3"),
                                    
                                    html.Div(id='resultado-prediccion', className="mt-4")
                                ])
                            ])
                        ], width=12)
                    ])
                ], fluid=True)
            ])
        ])
    ], style={'display': 'none'})
    
], fluid=True)

# ==================== CALLBACKS ====================

# Callback para cargar datos
@app.callback(
    [Output('output-data-upload', 'children'),
     Output('main-content', 'style')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def cargar_datos(contents, filename):
    if contents is None:
        return "", {'display': 'none'}
    
    df, mensaje = procesar_archivo(contents, filename)
    
    if df is not None:
        data_store['df'] = df
        return dbc.Alert(mensaje, color="success"), {'display': 'block'}
    else:
        return dbc.Alert(mensaje, color="danger"), {'display': 'none'}

# Callback para gráfico de ventas por CC
@app.callback(
    Output('grafico-ventas-cc', 'figure'),
    Input('main-content', 'style')
)
def actualizar_grafico_cc(style):
    if data_store['df'] is None or style.get('display') == 'none':
        return {}
    
    df = data_store['df']
    
    if 'rango_cc' not in df.columns:
        return {}
    
    # Buscar columna de ventas
    col_ventas = None
    for col in ['ventas', 'cantidad', 'unidades_vendidas']:
        if col in df.columns:
            col_ventas = col
            break
    
    # Agrupar por rango de CC
    if col_ventas:
        # Si hay columna de ventas, mostrar suma
        ventas_cc = df.groupby('rango_cc')[col_ventas].sum().reset_index(name='total_ventas')
    else:
        # Si no, contar registros
        ventas_cc = df.groupby('rango_cc').size().reset_index(name='total_ventas')
    
    # Ordenar los rangos correctamente
    orden = ['0-100cc', '100-200cc', '200-300cc', '300-400cc', '400-500cc', '500-600cc', '600+cc']
    ventas_cc['rango_cc'] = pd.Categorical(ventas_cc['rango_cc'], categories=orden, ordered=True)
    ventas_cc = ventas_cc.sort_values('rango_cc')
    
    # Crear gráfico con escala de colores invertida (más ventas = más oscuro)
    fig = px.bar(ventas_cc, x='rango_cc', y='total_ventas',
                 title='Ventas de Motos por Rango de Cilindrada',
                 labels={'rango_cc': 'Rango de Cilindrada', 'total_ventas': 'Total de Ventas (unidades)'},
                 color='total_ventas',
                 color_continuous_scale='Blues',
                 text='total_ventas')
    
    fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    
    fig.update_layout(
        height=450,
        showlegend=False,
        annotations=[
            dict(
                text="📊 Nota: Las motos de menor cilindrada (100-200cc) representan el mayor volumen de ventas",
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=11, color="gray"),
                xanchor='center'
            )
        ]
    )
    
    return fig

# Callback para tabla de top ciudades
@app.callback(
    Output('tabla-top-ciudades', 'children'),
    Input('main-content', 'style')
)
def actualizar_tabla_ciudades(style):
    if data_store['df'] is None or style.get('display') == 'none':
        return ""
    
    df = data_store['df']
    top_ciudades = obtener_top_ciudades(df, n=5)
    
    if top_ciudades.empty:
        return dbc.Alert("No se encontró información de ciudades", color="warning")
    
    # Agregar ranking
    top_ciudades.insert(0, 'Ranking', range(1, len(top_ciudades) + 1))
    
    tabla = dash_table.DataTable(
        data=top_ciudades.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in top_ciudades.columns],
        style_cell={'textAlign': 'left', 'padding': '10px'},
        style_header={
            'backgroundColor': 'rgb(30, 144, 255)',
            'color': 'white',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 0},
                'backgroundColor': '#FFD700',
                'fontWeight': 'bold'
            }
        ]
    )
    
    return tabla

# Callback para gráfico de ciudades
@app.callback(
    Output('grafico-ciudades', 'figure'),
    Input('main-content', 'style')
)
def actualizar_grafico_ciudades(style):
    if data_store['df'] is None or style.get('display') == 'none':
        return {}
    
    df = data_store['df']
    top_ciudades = obtener_top_ciudades(df, n=5)
    
    if top_ciudades.empty:
        return {}
    
    fig = px.pie(top_ciudades, 
                 values=top_ciudades.columns[1], 
                 names='Ciudad',
                 title='Distribución de Ventas - Top 5 Ciudades',
                 hole=0.4)
    
    fig.update_layout(height=400)
    return fig

# Callback para heatmap
@app.callback(
    Output('grafico-heatmap', 'figure'),
    Input('main-content', 'style')
)
def actualizar_heatmap(style):
    if data_store['df'] is None or style.get('display') == 'none':
        return {}
    
    df = data_store['df']
    
    if 'rango_cc' not in df.columns or 'ciudad' not in df.columns:
        return {}
    
    # Crear tabla cruzada
    tabla_cruzada = pd.crosstab(df['ciudad'], df['rango_cc'])
    
    # Filtrar top 10 ciudades
    top_10_ciudades = df['ciudad'].value_counts().head(10).index
    tabla_cruzada = tabla_cruzada.loc[top_10_ciudades]
    
    fig = px.imshow(tabla_cruzada,
                    title='Mapa de Calor: Ventas por Ciudad y Rango de Cilindrada',
                    labels=dict(x="Rango CC", y="Ciudad", color="Ventas"),
                    color_continuous_scale='YlOrRd')
    
    fig.update_layout(height=500)
    return fig

# Callback para gráfico temporal
@app.callback(
    Output('grafico-tiempo', 'figure'),
    Input('main-content', 'style')
)
def actualizar_grafico_tiempo(style):
    if data_store['df'] is None or style.get('display') == 'none':
        return {}
    
    df = data_store['df']
    
    # Buscar columna de fecha
    col_fecha = None
    for col in ['fecha', 'date', 'año', 'year', 'mes', 'month']:
        if col in df.columns:
            col_fecha = col
            break
    
    if col_fecha is None:
        return {}
    
    try:
        df_temp = df.copy()
        df_temp['fecha_parsed'] = pd.to_datetime(df_temp[col_fecha], errors='coerce')
        df_temp = df_temp.dropna(subset=['fecha_parsed'])
        
        ventas_tiempo = df_temp.groupby(df_temp['fecha_parsed'].dt.to_period('M')).size().reset_index(name='ventas')
        ventas_tiempo['fecha_parsed'] = ventas_tiempo['fecha_parsed'].dt.to_timestamp()
        
        fig = px.line(ventas_tiempo, x='fecha_parsed', y='ventas',
                     title='Evolución Temporal de Ventas',
                     labels={'fecha_parsed': 'Fecha', 'ventas': 'Cantidad de Ventas'})
        
        fig.update_layout(height=400)
        return fig
    except:
        return {}

# Callback para entrenar modelo
@app.callback(
    [Output('resultado-entrenamiento', 'children'),
     Output('metricas-modelo', 'children'),
     Output('grafico-prediccion', 'figure'),
     Output('grafico-importancia', 'figure')],
    Input('btn-entrenar', 'n_clicks'),
    State('selector-modelo', 'value'),
    prevent_initial_call=True
)
def entrenar_modelo(n_clicks, tipo_modelo):
    if data_store['df'] is None:
        return dbc.Alert("⚠️ Error: No hay datos cargados", color="danger"), "", {}, {}
    
    df = data_store['df']
    
    # Preparar datos para el modelo
    # Seleccionar features numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"🔍 DEBUG: Columnas numéricas encontradas: {numeric_cols}")
    
    # Buscar variable objetivo (ventas)
    target_col = None
    for col in ['ventas', 'cantidad', 'unidades_vendidas', 'total', 'precio_final', 'id']:
        if col in numeric_cols:
            target_col = col
            print(f"✅ DEBUG: Variable objetivo encontrada: {target_col}")
            break
    
    if target_col is None:
        mensaje_error = f"⚠️ Error: No se encontró columna de ventas. Columnas disponibles: {numeric_cols}"
        print(mensaje_error)
        return dbc.Alert(mensaje_error, color="danger"), "", {}, {}
    
    if len(numeric_cols) < 2:
        mensaje_error = f"⚠️ Error: Se necesitan al menos 2 columnas numéricas. Solo se encontraron: {len(numeric_cols)}"
        print(mensaje_error)
        return dbc.Alert(mensaje_error, color="danger"), "", {}, {}
    
    # Features y target
    features = [col for col in numeric_cols if col != target_col]
    
    print(f"🎯 DEBUG: Target = {target_col}")
    print(f"📊 DEBUG: Features = {features}")
    
    if len(features) == 0:
        mensaje_error = "⚠️ Error: No hay features disponibles después de excluir el target"
        print(mensaje_error)
        return dbc.Alert(mensaje_error, color="danger"), "", {}, {}
    
    X = df[features].fillna(0)
    y = df[target_col]
    
    print(f"✅ DEBUG: Shape de X = {X.shape}")
    print(f"✅ DEBUG: Shape de y = {y.shape}")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Seleccionar modelo
    if tipo_modelo == 'linear':
        modelo = LinearRegression()
        nombre_modelo = "Regresión Lineal"
    elif tipo_modelo == 'tree':
        modelo = DecisionTreeRegressor(random_state=42)
        nombre_modelo = "Árbol de Decisión"
    else:
        modelo = RandomForestRegressor(n_estimators=100, random_state=42)
        nombre_modelo = "Random Forest"
    
    # Entrenar
    print(f"🚀 DEBUG: Entrenando modelo {nombre_modelo}...")
    modelo.fit(X_train, y_train)
    print(f"✅ DEBUG: Modelo entrenado exitosamente!")
    
    # Predicciones
    y_pred = modelo.predict(X_test)
    
    # Métricas
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Guardar en data_store
    data_store['modelo'] = modelo
    data_store['features'] = features
    data_store['metricas'] = {'r2': r2, 'mae': mae, 'rmse': rmse}
    
    # Guardar modelo en disco
    try:
        import os
        os.makedirs('models', exist_ok=True)
        
        # Guardar modelo
        with open(f'models/modelo_{tipo_modelo}.pkl', 'wb') as f:
            pickle.dump(modelo, f)
        
        # Guardar features para usar después
        with open('models/features.pkl', 'wb') as f:
            pickle.dump(features, f)
        
        print(f"💾 DEBUG: Modelo guardado en models/modelo_{tipo_modelo}.pkl")
        mensaje_guardado = f" | 💾 Modelo guardado en disco"
    except Exception as e:
        print(f"⚠️ DEBUG: Error guardando modelo: {e}")
        mensaje_guardado = ""
    
    # Mensaje de éxito
    mensaje = dbc.Alert(
        f"✅ Modelo {nombre_modelo} entrenado exitosamente!{mensaje_guardado}", 
        color="success"
    )
    
    # Tarjetas de métricas
    metricas_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{r2:.4f}", className="text-primary"),
                    html.P("R² Score")
                ])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{mae:.2f}", className="text-success"),
                    html.P("MAE")
                ])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{rmse:.2f}", className="text-info"),
                    html.P("RMSE")
                ])
            ])
        ], width=4)
    ])
    
    # Gráfico de predicción vs real
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predicciones'))
    fig_pred.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                   y=[y_test.min(), y_test.max()], 
                                   mode='lines', name='Ideal', line=dict(color='red', dash='dash')))
    fig_pred.update_layout(title='Predicción vs Valores Reales',
                          xaxis_title='Valores Reales',
                          yaxis_title='Predicciones')
    
    # Gráfico de importancia de variables
    if hasattr(modelo, 'feature_importances_'):
        importancias = pd.DataFrame({
            'Feature': features,
            'Importancia': modelo.feature_importances_
        }).sort_values('Importancia', ascending=True)
        
        fig_imp = px.bar(importancias, x='Importancia', y='Feature',
                         title='Importancia de Variables',
                         orientation='h')
    else:
        fig_imp = {}
    
    return mensaje, metricas_cards, fig_pred, fig_imp

# Callback para predicción interactiva
@app.callback(
    Output('resultado-prediccion', 'children'),
    Input('btn-predecir', 'n_clicks'),
    [State('input-cc', 'value'),
     State('input-precio', 'value'),
     State('input-descuento', 'value')],
    prevent_initial_call=True
)
def realizar_prediccion(n_clicks, cc, precio, descuento):
    if data_store['modelo'] is None:
        return dbc.Alert("⚠️ Primero debes entrenar un modelo en la pestaña 'Modelo Predictivo'", color="warning")
    
    modelo = data_store['modelo']
    features = data_store['features']
    
    # Crear entrada para el modelo
    entrada = np.zeros(len(features))
    
    # Mapear valores ingresados (esto es una aproximación)
    for i, feat in enumerate(features):
        if 'cc' in feat.lower() or 'cilindrada' in feat.lower():
            entrada[i] = cc
        elif 'precio' in feat.lower():
            entrada[i] = precio
        elif 'descuento' in feat.lower():
            entrada[i] = descuento
    prediccion = ''
    # Hacer predicción
    print("🚀 DEBUG: Realizando predicción con modelo:", modelo.predict([entrada]))
    prediccion = modelo.predict([entrada])[0]
    
    resultado = dbc.Card([
        dbc.CardBody([
            html.H3("Resultado de la Predicción", className="text-center mb-3"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.H5("Parámetros de Entrada:"),
                    html.P(f"🏍️ Cilindrada: {cc} CC"),
                    html.P(f"💰 Precio: ${precio:,.0f} COP"),
                    html.P(f"🎯 Descuento: {descuento}%")
                ], width=6),
                dbc.Col([
                    html.H5("Predicción:"),
                    html.H2(f"{prediccion:.0f}", className="text-primary"),
                    html.P("Unidades estimadas de venta")
                ], width=6)
            ])
        ])
    ], color="light", className="mt-3")
    
    return resultado

# Callback para cargar modelo guardado
@app.callback(
    Output('resultado-entrenamiento', 'children', allow_duplicate=True),
    Input('btn-cargar-modelo', 'n_clicks'),
    State('selector-modelo', 'value'),
    prevent_initial_call=True
)
def cargar_modelo_guardado(n_clicks, tipo_modelo):
    import os
    
    modelo_path = f'models/modelo_{tipo_modelo}.pkl'
    features_path = 'models/features.pkl'
    
    if not os.path.exists(modelo_path):
        return dbc.Alert(
            f"⚠️ No se encontró un modelo guardado de tipo '{tipo_modelo}'. "
            f"Primero debes entrenar y guardar un modelo.",
            color="warning"
        )
    
    try:
        # Cargar modelo
        with open(modelo_path, 'rb') as f:
            modelo = pickle.load(f)
        
        # Cargar features
        with open(features_path, 'rb') as f:
            features = pickle.load(f)
        
        # Guardar en data_store
        data_store['modelo'] = modelo
        data_store['features'] = features
        
        print(f"✅ DEBUG: Modelo cargado desde {modelo_path}")
        
        return dbc.Alert(
            f"✅ Modelo '{tipo_modelo}' cargado exitosamente desde disco!",
            color="success"
        )
        
    except Exception as e:
        print(f"❌ DEBUG: Error cargando modelo: {e}")
        return dbc.Alert(
            f"❌ Error al cargar el modelo: {str(e)}",
            color="danger"
        )

# ==================== EJECUTAR APLICACIÓN ====================

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)