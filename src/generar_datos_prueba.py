"""
Script para generar un dataset de ejemplo de ventas de motos en Colombia
Útil para pruebas si no tienes un dataset real
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configurar semilla para reproducibilidad
np.random.seed(42)
random.seed(42)

# Parámetros del dataset
NUM_REGISTROS = 2000

# Datos de Colombia
CIUDADES = [
    'Bogotá', 'Medellín', 'Cali', 'Barranquilla', 'Cartagena',
    'Cúcuta', 'Bucaramanga', 'Pereira', 'Santa Marta', 'Ibagué',
    'Pasto', 'Manizales', 'Neiva', 'Villavicencio', 'Armenia'
]

# Pesos para hacer algunas ciudades más comunes (más ventas)
PESOS_CIUDADES = [0.25, 0.18, 0.15, 0.12, 0.08, 0.05, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.005, 0.005, 0.005]

# Distribución de cilindradas (más motos de baja cilindrada)
DISTRIBUCION_CILINDRADAS = {
    100: 0.15,   # 15% motos 100cc
    110: 0.12,   # 12% motos 110cc
    125: 0.25,   # 25% motos 125cc (más popular)
    135: 0.10,   # 10% motos 135cc
    150: 0.15,   # 15% motos 150cc
    160: 0.08,   # 8% motos 160cc
    180: 0.05,   # 5% motos 180cc
    200: 0.05,   # 5% motos 200cc
    250: 0.03,   # 3% motos 250cc
    300: 0.01,   # 1% motos 300cc
    390: 0.005,  # 0.5% motos 390cc
    400: 0.005,  # 0.5% motos 400cc
    650: 0.003,  # 0.3% motos 650cc
    750: 0.002,  # 0.2% motos 750cc
    800: 0.002,  # 0.2% motos 800cc
    1000: 0.001, # 0.1% motos 1000cc
    1200: 0.001  # 0.1% motos 1200cc
}

# Marcas de motos comunes en Colombia
MARCAS = ['Yamaha', 'Honda', 'Suzuki', 'Kawasaki', 'Bajaj', 'AKT', 'TVS', 'Hero', 'KTM', 'Ducati']

# Modelos por marca
MODELOS = {
    'Yamaha': ['XTZ 125', 'FZ16', 'R15', 'MT-03', 'YZF-R3'],
    'Honda': ['CB 125F', 'CBR 150R', 'CRF 250L', 'CB 190R', 'Africa Twin'],
    'Suzuki': ['GN 125', 'GSX-R150', 'V-Strom 250', 'GSX-S750', 'Hayabusa'],
    'Kawasaki': ['Z125', 'Ninja 250', 'Z400', 'Versys 650', 'Ninja ZX-10R'],
    'Bajaj': ['Pulsar 135', 'Pulsar NS 160', 'Pulsar RS 200', 'Dominar 250', 'Dominar 400'],
    'AKT': ['NKD 125', 'TT 150', 'TTR 180', 'RTX 200', 'JET 250'],
    'TVS': ['Sport 100', 'Apache 160', 'Apache RTR 200', 'Apache RR 310'],
    'Hero': ['Splendor 100', 'Xpulse 200', 'Xtreme 160R'],
    'KTM': ['Duke 200', 'RC 200', 'Duke 390', 'Adventure 390', '790 Duke'],
    'Ducati': ['Monster 821', 'Scrambler 800', 'Panigale V2', 'Multistrada 950']
}

# Cilindradas típicas
CILINDRADAS = [100, 110, 125, 135, 150, 160, 180, 200, 250, 300, 390, 400, 650, 750, 800, 1000, 1200]

def generar_fecha_aleatoria():
    """Genera una fecha aleatoria en los últimos 3 años"""
    fecha_inicio = datetime.now() - timedelta(days=3*365)
    fecha_fin = datetime.now()
    delta = fecha_fin - fecha_inicio
    dias_aleatorios = random.randint(0, delta.days)
    return fecha_inicio + timedelta(days=dias_aleatorios)

def calcular_precio(cilindrada, marca):
    """Calcula un precio realista basado en la cilindrada y marca"""
    # Precio base por CC
    precio_base = cilindrada * 15000
    
    # Factor de marca (marcas premium más caras)
    factores_marca = {
        'Yamaha': 1.2, 'Honda': 1.15, 'Suzuki': 1.1, 'Kawasaki': 1.25,
        'Bajaj': 0.9, 'AKT': 0.85, 'TVS': 0.88, 'Hero': 0.87,
        'KTM': 1.4, 'Ducati': 2.0
    }
    
    factor = factores_marca.get(marca, 1.0)
    precio = precio_base * factor
    
    # Agregar variación aleatoria ±15%
    variacion = np.random.uniform(0.85, 1.15)
    precio_final = precio * variacion
    
    # Redondear a miles
    return round(precio_final / 1000) * 1000

def generar_dataset():
    """Genera el dataset completo"""
    datos = []
    
    for i in range(NUM_REGISTROS):
        # Seleccionar marca y modelo
        marca = random.choice(MARCAS)
        modelo = random.choice(MODELOS[marca])
        
        # Extraer cilindrada del modelo (si está en el nombre)
        # Si no, asignar una cilindrada típica para la marca
        cilindrada = None
        for cc in CILINDRADAS:
            if str(cc) in modelo:
                cilindrada = cc
                break
        
        if cilindrada is None:
            # Asignar cilindrada basada en la marca
            if marca in ['AKT', 'Hero', 'TVS', 'Bajaj']:
                cilindrada = random.choice([100, 110, 125, 135, 150, 160, 180, 200])
            elif marca in ['Yamaha', 'Honda', 'Suzuki']:
                cilindrada = random.choice([125, 150, 160, 200, 250, 300, 400])
            elif marca == 'Kawasaki':
                cilindrada = random.choice([125, 250, 400, 650])
            elif marca == 'KTM':
                cilindrada = random.choice([200, 250, 390, 790])
            else:  # Ducati
                cilindrada = random.choice([800, 950, 1200])
        
        # Calcular precio
        precio = calcular_precio(cilindrada, marca)
        
        # Generar descuento (0-20%, más común 0-10%)
        if random.random() < 0.7:  # 70% de probabilidad de tener descuento
            descuento = random.randint(0, 10)
        else:
            descuento = random.randint(10, 20)
        
        # Precio final con descuento
        precio_final = precio * (1 - descuento/100)
        
        # Ciudad (con distribución no uniforme)
        ciudad = random.choices(CIUDADES, weights=PESOS_CIUDADES)[0]
        
        # Fecha
        fecha = generar_fecha_aleatoria()
        
        # Ventas (cantidad vendida en esa transacción, típicamente 1-5)
        ventas = random.choices([1, 2, 3, 4, 5], weights=[0.70, 0.15, 0.10, 0.03, 0.02])[0]
        
        # Color
        colores = ['Negro', 'Rojo', 'Azul', 'Blanco', 'Gris', 'Verde', 'Amarillo']
        color = random.choice(colores)
        
        # Tipo de transmisión
        transmision = 'Manual' if random.random() < 0.95 else 'Automática'
        
        # Tipo de uso
        tipo_uso = random.choices(
            ['Urbano', 'Deportivo', 'Aventura', 'Trabajo'],
            weights=[0.50, 0.25, 0.15, 0.10]
        )[0]
        
        # Cliente (tipo)
        tipo_cliente = random.choices(
            ['Individual', 'Empresa', 'Gobierno'],
            weights=[0.85, 0.12, 0.03]
        )[0]
        
        # Crear registro
        registro = {
            'id': i + 1,
            'fecha': fecha.strftime('%Y-%m-%d'),
            'año': fecha.year,
            'mes': fecha.month,
            'marca': marca,
            'modelo': modelo,
            'cilindrada': cilindrada,
            'ciudad': ciudad,
            'precio': int(precio),
            'descuento': descuento,
            'precio_final': int(precio_final),
            'ventas': ventas,
            'color': color,
            'transmision': transmision,
            'tipo_uso': tipo_uso,
            'tipo_cliente': tipo_cliente
        }
        
        datos.append(registro)
    
    # Crear DataFrame
    df = pd.DataFrame(datos)
    
    return df

def main():
    """Función principal"""
    print("🏍️ Generando dataset de ventas de motos en Colombia...")
    print(f"📊 Número de registros: {NUM_REGISTROS}")
    
    # Generar dataset
    df = generar_dataset()
    
    # Guardar CSV
    nombre_archivo = 'ventas_motos_colombia.csv'
    df.to_csv(nombre_archivo, index=False)
    
    print(f"\n✅ Dataset generado exitosamente: {nombre_archivo}")
    print(f"📁 Tamaño: {df.shape[0]} filas x {df.shape[1]} columnas")
    
    # Mostrar estadísticas
    print("\n📊 Estadísticas del Dataset:")
    print("=" * 60)
    print(f"Rango de fechas: {df['fecha'].min()} a {df['fecha'].max()}")
    print(f"Ciudades únicas: {df['ciudad'].nunique()}")
    print(f"Marcas únicas: {df['marca'].nunique()}")
    print(f"Rango de cilindradas: {df['cilindrada'].min()}cc - {df['cilindrada'].max()}cc")
    print(f"Precio promedio: ${df['precio_final'].mean():,.0f} COP")
    print(f"Total ventas (unidades): {df['ventas'].sum():,}")
    
    print("\n🏆 Top 5 ciudades con más ventas:")
    print(df.groupby('ciudad')['ventas'].sum().sort_values(ascending=False).head())
    
    print("\n🏍️ Top 5 marcas más vendidas:")
    print(df['marca'].value_counts().head())
    
    print("\n📋 Primeras filas del dataset:")
    print(df.head())
    
    print("\n✅ ¡Listo! Ahora puedes usar este archivo en tu dashboard.")

if __name__ == '__main__':
    main()