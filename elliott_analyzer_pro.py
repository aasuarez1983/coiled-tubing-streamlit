#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analizador Avanzado de Ondas de Elliott con Alertas de Trading
Autor: Trading Analysis Tool Pro
Versión: 4.0 (Weighted, R:R filter ≥ 3:1, Intelligent Reports)
"""

import sys
import subprocess
import importlib.util
from datetime import datetime, timedelta
import threading
import time

# Lista de paquetes requeridos
REQUIRED_PACKAGES = {
    'yfinance': 'yfinance',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'matplotlib': 'matplotlib',
    'scipy': 'scipy',
    'playsound': 'playsound'  # Para alertas de sonido
}


def check_and_install_packages():
    """Verificar e instalar paquetes faltantes"""
    missing_packages = []

    for package_name, pip_name in REQUIRED_PACKAGES.items():
        if package_name == 'playsound':  # Opcional para alertas de sonido
            continue
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            missing_packages.append(pip_name)

    if missing_packages:
        print("=" * 60)
        print("INSTALACIÓN DE DEPENDENCIAS NECESARIAS")
        print("=" * 60)
        print(f"\nPaquetes faltantes detectados: {', '.join(missing_packages)}")

        response = input("\n¿Deseas instalar automáticamente los paquetes faltantes? (s/n): ")

        if response.lower() == 's':
            print("\nInstalando paquetes...")
            for package in missing_packages:
                print(f"\nInstalando {package}...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    print(f"✓ {package} instalado correctamente")
                except subprocess.CalledProcessError:
                    print(f"✗ Error al instalar {package}")
                    print(f"Intenta instalar manualmente con: pip install {package}")
                    return False

            print("\n✅ Todas las dependencias instaladas correctamente")
            print("Por favor, reinicia el programa para cargar los módulos instalados.")
            input("\nPresiona Enter para salir...")
            sys.exit(0)
        else:
            print("\n⚠️ Para usar esta aplicación necesitas instalar:")
            print(f"pip install {' '.join(missing_packages)}")
            input("\nPresiona Enter para salir...")
            sys.exit(1)

    return True


# Verificar dependencias antes de importar
if check_and_install_packages():
    print("✅ Todas las dependencias están instaladas")
    print("Iniciando aplicación...\n")

# Ahora importar los módulos
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings('ignore')

# Intentar importar playsound para alertas
try:
    from playsound import playsound

    SOUND_ENABLED = True
except Exception:
    SOUND_ENABLED = False


class TradingSignal:
    """Clase para manejar señales de trading con cálculo correcto de R:R."""

    def __init__(self, tipo, precio, fecha, razon, fuerza, objetivo1, objetivo2, objetivo3, stop_loss, source="GENERIC"):
        self.tipo = tipo  # 'COMPRA' o 'VENTA'
        self.precio = float(precio)
        self.fecha = fecha
        self.razon = razon
        self.fuerza = int(fuerza)  # 1-5 estrellas
        self.objetivo1 = float(objetivo1) if objetivo1 is not None else None
        self.objetivo2 = float(objetivo2) if objetivo2 is not None else None
        self.objetivo3 = float(objetivo3) if objetivo3 is not None else None
        self.stop_loss = float(stop_loss)
        self.source = str(source).upper()
        self.targets = [t for t in [self.objetivo1, self.objetivo2, self.objetivo3] if t is not None]
        self.risk_reward = self._compute_best_risk_reward()

    def _compute_best_risk_reward(self) -> float:
        if not self.targets:
            return 0.0
        if self.tipo.upper() == 'COMPRA':
            risk = self.precio - self.stop_loss
            if risk <= 0:
                return 0.0
            best_reward = max(max(t - self.precio, 0.0) for t in self.targets)
            return round(best_reward / risk, 2) if risk > 0 else 0.0
        else:  # VENTA
            risk = self.stop_loss - self.precio
            if risk <= 0:
                return 0.0
            best_reward = max(max(self.precio - t, 0.0) for t in self.targets)
            return round(best_reward / risk, 2) if risk > 0 else 0.0


class ElliottWaveAnalyzerPro:
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 Elliott Wave Analyzer PRO - Sistema Automático de Trading v4.0")
        self.root.geometry("1500x900")

        # Configurar estilo
        self.setup_styles()

        # Variables
        self.df = None
        self.symbol = tk.StringVar(value="AAPL")
        self.period = tk.StringVar(value="6mo")
        self.auto_refresh = tk.BooleanVar(value=False)
        self.alert_enabled = tk.BooleanVar(value=True)
        self.wave_points = []
        self.fibonacci_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.618, 4.236]
        self.trading_signals = []  # Solo señales accionables (R:R >= min)
        self.all_signals = []       # Todas las señales detectadas (para referencia)
        self.current_position = None

        # Pesos de consenso (prioriza Elliott)
        self.signal_weights = {
            'ELLIOTT': 0.50,
            'FIBONACCI': 0.15,
            'MACD': 0.15,
            'RSI': 0.10,
            'BOLLINGER': 0.10,
        }
        self.weighted_decision = None

        # Listas predefinidas de activos
        self.setup_asset_lists()

        # Configuración ideal para máxima ganancia
        self.max_profit_config = {
            'entrada_onda2_fib': 0.618,
            'entrada_onda4_fib': 0.382,
            'salida_onda3_fib': 1.618,
            'salida_onda5_fib': 1.0,
            'stop_loss_percent': 3.0,
            'position_size_percent': 2.0,
            'min_risk_reward': 3.0
        }

        self.setup_ui()
        self.start_monitoring()

    def setup_styles(self):
        """Configurar estilos visuales avanzados"""
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass

        style.configure('Title.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Buy.TLabel', foreground='green', font=('Arial', 11, 'bold'))
        style.configure('Sell.TLabel', foreground='red', font=('Arial', 11, 'bold'))
        style.configure('Alert.TLabel', foreground='orange', font=('Arial', 11, 'bold'))

    def setup_asset_lists(self):
        """Configurar listas de activos populares"""
        self.asset_categories = {
            "📈 ACCIONES POPULARES": [
                "AAPL - Apple",
                "MSFT - Microsoft",
                "GOOGL - Google",
                "AMZN - Amazon",
                "TSLA - Tesla",
                "META - Meta/Facebook",
                "NVDA - Nvidia",
                "JPM - JP Morgan",
                "V - Visa",
                "JNJ - Johnson & Johnson",
                "WMT - Walmart",
                "PG - Procter & Gamble",
                "DIS - Disney",
                "NFLX - Netflix",
                "BA - Boeing"
            ],
            "💰 CRIPTOMONEDAS": [
                "BTC-USD - Bitcoin",
                "ETH-USD - Ethereum",
                "BNB-USD - Binance Coin",
                "ADA-USD - Cardano",
                "XRP-USD - Ripple",
                "SOL-USD - Solana",
                "DOT-USD - Polkadot",
                "DOGE-USD - Dogecoin",
                "AVAX-USD - Avalanche",
                "MATIC-USD - Polygon",
                "LINK-USD - Chainlink",
                "UNI-USD - Uniswap",
                "ATOM-USD - Cosmos",
                "LTC-USD - Litecoin",
                "BCH-USD - Bitcoin Cash"
            ],
            "📊 ETFs PRINCIPALES": [
                "SPY - S&P 500",
                "QQQ - Nasdaq 100",
                "DIA - Dow Jones",
                "IWM - Russell 2000",
                "VTI - Total Market",
                "VOO - Vanguard S&P 500",
                "GLD - Gold",
                "SLV - Silver",
                "USO - Oil",
                "TLT - Treasury Bonds",
                "VNQ - Real Estate",
                "ARKK - ARK Innovation",
                "XLF - Financials",
                "XLK - Technology",
                "XLE - Energy"
            ],
            "🌎 MERCADOS INTERNACIONALES": [
                "EWJ - Japón",
                "EWG - Alemania",
                "EWU - Reino Unido",
                "EWC - Canadá",
                "EWA - Australia",
                "EWZ - Brasil",
                "INDA - India",
                "FXI - China",
                "EWT - Taiwan",
                "EWY - Corea del Sur"
            ]
        }

    def setup_ui(self):
        """Configurar interfaz de usuario avanzada"""
        # Notebook principal
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Pestaña principal de análisis
        main_tab = ttk.Frame(notebook)
        notebook.add(main_tab, text="📊 Análisis Principal")

        # Pestaña de señales de trading
        signals_tab = ttk.Frame(notebook)
        notebook.add(signals_tab, text="🎯 Señales de Trading")

        # Pestaña de configuración
        config_tab = ttk.Frame(notebook)
        notebook.add(config_tab, text="⚙️ Configuración")

        # Configurar pestaña principal
        self.setup_main_tab(main_tab)

        # Configurar pestaña de señales
        self.setup_signals_tab(signals_tab)

        # Configurar pestaña de configuración
        self.setup_config_tab(config_tab)

    def setup_main_tab(self, parent):
        """Configurar pestaña principal de análisis"""
        main_container = ttk.Frame(parent)
        main_container.pack(fill='both', expand=True)

        left_frame = ttk.Frame(main_container, width=250)
        left_frame.pack(side='left', fill='y', padx=5, pady=5)
        left_frame.pack_propagate(False)

        right_frame = ttk.Frame(main_container)
        right_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)

        # ========== PANEL IZQUIERDO ==========
        asset_frame = ttk.LabelFrame(left_frame, text="📊 Selección de Activo", padding="10")
        asset_frame.pack(fill='x', pady=5)

        ttk.Label(asset_frame, text="Categoría:", font=('Arial', 9, 'bold')).pack(anchor='w', pady=2)
        self.category_var = tk.StringVar(value="📈 ACCIONES POPULARES")
        category_combo = ttk.Combobox(asset_frame, textvariable=self.category_var,
                                      values=list(self.asset_categories.keys()),
                                      width=25, state='readonly')
        category_combo.pack(fill='x', pady=2)
        category_combo.bind('<<ComboboxSelected>>', self.on_category_change)

        ttk.Label(asset_frame, text="Activo:", font=('Arial', 9, 'bold')).pack(anchor='w', pady=2)
        self.asset_combo = ttk.Combobox(asset_frame, width=25, state='readonly')
        self.asset_combo.pack(fill='x', pady=2)
        self.asset_combo.bind('<<ComboboxSelected>>', self.on_asset_change)

        ttk.Label(asset_frame, text="Símbolo Manual:", font=('Arial', 9, 'bold')).pack(anchor='w', pady=2)
        self.custom_symbol = ttk.Entry(asset_frame, textvariable=self.symbol)
        self.custom_symbol.pack(fill='x', pady=2)

        ttk.Label(asset_frame, text="Período:", font=('Arial', 9, 'bold')).pack(anchor='w', pady=2)
        period_combo = ttk.Combobox(asset_frame, textvariable=self.period, state='readonly')
        period_combo['values'] = ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
        period_combo.pack(fill='x', pady=2)

        viz_frame = ttk.LabelFrame(left_frame, text="🎨 Controles de Visualización", padding="10")
        viz_frame.pack(fill='x', pady=5)

        self.show_elliott = tk.BooleanVar(value=True)
        self.show_bollinger = tk.BooleanVar(value=True)
        self.show_fibonacci = tk.BooleanVar(value=True)
        self.show_sma = tk.BooleanVar(value=True)
        self.show_volume = tk.BooleanVar(value=True)
        self.show_signals = tk.BooleanVar(value=True)

        ttk.Checkbutton(viz_frame, text="📐 Ondas Elliott",
                        variable=self.show_elliott, command=self.update_visualization).pack(anchor='w', pady=1)
        ttk.Checkbutton(viz_frame, text="📊 Bandas Bollinger",
                        variable=self.show_bollinger, command=self.update_visualization).pack(anchor='w', pady=1)
        ttk.Checkbutton(viz_frame, text="📏 Fibonacci",
                        variable=self.show_fibonacci, command=self.update_visualization).pack(anchor='w', pady=1)
        ttk.Checkbutton(viz_frame, text="📈 Medias Móviles",
                        variable=self.show_sma, command=self.update_visualization).pack(anchor='w', pady=1)
        ttk.Checkbutton(viz_frame, text="📊 Volumen",
                        variable=self.show_volume, command=self.update_visualization).pack(anchor='w', pady=1)
        ttk.Checkbutton(viz_frame, text="🎯 Señales Trading",
                        variable=self.show_signals, command=self.update_visualization).pack(anchor='w', pady=1)

        action_frame = ttk.LabelFrame(left_frame, text="⚡ Acciones", padding="10")
        action_frame.pack(fill='x', pady=5)

        ttk.Button(action_frame, text="📥 Cargar Datos",
                   command=self.load_data).pack(fill='x', pady=2)
        ttk.Button(action_frame, text="🔄 Análisis Completo",
                   command=self.run_complete_analysis).pack(fill='x', pady=2)
        ttk.Button(action_frame, text="🎯 Detectar Señales",
                   command=self.detect_trading_signals).pack(fill='x', pady=2)
        ttk.Button(action_frame, text="🔍 Zoom Reset",
                   command=self.reset_zoom).pack(fill='x', pady=2)

        options_frame = ttk.LabelFrame(left_frame, text="⚙️ Opciones", padding="10")
        options_frame.pack(fill='x', pady=5)

        ttk.Checkbutton(options_frame, text="🔄 Auto-Actualizar",
                        variable=self.auto_refresh).pack(anchor='w', pady=2)
        ttk.Checkbutton(options_frame, text="🔔 Alertas Activas",
                        variable=self.alert_enabled).pack(anchor='w', pady=2)

        legend_frame = ttk.LabelFrame(left_frame, text="📖 Leyenda", padding="10")
        legend_frame.pack(fill='both', expand=True, pady=5)

        legend_text = tk.Text(legend_frame, wrap=tk.WORD, height=10, width=25, font=('Consolas', 8))
        legend_text.insert('1.0', """ONDAS ELLIOTT:
🟢 1,3,5 = Impulso alcista
🔴 2,4 = Corrección
🟠 A,B,C = Corrección ABC

BOLLINGER:
--- Verde = Banda superior
--- Azul = Banda media
--- Rojo = Banda inferior

FIBONACCI:
═══ Retrocesos (0-100%)
┅┅┅ Extensiones (>100%)

SEÑALES:
▲ Verde = COMPRAR
▼ Rojo = VENDER

MEDIAS MÓVILES:
— Naranja = SMA 20
— Azul = SMA 50""")
        legend_text.config(state='disabled')
        legend_text.pack(fill='both', expand=True)

        # ========== PANEL DERECHO ==========
        self.alert_frame = ttk.LabelFrame(right_frame, text="🚨 ALERTAS", padding="5")
        self.alert_frame.pack(fill='x', pady=2)

        self.alert_label = ttk.Label(self.alert_frame, text="Esperando análisis...",
                                     style='Alert.TLabel', font=('Arial', 10, 'bold'))
        self.alert_label.pack(padx=5, pady=2)

        self.chart_frame = ttk.LabelFrame(right_frame, text="📈 Gráfico Integrado - Elliott + Bollinger + Fibonacci",
                                          padding="5")
        self.chart_frame.pack(fill='both', expand=True, pady=2)

        info_notebook = ttk.Notebook(right_frame)
        info_notebook.pack(fill='x', pady=2)

        analysis_frame = ttk.Frame(info_notebook)
        info_notebook.add(analysis_frame, text="📊 Análisis")

        self.info_text = scrolledtext.ScrolledText(analysis_frame, height=6, wrap=tk.WORD,
                                                   font=('Consolas', 9))
        self.info_text.pack(fill='both', expand=True, padx=5, pady=5)

        points_frame = ttk.Frame(info_notebook)
        info_notebook.add(points_frame, text="💰 Puntos de Trading")

        self.points_text = scrolledtext.ScrolledText(points_frame, height=6, wrap=tk.WORD,
                                                     font=('Consolas', 9))
        self.points_text.pack(fill='both', expand=True, padx=5, pady=5)

        self.on_category_change()

    def setup_signals_tab(self, parent):
        """Configurar pestaña de señales de trading"""
        current_frame = ttk.LabelFrame(parent, text="📍 Señales Actuales", padding="10")
        current_frame.pack(fill='both', expand=True, padx=5, pady=5)

        columns = ('Fecha', 'Tipo', 'Precio', 'Stop Loss', 'Objetivo 1', 'Objetivo 2',
                   'Objetivo 3', 'R:R', 'Fuerza', 'Razón')

        self.signals_tree = ttk.Treeview(current_frame, columns=columns, height=10)
        self.signals_tree.pack(fill='both', expand=True)

        self.signals_tree.heading('#0', text='ID')
        self.signals_tree.column('#0', width=50)

        for col in columns:
            self.signals_tree.heading(col, text=col)
            if col in ['Tipo', 'Fuerza']:
                self.signals_tree.column(col, width=100)
            elif col == 'Razón':
                self.signals_tree.column(col, width=260)
            else:
                self.signals_tree.column(col, width=110)

        scrollbar = ttk.Scrollbar(current_frame, orient='vertical', command=self.signals_tree.yview)
        scrollbar.pack(side='right', fill='y')
        self.signals_tree.configure(yscrollcommand=scrollbar.set)

        summary_frame = ttk.LabelFrame(parent, text="📊 Resumen de Trading", padding="10")
        summary_frame.pack(fill='x', padx=5, pady=5)

        self.summary_text = tk.Text(summary_frame, height=12, font=('Consolas', 10))
        self.summary_text.pack(fill='both', expand=True)

    def setup_config_tab(self, parent):
        """Configurar pestaña de configuración"""
        profit_frame = ttk.LabelFrame(parent, text="💎 Configuración Ideal para Máxima Ganancia",
                                      padding="10")
        profit_frame.pack(fill='x', padx=10, pady=10)

        configs = [
            ("Entrada Onda 2 (Fibonacci):", 'entrada_onda2_fib', 0.618),
            ("Entrada Onda 4 (Fibonacci):", 'entrada_onda4_fib', 0.382),
            ("Salida Onda 3 (Fibonacci):", 'salida_onda3_fib', 1.618),
            ("Salida Onda 5 (Fibonacci):", 'salida_onda5_fib', 1.0),
            ("Stop Loss (%):", 'stop_loss_percent', 3.0),
            ("Tamaño Posición (% capital):", 'position_size_percent', 2.0),
            ("Risk/Reward Mínimo:", 'min_risk_reward', 3.0)
        ]

        self.config_vars = {}
        for i, (label, key, default) in enumerate(configs):
            ttk.Label(profit_frame, text=label).grid(row=i, column=0, sticky='w', padx=5, pady=5)
            var = tk.DoubleVar(value=default)
            self.config_vars[key] = var
            entry = ttk.Entry(profit_frame, textvariable=var, width=10)
            entry.grid(row=i, column=1, padx=5, pady=5)

        ttk.Button(profit_frame, text="💾 Guardar Configuración",
                   command=self.save_config).grid(row=len(configs), column=0,
                                                  columnspan=2, pady=10)

        rules_frame = ttk.LabelFrame(parent, text="📋 Checklist de Trading", padding="10")
        rules_frame.pack(fill='both', expand=True, padx=10, pady=10)

        rules_text = """
        ✅ CHECKLIST ANTES DE COMPRAR:

        □ Identificación clara de las 5 ondas de Elliott
        □ Retroceso en zona de Fibonacci (38.2%-61.8%)
        □ Divergencia positiva en RSI
        □ Volumen confirma el patrón
        □ Ratio Riesgo/Beneficio mínimo 1:3
        □ Stop loss definido
        □ Tamaño de posición calculado (máx 2% riesgo)

        🎯 PUNTOS IDEALES DE ENTRADA:
        • Final de Onda 2: Retroceso 50%-61.8% ⭐⭐⭐⭐⭐
        • Final de Onda 4: Retroceso 38.2%-50% ⭐⭐⭐⭐
        • Final de Onda C: Corrección completa ⭐⭐⭐⭐

        🔴 PUNTOS DE SALIDA:
        • Final de Onda 3: Venta parcial (50%)
        • Final de Onda 5: Venta total
        • Stop Loss tocado: Salida inmediata

        ⚠️ NUNCA COMPRAR SI:
        • Onda 2 retrocede >100% de Onda 1
        • Onda 4 entra en territorio de Onda 1
        • No hay alternancia entre Ondas 2 y 4
        • RSI en sobrecompra extrema (>80)
        """

        rules_label = tk.Text(rules_frame, wrap=tk.WORD, font=('Consolas', 10))
        rules_label.insert('1.0', rules_text)
        rules_label.config(state='disabled')
        rules_label.pack(fill='both', expand=True)

    def on_category_change(self, event=None):
        """Manejar cambio de categoría de activos"""
        category = self.category_var.get()
        assets = self.asset_categories.get(category, [])
        self.asset_combo['values'] = assets
        if assets:
            self.asset_combo.current(0)
            self.on_asset_change()

    def on_asset_change(self, event=None):
        """Manejar cambio de activo seleccionado"""
        selected = self.asset_combo.get()
        if selected:
            symbol = selected.split(' - ')[0]
            self.symbol.set(symbol)

    def save_config(self):
        """Guardar configuración de trading"""
        for key, var in self.config_vars.items():
            self.max_profit_config[key] = var.get()
        messagebox.showinfo("Configuración", "✅ Configuración guardada exitosamente")

    def start_monitoring(self):
        """Iniciar monitoreo automático"""

        def monitor():
            while True:
                if self.auto_refresh.get() and self.df is not None:
                    self.root.after(0, self.refresh_data)
                time.sleep(60)

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

    def refresh_data(self):
        """Refrescar datos y análisis"""
        try:
            self.load_data(silent=True)
            self.run_complete_analysis()
        except Exception:
            pass

    def load_data(self, silent=False):
        """Cargar datos de Yahoo Finance"""
        try:
            if not silent:
                self.info_text.delete(1.0, tk.END)
                self.info_text.insert(tk.END, f"⏳ Cargando datos para {self.symbol.get()}...\n")
                self.root.update()

            ticker = yf.Ticker(self.symbol.get())
            self.df = ticker.history(period=self.period.get())

            if self.df.empty:
                if not silent:
                    messagebox.showerror("Error", "No se pudieron cargar los datos")
                return False

            self.calculate_indicators()

            if not silent:
                self.display_basic_info()
                self.plot_basic_chart()
                self.info_text.insert(tk.END, "\n✅ Datos cargados exitosamente\n")

            return True

        except Exception as e:
            if not silent:
                messagebox.showerror("Error", f"Error al cargar datos: {str(e)}")
            return False

    def calculate_indicators(self):
        """Calcular todos los indicadores técnicos"""
        self.df['SMA20'] = self.df['Close'].rolling(window=20).mean()
        self.df['SMA50'] = self.df['Close'].rolling(window=50).mean()
        self.df['EMA12'] = self.df['Close'].ewm(span=12, adjust=False).mean()
        self.df['EMA26'] = self.df['Close'].ewm(span=26, adjust=False).mean()

        self.df['MACD'] = self.df['EMA12'] - self.df['EMA26']
        self.df['Signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.df['MACD_Histogram'] = self.df['MACD'] - self.df['Signal']

        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))

        self.df['BB_Middle'] = self.df['Close'].rolling(window=20).mean()
        bb_std = self.df['Close'].rolling(window=20).std()
        self.df['BB_Upper'] = self.df['BB_Middle'] + (bb_std * 2)
        self.df['BB_Lower'] = self.df['BB_Middle'] - (bb_std * 2)

        high = self.df['High']
        low = self.df['Low']
        close = self.df['Close'].shift()

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.df['ATR'] = tr.rolling(window=14).mean()

        low_14 = self.df['Low'].rolling(window=14).min()
        high_14 = self.df['High'].rolling(window=14).max()
        self.df['%K'] = 100 * ((self.df['Close'] - low_14) / (high_14 - low_14))
        self.df['%D'] = self.df['%K'].rolling(window=3).mean()

    def display_basic_info(self):
        """Mostrar información básica del activo"""
        current_price = self.df['Close'].iloc[-1]
        change = self.df['Close'].iloc[-1] - self.df['Close'].iloc[-2]
        change_pct = (change / self.df['Close'].iloc[-2]) * 100

        info = f"""
╔════════════════════════════════════════════════════╗
║           INFORMACIÓN DEL ACTIVO                   ║
╚════════════════════════════════════════════════════╝
📊 Símbolo: {self.symbol.get()}
💵 Precio Actual: ${current_price:.2f}
📈 Cambio: ${change:.2f} ({change_pct:+.2f}%)
📅 Período: {len(self.df)} días
📈 Máximo: ${self.df['High'].max():.2f}
📉 Mínimo: ${self.df['Low'].min():.2f}
📊 Volumen: {self.df['Volume'].iloc[-1]:,.0f}
        """
        self.info_text.insert(tk.END, info)

    def update_visualization(self):
        """Actualizar visualización según checkboxes seleccionados"""
        if self.df is not None and hasattr(self, 'canvas'):
            self.update_complete_chart()

    def reset_zoom(self):
        """Resetear zoom del gráfico"""
        if hasattr(self, 'ax'):
            self.ax.autoscale()
            if hasattr(self, 'ax2'):
                self.ax2.autoscale()
            if hasattr(self, 'ax3'):
                self.ax3.autoscale()
            if hasattr(self, 'canvas'):
                self.canvas.draw()

    def plot_basic_chart(self):
        """Graficar el chart con indicadores y controles de visualización"""
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        num_subplots = 1
        if hasattr(self, 'show_volume') and self.show_volume.get():
            num_subplots += 1
        num_subplots += 2

        fig = Figure(figsize=(12, 8))

        height_ratios = [3]
        if hasattr(self, 'show_volume') and self.show_volume.get():
            height_ratios.append(1)
        height_ratios.extend([1, 1])

        gs = fig.add_gridspec(len(height_ratios), 1, height_ratios=height_ratios, hspace=0.1)

        subplot_idx = 0
        self.ax = fig.add_subplot(gs[subplot_idx])
        subplot_idx += 1

        self.plot_candlesticks_enhanced(self.ax)

        if hasattr(self, 'show_volume') and self.show_volume.get():
            self.ax_volume = fig.add_subplot(gs[subplot_idx], sharex=self.ax)
            self.plot_volume(self.ax_volume)
            subplot_idx += 1

        self.ax2 = fig.add_subplot(gs[subplot_idx], sharex=self.ax)
        self.plot_rsi(self.ax2)
        subplot_idx += 1

        self.ax3 = fig.add_subplot(gs[subplot_idx], sharex=self.ax)
        self.plot_macd(self.ax3)

        current_price = self.df['Close'].iloc[-1]
        change = self.df['Close'].iloc[-1] - self.df['Close'].iloc[-2]
        change_pct = (change / self.df['Close'].iloc[-2]) * 100

        title = f'{self.symbol.get()} - ${current_price:.2f} ({change_pct:+.2f}%) | Elliott + Bollinger + Fibonacci'
        self.ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

        for ax in fig.axes[:-1]:
            ax.set_xticklabels([])

        fig.tight_layout()

        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

        canvas_frame = ttk.Frame(self.chart_frame)
        canvas_frame.pack(fill='both', expand=True)

        self.canvas = FigureCanvasTkAgg(fig, canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, canvas_frame)
        toolbar.update()

    def plot_candlesticks_enhanced(self, ax):
        """Graficar velas japonesas mejoradas con elementos opcionales"""
        for idx in range(len(self.df)):
            date = self.df.index[idx]
            open_price = self.df['Open'].iloc[idx]
            close_price = self.df['Close'].iloc[idx]
            high_price = self.df['High'].iloc[idx]
            low_price = self.df['Low'].iloc[idx]

            color = '#00b050' if close_price >= open_price else '#ff0000'

            ax.plot([date, date], [low_price, high_price],
                    color='black', linewidth=0.5, alpha=0.6)

            height = abs(close_price - open_price)
            bottom = min(close_price, open_price)

            if height > 0:
                ax.bar(date, height, bottom=bottom, width=0.8,
                       color=color, alpha=0.8, edgecolor='black', linewidth=0.5)

        if hasattr(self, 'show_sma') and self.show_sma.get():
            if 'SMA20' in self.df.columns:
                ax.plot(self.df.index, self.df['SMA20'],
                        label='SMA 20', color='orange', alpha=0.9, linewidth=1.5)
            if 'SMA50' in self.df.columns:
                ax.plot(self.df.index, self.df['SMA50'],
                        label='SMA 50', color='blue', alpha=0.9, linewidth=1.5)

        ax.set_ylabel('Precio ($)', fontsize=10)
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    def plot_volume(self, ax):
        """Graficar volumen con colores"""
        colors = ['#00b050' if self.df['Close'].iloc[i] >= self.df['Open'].iloc[i]
                  else '#ff0000' for i in range(len(self.df))]

        ax.bar(self.df.index, self.df['Volume'], color=colors, alpha=0.6)
        ax.set_ylabel('Volumen', fontsize=9)
        ax.grid(True, alpha=0.2)

        avg_volume = self.df['Volume'].rolling(window=20).mean()
        ax.plot(self.df.index, avg_volume, color='purple', alpha=0.5, linewidth=1, label='Vol. Promedio')
        ax.legend(loc='upper left', fontsize=7)

    def plot_rsi(self, ax):
        """Graficar RSI con zonas coloreadas"""
        ax.plot(self.df.index, self.df['RSI'], 'b-', linewidth=1.5, label='RSI')
        ax.axhline(y=70, color='r', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=30, color='g', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=50, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.fill_between(self.df.index, 70, 100, alpha=0.1, color='red', label='Sobrecompra')
        ax.fill_between(self.df.index, 0, 30, alpha=0.1, color='green', label='Sobreventa')
        ax.fill_between(self.df.index, 30, 70, alpha=0.05, color='gray')
        ax.set_ylabel('RSI', fontsize=9)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.2)

    def plot_macd(self, ax):
        """Graficar MACD con histograma coloreado"""
        ax.plot(self.df.index, self.df['MACD'], 'b-', label='MACD', linewidth=1.2)
        ax.plot(self.df.index, self.df['Signal'], 'r-', label='Signal', linewidth=1.2)
        colors = ['#00b050' if val >= 0 else '#ff0000' for val in self.df['MACD_Histogram']]
        ax.bar(self.df.index, self.df['MACD_Histogram'], color=colors, alpha=0.3, label='Histograma')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=0.5)
        ax.set_xlabel('Fecha', fontsize=9)
        ax.set_ylabel('MACD', fontsize=9)
        ax.legend(loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.2)

    def run_complete_analysis(self):
        """Ejecutar análisis completo de ondas de Elliott"""
        if self.df is None:
            messagebox.showwarning("Advertencia", "Primero cargue los datos")
            return

        try:
            self.info_text.delete(1.0, tk.END)
            self.points_text.delete(1.0, tk.END)
            self.trading_signals.clear()
            self.all_signals.clear()

            self.info_text.insert(tk.END, "🔄 Ejecutando análisis completo...\n\n")
            self.root.update()

            wave_pattern = self.detect_elliott_waves_advanced()
            self.calculate_fibonacci_levels()
            self.add_bollinger_analysis()

            if wave_pattern:
                self.identify_buy_sell_points(wave_pattern)

            self.detect_trading_signals()
            self.update_complete_chart()

            self.info_text.insert(tk.END, "\n✅ Análisis completo finalizado\n")

        except Exception as e:
            messagebox.showerror("Error", f"Error en análisis: {str(e)}")

    def detect_elliott_waves_advanced(self):
        """Detectar ondas de Elliott con algoritmo mejorado"""
        if self.df is None:
            return None

        try:
            prices = self.df['Close'].values
            from scipy.signal import argrelextrema

            order = max(5, len(prices) // 50)

            local_maxima = argrelextrema(prices, np.greater, order=order)[0]
            local_minima = argrelextrema(prices, np.less, order=order)[0]

            extrema = []
            for idx in local_maxima:
                extrema.append((idx, prices[idx], 'max'))
            for idx in local_minima:
                extrema.append((idx, prices[idx], 'min'))
            extrema.sort(key=lambda x: x[0])

            filtered_extrema = []
            if extrema:
                filtered_extrema.append(extrema[0])
                for i in range(1, len(extrema)):
                    if abs(extrema[i][0] - filtered_extrema[-1][0]) > order // 2:
                        filtered_extrema.append(extrema[i])

            best_pattern = self.find_best_elliott_pattern(filtered_extrema, prices)

            if best_pattern:
                self.wave_points = best_pattern
                self.display_wave_analysis(best_pattern)
                return best_pattern

            return None

        except Exception as e:
            self.info_text.insert(tk.END, f"\n❌ Error detectando ondas: {str(e)}\n")
            return None

    def find_best_elliott_pattern(self, extrema, prices):
        """Encontrar el mejor patrón de Elliott"""
        if len(extrema) < 5:
            return None

        best_pattern = None
        best_score = -1

        for start in range(len(extrema) - 4):
            pattern = extrema[start:start + 5]
            types = [p[2] for p in pattern]
            if types == ['min', 'max', 'min', 'max', 'min'] or \
                    types == ['max', 'min', 'max', 'min', 'max']:
                score = self.score_elliott_pattern(pattern)
                if score > best_score:
                    best_score = score
                    best_pattern = pattern

        if best_pattern and len(extrema) > extrema.index(best_pattern[-1]) + 1:
            remaining = extrema[extrema.index(best_pattern[-1]) + 1:]
            if len(remaining) >= 3:
                best_pattern.extend(remaining[:3])

        return best_pattern

    def score_elliott_pattern(self, pattern):
        """Puntuar el patrón de Elliott"""
        score = 0
        prices = [p[1] for p in pattern]

        wave1_length = abs(prices[1] - prices[0])
        wave3_length = abs(prices[2] - prices[1])
        wave5_length = abs(prices[4] - prices[3]) if len(prices) > 4 else 0

        if wave3_length >= min(wave1_length, wave5_length):
            score += 40

        wave2_retrace = abs(prices[2] - prices[1]) / abs(prices[1] - prices[0]) if prices[1] != prices[0] else 0
        if 0.382 <= wave2_retrace <= 0.618:
            score += 30

        if pattern[0][2] == 'min':
            if len(prices) > 3 and prices[3] > prices[1]:
                score += 30

        return score

    def display_wave_analysis(self, wave_pattern):
        """Mostrar análisis de ondas detectadas"""
        self.info_text.insert(tk.END, "\n" + "=" * 50 + "\n")
        self.info_text.insert(tk.END, "     ONDAS DE ELLIOTT DETECTADAS\n")
        self.info_text.insert(tk.END, "=" * 50 + "\n")

        wave_labels = ['1', '2', '3', '4', '5', 'A', 'B', 'C']

        for i, (idx, price, type_) in enumerate(wave_pattern[:8]):
            if i < len(wave_labels):
                date = self.df.index[idx]
                emoji = "📈" if type_ == 'max' else "📉"
                self.info_text.insert(tk.END,
                                      f"{emoji} Onda {wave_labels[i]}: ${price:.2f} - {date.strftime('%Y-%m-%d')}\n")

    def calculate_fibonacci_levels(self):
        """Calcular niveles de Fibonacci"""
        if self.df is None:
            return

        high = self.df['High'].max()
        low = self.df['Low'].min()
        diff = high - low

        self.info_text.insert(tk.END, "\n" + "=" * 50 + "\n")
        self.info_text.insert(tk.END, "     NIVELES DE FIBONACCI\n")
        self.info_text.insert(tk.END, "=" * 50 + "\n")

        for level in self.fibonacci_levels:
            price = low + (diff * level)
            if level <= 1.0:
                self.info_text.insert(tk.END, f"📊 Retroceso {level:.1%}: ${price:.2f}\n")
            else:
                self.info_text.insert(tk.END, f"🎯 Extensión {level:.1%}: ${price:.2f}\n")

    def add_bollinger_analysis(self):
        """Análisis de Bandas de Bollinger"""
        if 'BB_Upper' not in self.df.columns:
            return

        current_price = self.df['Close'].iloc[-1]
        bb_upper = self.df['BB_Upper'].iloc[-1]
        bb_lower = self.df['BB_Lower'].iloc[-1]

        b_percent = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5

        self.info_text.insert(tk.END, "\n" + "=" * 50 + "\n")
        self.info_text.insert(tk.END, "     BANDAS DE BOLLINGER\n")
        self.info_text.insert(tk.END, "=" * 50 + "\n")
        self.info_text.insert(tk.END, f"Banda Superior: ${bb_upper:.2f}\n")
        self.info_text.insert(tk.END, f"Banda Inferior: ${bb_lower:.2f}\n")
        self.info_text.insert(tk.END, f"Indicador %B: {b_percent:.2%}\n")

    def identify_buy_sell_points(self, wave_pattern):
        """Identificar puntos óptimos de compra y venta según Elliott"""
        if not wave_pattern:
            return

        self.points_text.insert(tk.END, "=" * 60 + "\n")
        self.points_text.insert(tk.END, "   PUNTOS ÓPTIMOS DE COMPRA Y VENTA\n")
        self.points_text.insert(tk.END, "=" * 60 + "\n\n")

        current_price = self.df['Close'].iloc[-1]
        prices = [p[1] for p in wave_pattern]

        current_wave = self.identify_current_wave(wave_pattern, current_price)

        self.points_text.insert(tk.END, "🟢 PUNTOS DE COMPRA ÓPTIMOS:\n" + "-" * 40 + "\n")

        if len(prices) > 2:
            wave2_price = prices[2]
            wave1_length = abs(prices[1] - prices[0])
            buy_zone_start = prices[0] + (wave1_length * 0.5)
            buy_zone_end = prices[0] + (wave1_length * 0.618)
            stop_pct = self.max_profit_config.get('stop_loss_percent', 3.0) / 100.0

            self.points_text.insert(tk.END, f"""
📍 FINAL DE ONDA 2 ⭐⭐⭐⭐⭐ (MEJOR ENTRADA)
   • Zona de compra: ${buy_zone_start:.2f} - ${buy_zone_end:.2f}
   • Stop Loss: ${buy_zone_end * (1 - stop_pct):.2f} (-{stop_pct*100:.1f}%)
   • Objetivo 1: ${prices[1] + (wave1_length * 0.618):.2f}
   • Objetivo 2: ${prices[1] + (wave1_length * 1.618):.2f}
   • Objetivo 3: ${prices[1] + (wave1_length * 2.618):.2f}
            \n""")

        if len(prices) > 4:
            wave3_length = abs(prices[2] - prices[1])
            wave4_retrace_382 = prices[2] - (wave3_length * 0.382)
            wave4_retrace_50 = prices[2] - (wave3_length * 0.5)
            stop_pct = self.max_profit_config.get('stop_loss_percent', 3.0) / 100.0

            self.points_text.insert(tk.END, f"""
📍 FINAL DE ONDA 4 ⭐⭐⭐⭐ (MUY BUENA ENTRADA)
   • Zona de compra: ${wave4_retrace_50:.2f} - ${wave4_retrace_382:.2f}
   • Stop Loss: ${wave4_retrace_382 * (1 - stop_pct):.2f} (-{stop_pct*100:.1f}%)
   • Objetivo (Onda 5): ${prices[2] + (prices[1] - prices[0]):.2f}
            \n""")

        self.points_text.insert(tk.END, "\n🔴 PUNTOS DE VENTA ÓPTIMOS:\n" + "-" * 40 + "\n")

        if len(prices) > 2:
            wave1_length = abs(prices[1] - prices[0])
            target_wave3_1 = prices[0] + (wave1_length * 1.618)
            target_wave3_2 = prices[0] + (wave1_length * 2.618)

            self.points_text.insert(tk.END, f"""
📍 FINAL DE ONDA 3 (VENTA PARCIAL 50%)
   • Objetivo 1: ${target_wave3_1:.2f} (1.618 × Onda 1)
   • Objetivo 2: ${target_wave3_2:.2f} (2.618 × Onda 1)
   • Señales: RSI > 70, Divergencia bajista
   • Acción: Vender 50%, mover stop a breakeven
            \n""")

        if len(prices) > 4:
            target_wave5 = prices[2] + (prices[1] - prices[0])

            self.points_text.insert(tk.END, f"""
📍 FINAL DE ONDA 5 (VENTA TOTAL)
   • Objetivo: ${target_wave5:.2f} (Onda 5 = Onda 1)
   • Señales: Divergencia extrema, volumen decreciente
   • Acción: Cerrar 100% posiciones largas
            \n""")

        self.points_text.insert(tk.END, "\n" + "=" * 60 + "\n")
        self.points_text.insert(tk.END, "   SITUACIÓN ACTUAL Y RECOMENDACIÓN\n")
        self.points_text.insert(tk.END, "=" * 60 + "\n\n")

        self.points_text.insert(tk.END, f"💹 Precio actual: ${current_price:.2f}\n")
        self.points_text.insert(tk.END, f"📍 Onda actual estimada: {current_wave}\n\n")

        recommendation = self.get_trading_recommendation(current_wave, current_price, wave_pattern)
        self.points_text.insert(tk.END, recommendation)

    def identify_current_wave(self, wave_pattern, current_price):
        """Identificar en qué onda estamos actualmente"""
        if not wave_pattern:
            return "Indeterminada"

        prices = [p[1] for p in wave_pattern]
        indices = [p[0] for p in wave_pattern]

        current_idx = len(self.df) - 1
        wave_labels = ['1', '2', '3', '4', '5', 'A', 'B', 'C']

        for i in range(len(indices) - 1):
            if indices[i] <= current_idx < indices[i + 1]:
                return f"Entre onda {wave_labels[i]} y {wave_labels[i + 1]}"

        if current_idx >= indices[-1]:
            return f"Después de onda {wave_labels[min(len(indices) - 1, len(wave_labels) - 1)]}"

        return "Inicio del patrón"

    def get_trading_recommendation(self, current_wave, current_price, wave_pattern):
        """Obtener recomendación de trading basada en la situación actual"""
        rsi = self.df['RSI'].iloc[-1] if 'RSI' in self.df.columns else 50

        recommendation = "🎯 RECOMENDACIÓN:\n" + "-" * 40 + "\n"

        if "2" in current_wave and "3" in current_wave:
            recommendation += """
✅ SEÑAL DE COMPRA FUERTE
• Estamos en zona óptima de entrada (final Onda 2)
• Esperar confirmación de reversión alcista
• RSI debe estar en sobreventa (<40)
• Configurar stop loss 3% por debajo
• Objetivos: Niveles de Fibonacci 1.618 y 2.618
• Risk/Reward esperado: 1:5 o mejor
            """
        elif "4" in current_wave and "5" in current_wave:
            recommendation += """
✅ SEÑAL DE COMPRA MODERADA
• Posible última oportunidad de entrada
• Confirmar que Onda 4 no invade Onda 1
• Stop loss en mínimo de Onda 4
• Objetivo: Onda 5 = Onda 1
• Risk/Reward esperado: 1:3
            """
        elif "3" in current_wave:
            if rsi > 70:
                recommendation += """
⚠️ SEÑAL DE VENTA PARCIAL
• Onda 3 en zona de sobrecompra
• Considerar tomar ganancias parciales (50%)
• Mover stop loss a breakeven
            """
            else:
                recommendation += """
➡️ MANTENER POSICIÓN
• Onda 3 en desarrollo
• Dejar correr las ganancias
• Ajustar stop loss progresivamente
            """
        elif "5" in current_wave:
            recommendation += """
🔴 SEÑAL DE VENTA FUERTE
• Final de ciclo alcista
• Cerrar todas las posiciones largas
• Prepararse para corrección ABC
• Considerar posiciones cortas con confirmación
            """
        elif "C" in current_wave:
            recommendation += """
✅ PREPARAR NUEVA ENTRADA
• Corrección ABC cerca de completarse
• Buscar divergencia alcista en RSI
• Nuevo ciclo de 5 ondas próximo
• Preparar capital para entrada
            """
        else:
            recommendation += """
⏳ ESPERAR Y OBSERVAR
• Patrón en desarrollo
• No hay señal clara actualmente
• Mantener análisis actualizado
• Esperar mejor punto de entrada
            """

        recommendation += f"\n\n📊 RSI Actual: {rsi:.1f}"
        if rsi > 70:
            recommendation += " (SOBRECOMPRA ⚠️)"
        elif rsi < 30:
            recommendation += " (SOBREVENTA ✅)"
        else:
            recommendation += " (NEUTRAL)"

        return recommendation

    def detect_trading_signals(self):
        """Detectar y mostrar señales de trading automáticas"""
        if self.df is None or not self.wave_points:
            return

        current_price = self.df['Close'].iloc[-1]

        self.trading_signals.clear()
        self.all_signals.clear()

        for item in self.signals_tree.get_children():
            self.signals_tree.delete(item)

        signals = []

        elliott_signal = self.check_elliott_signal(current_price)
        if elliott_signal:
            signals.append(elliott_signal)

        fib_signal = self.check_fibonacci_signal(current_price)
        if fib_signal:
            signals.append(fib_signal)

        bb_signal = self.check_bollinger_signal(current_price)
        if bb_signal:
            signals.append(bb_signal)

        rsi_signal = self.check_rsi_signal()
        if rsi_signal:
            signals.append(rsi_signal)

        macd_signal = self.check_macd_signal()
        if macd_signal:
            signals.append(macd_signal)

        self.evaluate_signals(signals)
        self.show_signals_summary(self.weighted_decision)
        self.check_alerts(self.weighted_decision)

    def _stop_pct(self) -> float:
        return float(self.max_profit_config.get('stop_loss_percent', 3.0)) / 100.0

    def check_elliott_signal(self, current_price):
        """Verificar señal basada en ondas de Elliott"""
        if not self.wave_points:
            return None

        prices = [p[1] for p in self.wave_points]
        current_wave = self.identify_current_wave(self.wave_points, current_price)
        stop_pct = self._stop_pct()

        if "2" in current_wave and len(prices) > 2:
            wave1_length = abs(prices[1] - prices[0])
            target = prices[0] + (wave1_length * 0.618)

            if abs(current_price - target) / target < 0.02:
                return TradingSignal(
                    tipo="COMPRA",
                    precio=current_price,
                    fecha=self.df.index[-1],
                    razon="Elliott: Final de Onda 2 - Zona 61.8%",
                    fuerza=5,
                    objetivo1=prices[1] + (wave1_length * 0.618),
                    objetivo2=prices[1] + (wave1_length * 1.618),
                    objetivo3=prices[1] + (wave1_length * 2.618),
                    stop_loss=current_price * (1 - stop_pct),
                    source="ELLIOTT",
                )

        elif "5" in current_wave and len(prices) > 4:
            return TradingSignal(
                tipo="VENTA",
                precio=current_price,
                fecha=self.df.index[-1],
                razon="Elliott: Final de Onda 5 - Fin de ciclo",
                fuerza=4,
                objetivo1=current_price * 0.95,
                objetivo2=current_price * 0.90,
                objetivo3=current_price * 0.85,
                stop_loss=current_price * (1 + stop_pct),
                source="ELLIOTT",
            )

        return None

    def check_fibonacci_signal(self, current_price):
        """Verificar señal basada en niveles de Fibonacci"""
        high = self.df['High'].max()
        low = self.df['Low'].min()
        diff = high - low
        stop_pct = self._stop_pct()

        key_levels = [0.382, 0.5, 0.618]

        for level in key_levels:
            fib_price = low + (diff * level)
            if abs(current_price - fib_price) / fib_price < 0.01:
                if current_price > self.df['Close'].iloc[-2]:
                    return TradingSignal(
                        tipo="COMPRA",
                        precio=current_price,
                        fecha=self.df.index[-1],
                        razon=f"Fibonacci: Rebote en {level:.1%}",
                        fuerza=3,
                        objetivo1=low + (diff * (level + 0.236)),
                        objetivo2=low + (diff * (level + 0.382)),
                        objetivo3=low + (diff * min(1.0, level + 0.618)),
                        stop_loss=current_price * (1 - stop_pct),
                        source="FIBONACCI",
                    )

        return None

    def check_bollinger_signal(self, current_price):
        """Verificar señal basada en Bandas de Bollinger"""
        if 'BB_Upper' not in self.df.columns:
            return None

        bb_upper = self.df['BB_Upper'].iloc[-1]
        bb_lower = self.df['BB_Lower'].iloc[-1]
        bb_middle = self.df['BB_Middle'].iloc[-1]
        stop_pct = self._stop_pct()

        if current_price <= bb_lower:
            return TradingSignal(
                tipo="COMPRA",
                precio=current_price,
                fecha=self.df.index[-1],
                razon="Bollinger: Toque banda inferior",
                fuerza=3,
                objetivo1=bb_middle,
                objetivo2=(bb_middle + bb_upper) / 2,
                objetivo3=bb_upper,
                stop_loss=current_price * (1 - stop_pct),
                source="BOLLINGER",
            )
        elif current_price >= bb_upper:
            return TradingSignal(
                tipo="VENTA",
                precio=current_price,
                fecha=self.df.index[-1],
                razon="Bollinger: Toque banda superior",
                fuerza=3,
                objetivo1=bb_middle,
                objetivo2=(bb_middle + bb_lower) / 2,
                objetivo3=bb_lower,
                stop_loss=current_price * (1 + stop_pct),
                source="BOLLINGER",
            )

        return None

    def check_rsi_signal(self):
        """Verificar señal basada en RSI"""
        if 'RSI' not in self.df.columns:
            return None

        rsi = self.df['RSI'].iloc[-1]
        rsi_prev = self.df['RSI'].iloc[-2]
        stop_pct = self._stop_pct()

        if rsi < 30 and rsi > rsi_prev:
            return TradingSignal(
                tipo="COMPRA",
                precio=self.df['Close'].iloc[-1],
                fecha=self.df.index[-1],
                razon="RSI: Sobreventa (<30) con reversión",
                fuerza=4,
                objetivo1=self.df['Close'].iloc[-1] * 1.03,
                objetivo2=self.df['Close'].iloc[-1] * 1.05,
                objetivo3=self.df['Close'].iloc[-1] * 1.08,
                stop_loss=self.df['Close'].iloc[-1] * (1 - stop_pct),
                source="RSI",
            )
        elif rsi > 70 and rsi < rsi_prev:
            return TradingSignal(
                tipo="VENTA",
                precio=self.df['Close'].iloc[-1],
                fecha=self.df.index[-1],
                razon="RSI: Sobrecompra (>70) con reversión",
                fuerza=3,
                objetivo1=self.df['Close'].iloc[-1] * 0.97,
                objetivo2=self.df['Close'].iloc[-1] * 0.95,
                objetivo3=self.df['Close'].iloc[-1] * 0.92,
                stop_loss=self.df['Close'].iloc[-1] * (1 + stop_pct),
                source="RSI",
            )

        return None

    def check_macd_signal(self):
        """Verificar señal basada en MACD"""
        if 'MACD' not in self.df.columns:
            return None

        macd = self.df['MACD'].iloc[-1]
        signal = self.df['Signal'].iloc[-1]
        macd_prev = self.df['MACD'].iloc[-2]
        signal_prev = self.df['Signal'].iloc[-2]
        stop_pct = self._stop_pct()

        if macd > signal and macd_prev <= signal_prev:
            return TradingSignal(
                tipo="COMPRA",
                precio=self.df['Close'].iloc[-1],
                fecha=self.df.index[-1],
                razon="MACD: Cruce alcista",
                fuerza=3,
                objetivo1=self.df['Close'].iloc[-1] * 1.02,
                objetivo2=self.df['Close'].iloc[-1] * 1.04,
                objetivo3=self.df['Close'].iloc[-1] * 1.06,
                stop_loss=self.df['Close'].iloc[-1] * (1 - stop_pct),
                source="MACD",
            )
        elif macd < signal and macd_prev >= signal_prev:
            return TradingSignal(
                tipo="VENTA",
                precio=self.df['Close'].iloc[-1],
                fecha=self.df.index[-1],
                razon="MACD: Cruce bajista",
                fuerza=3,
                objetivo1=self.df['Close'].iloc[-1] * 0.98,
                objetivo2=self.df['Close'].iloc[-1] * 0.96,
                objetivo3=self.df['Close'].iloc[-1] * 0.94,
                stop_loss=self.df['Close'].iloc[-1] * (1 + stop_pct),
                source="MACD",
            )

        return None

    def _aggregate_weighted_recommendation(self, signals):
        """Construye recomendación ponderada priorizando Elliott y filtrando por R:R mínimo."""
        min_rr = float(self.max_profit_config.get('min_risk_reward', 3.0))
        actionable = [s for s in signals if s.risk_reward >= min_rr]

        buy_score = 0.0
        sell_score = 0.0
        details = []

        for s in actionable:
            base_weight = float(self.signal_weights.get(s.source, 0.10))
            strength_factor = max(0.0, min(1.0, s.fuerza / 5.0))
            rr_factor = max(0.0, min(1.0, min(s.risk_reward / min_rr, 2.0) / 2.0))
            # Combina fuerza y R:R (60/40)
            combined = (0.6 * strength_factor) + (0.4 * rr_factor)
            score = base_weight * combined

            details.append((s, score))
            if s.tipo == 'COMPRA':
                buy_score += score
            else:
                sell_score += score

        # Clasificación
        action = 'ESPERAR'
        classification = 'SIN SEÑAL ACCIONABLE'
        chosen = None
        confidence = 0

        if buy_score > sell_score and buy_score >= 0.30:
            action = 'COMPRA'
            chosen = max([s for s in actionable if s.tipo == 'COMPRA'], key=lambda x: (x.risk_reward, x.fuerza, self.signal_weights.get(x.source, 0)), default=None)
        elif sell_score > buy_score and sell_score >= 0.30:
            action = 'VENTA'
            chosen = max([s for s in actionable if s.tipo == 'VENTA'], key=lambda x: (x.risk_reward, x.fuerza, self.signal_weights.get(x.source, 0)), default=None)

        dominance = abs(buy_score - sell_score)
        top_score = max(buy_score, sell_score)

        if action != 'ESPERAR':
            elliott_present = any(s.source == 'ELLIOTT' and s.tipo == action for s in actionable)
            if top_score >= 0.55 and elliott_present:
                classification = f"{action} FUERTE"
            elif top_score >= 0.40:
                classification = f"{action} MODERADA"
            else:
                classification = f"{action} DÉBIL"

            # Confianza aproximada 0-100
            confidence = int(max(30, min(95, (top_score * 100) + (dominance * 50))))

        return {
            'action': action,
            'classification': classification,
            'confidence': confidence,
            'chosen_signal': chosen,
            'actionable_signals': actionable,
            'raw_signals': signals,
            'scores': {
                'buy': round(buy_score, 3),
                'sell': round(sell_score, 3),
            }
        }

    def evaluate_signals(self, signals):
        """Evaluar señales con consenso ponderado y filtrar por R:R mínimo"""
        self.all_signals = list(signals)

        # Mostrar todas en el árbol, clasificando las no accionables (gris)
        min_rr = float(self.max_profit_config.get('min_risk_reward', 3.0))
        for idx, s in enumerate(signals, start=1):
            is_actionable = s.risk_reward >= min_rr
            tipo_desc = ("COMPRA" if s.tipo == 'COMPRA' else "VENTA") + ("" if is_actionable else " (NO ACCIÓN R:R<" + str(min_rr) + ")")
            self.add_signal_to_tree(s, tipo_desc, is_actionable=is_actionable)

        decision = self._aggregate_weighted_recommendation(signals)
        self.weighted_decision = decision

        # Solo señales accionables se guardan para pintar en el gráfico
        self.trading_signals = decision['actionable_signals']

    def add_signal_to_tree(self, signal, tipo_desc, is_actionable=True):
        """Añadir señal al TreeView"""
        tag = 'buy' if signal.tipo == "COMPRA" else 'sell'
        if not is_actionable:
            tag = 'filtered'

        self.signals_tree.insert('', 'end',
                                 values=(
                                     signal.fecha.strftime('%Y-%m-%d %H:%M'),
                                     tipo_desc,
                                     f"${signal.precio:.2f}",
                                     f"${signal.stop_loss:.2f}",
                                     f"${signal.objetivo1:.2f}" if signal.objetivo1 is not None else '-',
                                     f"${signal.objetivo2:.2f}" if signal.objetivo2 is not None else '-',
                                     f"${signal.objetivo3:.2f}" if signal.objetivo3 is not None else '-',
                                     f"{signal.risk_reward:.1f}:1",
                                     "⭐" * signal.fuerza,
                                     f"[{signal.source}] {signal.razon}"
                                 ),
                                 tags=(tag,)
                                 )

        self.signals_tree.tag_configure('buy', foreground='green')
        self.signals_tree.tag_configure('sell', foreground='red')
        self.signals_tree.tag_configure('filtered', foreground='gray')

    def _build_indicator_snapshot(self):
        """Construye un breve snapshot de indicadores para el reporte."""
        snap = []
        price = self.df['Close'].iloc[-1]
        snap.append(f"Precio: ${price:.2f}")
        if 'RSI' in self.df.columns:
            rsi = self.df['RSI'].iloc[-1]
            rsi_state = 'Sobreventa' if rsi < 30 else ('Sobrecompra' if rsi > 70 else 'Neutral')
            snap.append(f"RSI: {rsi:.1f} ({rsi_state})")
        if 'MACD' in self.df.columns:
            macd = self.df['MACD'].iloc[-1]
            signal = self.df['Signal'].iloc[-1]
            snap.append(f"MACD: {macd:.2f} vs Signal {signal:.2f}")
        if 'BB_Upper' in self.df.columns:
            bb_upper = self.df['BB_Upper'].iloc[-1]
            bb_lower = self.df['BB_Lower'].iloc[-1]
            b_percent = (price - bb_lower) / (bb_upper - bb_lower) * 100 if bb_upper != bb_lower else 50
            snap.append(f"%B: {b_percent:.1f}%")
        return ' | '.join(snap)

    def generate_intelligent_report(self, decision):
        """Genera un reporte inteligente en español con pesos y R:R."""
        if decision is None:
            return "No hay datos suficientes para generar reporte."

        min_rr = float(self.max_profit_config.get('min_risk_reward', 3.0))
        lines = []
        lines.append("📑 REPORTE INTELIGENTE")
        lines.append("═" * 40)
        lines.append(f"Clasificación Ponderada: {decision['classification']}")
        if decision['action'] != 'ESPERAR':
            lines.append(f"Confianza: {decision['confidence']}%")
        lines.append(f"Scores → Compra: {decision['scores']['buy']:.3f} | Venta: {decision['scores']['sell']:.3f}")
        lines.append(self._build_indicator_snapshot())
        lines.append("")

        if decision['chosen_signal'] is not None:
            s = decision['chosen_signal']
            lines.append("Señal ejecutable (priorizada):")
            lines.append(f"• Tipo: {s.tipo} | Fuente: {s.source} | Fuerza: {'⭐'*s.fuerza}")
            lines.append(f"• Precio: ${s.precio:.2f} | Stop: ${s.stop_loss:.2f} | Mejor R:R: {s.risk_reward:.1f}:1")
            lines.append(f"• Objetivos: " + ", ".join([f"${t:.2f}" for t in s.targets]))
        else:
            lines.append(f"No hay señales accionables con R:R ≥ {min_rr:.1f}:1. Esperar.")

        lines.append("")
        lines.append("Reglas aplicadas:")
        lines.append(f"• Filtro de R:R mínimo: {min_rr:.1f}:1")
        lines.append("• Ponderación prioriza Ondas de Elliott (50%)")
        lines.append("• Confirmaciones complementarias: Fibonacci, MACD, RSI, Bollinger")

        return "\n".join(lines)

    def show_signals_summary(self, decision=None):
        """Mostrar resumen de señales y reporte inteligente"""
        self.summary_text.delete(1.0, tk.END)

        buy_count = sum(1 for s in self.trading_signals if s.tipo == "COMPRA")
        sell_count = sum(1 for s in self.trading_signals if s.tipo == "VENTA")

        summary = f"""
📊 RESUMEN DE SEÑALES
═══════════════════════════════════
🟢 Señales de COMPRA (accionables): {buy_count}
🔴 Señales de VENTA (accionables): {sell_count}
📈 Mejor R:R: {max(self.trading_signals, key=lambda s: s.risk_reward).risk_reward:.1f}:1 if self.trading_signals else '—'
        """
        # Ajuste de la línea de Mejor R:R si no hay señales
        if self.trading_signals:
            best_rr = max(self.trading_signals, key=lambda s: s.risk_reward).risk_reward
            best_line = f"💹 Mejor R:R: {best_rr:.1f}:1"
        else:
            best_line = "💹 Mejor R:R: —"

        summary = summary.replace("📈 Mejor R:R: {max(self.trading_signals, key=lambda s: s.risk_reward).risk_reward:.1f}:1 if self.trading_signals else '—'", best_line)
        self.summary_text.insert(tk.END, summary)

        report = self.generate_intelligent_report(decision)
        self.summary_text.insert(tk.END, "\n" + report + "\n")

    def check_alerts(self, decision=None):
        """Verificar y mostrar alertas"""
        if not self.alert_enabled.get():
            return

        if decision is None:
            self.alert_label.config(text="Esperando señales...")
            return

        min_rr = float(self.max_profit_config.get('min_risk_reward', 3.0))

        if decision['action'] in ('COMPRA', 'VENTA') and decision['chosen_signal'] is not None and decision['chosen_signal'].risk_reward >= min_rr:
            s = decision['chosen_signal']
            alert_text = f"""
🚨 ALERTA DE TRADING {decision['classification']} 🚨
Tipo: {s.tipo} | Fuente: {s.source}
Precio: ${s.precio:.2f}
Fuerza: {'⭐' * s.fuerza}
Stop Loss: ${s.stop_loss:.2f}
Objetivo 1: ${s.objetivo1:.2f}
Mejor R:R: {s.risk_reward:.1f}:1 | Confianza: {decision['confidence']}%
            """
            self.alert_label.config(text=alert_text, style='Alert.TLabel')

            if SOUND_ENABLED:
                try:
                    pass
                except Exception:
                    pass

            self.flash_window()
        else:
            self.alert_label.config(text="Señales débiles o sin R:R suficiente - Esperar")

    def flash_window(self):
        """Hacer parpadear la ventana para alertar"""

        def flash():
            for _ in range(3):
                try:
                    self.root.bell()
                except Exception:
                    pass
                time.sleep(0.3)

        threading.Thread(target=flash, daemon=True).start()

    def update_complete_chart(self):
        """Actualizar gráfico con todas las señales y análisis según preferencias"""
        if not hasattr(self, 'ax') or self.df is None:
            return

        self.plot_basic_chart()

        if hasattr(self, 'show_elliott') and self.show_elliott.get() and self.wave_points:
            wave_labels = ['1', '2', '3', '4', '5', 'A', 'B', 'C']
            colors_elliott = {
                '1': '#00b050',
                '2': '#ff4444',
                '3': '#00b050',
                '4': '#ff4444',
                '5': '#00b050',
                'A': '#ff8800',
                'B': '#ff8800',
                'C': '#ff8800'
            }

            for i, (idx, price, type_) in enumerate(self.wave_points[:8]):
                if i < len(wave_labels):
                    date = self.df.index[idx]
                    label = wave_labels[i]
                    color = colors_elliott[label]

                    self.ax.scatter(date, price, color=color, s=150, zorder=10,
                                    edgecolor='black', linewidth=2, alpha=0.9)

                    self.ax.annotate(label, (date, price),
                                     xytext=(0, 0), textcoords='offset points',
                                     fontsize=12, fontweight='bold', color='white',
                                     ha='center', va='center', zorder=11)

                    if i > 0:
                        prev_idx, prev_price, _ = self.wave_points[i - 1]
                        prev_date = self.df.index[prev_idx]
                        self.ax.plot([prev_date, date], [prev_price, price],
                                     color='black', alpha=0.5, linestyle='--',
                                     linewidth=1.5, zorder=4)

        if hasattr(self, 'show_fibonacci') and self.show_fibonacci.get():
            high = self.df['High'].max()
            low = self.df['Low'].min()
            diff = high - low

            fib_colors = {
                0.0: ('darkgreen', 'Soporte Base'),
                0.236: ('#00b050', '23.6%'),
                0.382: ('#40ff40', '38.2%'),
                0.5: ('#ffff00', '50%'),
                0.618: ('#ff8800', '61.8% KEY'),
                0.786: ('#ff4444', '78.6%'),
                1.0: ('darkred', 'Resistencia'),
                1.618: ('purple', 'Extensión 161.8%'),
                2.618: ('magenta', 'Extensión 261.8%')
            }

            for level, (color, desc) in fib_colors.items():
                if level <= 2.618:
                    price = low + (diff * level)
                    self.ax.axhline(y=price, color=color, linestyle='--' if level <= 1 else ':',
                                    alpha=0.4, linewidth=1, zorder=2)
                    self.ax.text(self.df.index[-1], price, f' {desc}: ${price:.2f}',
                                 fontsize=8, va='center', ha='left',
                                 bbox=dict(boxstyle="round,pad=0.2",
                                           facecolor='white', edgecolor=color,
                                           alpha=0.8), zorder=3)

        if hasattr(self, 'show_bollinger') and self.show_bollinger.get() and 'BB_Upper' in self.df.columns:
            self.ax.plot(self.df.index, self.df['BB_Upper'],
                         color='green', linestyle='-', alpha=0.6,
                         linewidth=1.5, label='BB Superior', zorder=5)
            self.ax.plot(self.df.index, self.df['BB_Middle'],
                         color='blue', linestyle='-', alpha=0.5,
                         linewidth=1, label='BB Media', zorder=5)
            self.ax.plot(self.df.index, self.df['BB_Lower'],
                         color='red', linestyle='-', alpha=0.6,
                         linewidth=1.5, label='BB Inferior', zorder=5)
            self.ax.fill_between(self.df.index, self.df['BB_Upper'], self.df['BB_Lower'],
                                 alpha=0.1, color='gray', zorder=1)

            recent_data = self.df.iloc[-20:]
            upper_touches = recent_data[recent_data['Close'] > recent_data['BB_Upper']]
            for date in upper_touches.index:
                self.ax.scatter(date, recent_data.loc[date, 'Close'],
                                color='red', s=50, marker='v', zorder=8,
                                edgecolor='black', linewidth=1)
            lower_touches = recent_data[recent_data['Close'] < recent_data['BB_Lower']]
            for date in lower_touches.index:
                self.ax.scatter(date, recent_data.loc[date, 'Close'],
                                color='green', s=50, marker='^', zorder=8,
                                edgecolor='black', linewidth=1)

        if hasattr(self, 'show_signals') and self.show_signals.get() and self.trading_signals:
            for signal in self.trading_signals:
                if signal.tipo == "COMPRA":
                    self.ax.scatter(signal.fecha, signal.precio,
                                    marker='^', s=300, color='lime',
                                    edgecolor='darkgreen', linewidth=2, zorder=9)

                    self.ax.annotate('BUY', (signal.fecha, signal.precio),
                                     xytext=(0, -20), textcoords='offset points',
                                     fontsize=10, fontweight='bold', color='darkgreen',
                                     bbox=dict(boxstyle="round,pad=0.3",
                                               facecolor='lightgreen', edgecolor='darkgreen',
                                               alpha=0.9), zorder=10,
                                     arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1))

                    self.ax.axhline(y=signal.objetivo1, color='green',
                                    linestyle=':', alpha=0.3, linewidth=0.8)
                    self.ax.axhline(y=signal.stop_loss, color='red',
                                    linestyle=':', alpha=0.3, linewidth=0.8)

                else:
                    self.ax.scatter(signal.fecha, signal.precio,
                                    marker='v', s=300, color='red',
                                    edgecolor='darkred', linewidth=2, zorder=9)

                    self.ax.annotate('SELL', (signal.fecha, signal.precio),
                                     xytext=(0, 20), textcoords='offset points',
                                     fontsize=10, fontweight='bold', color='darkred',
                                     bbox=dict(boxstyle="round,pad=0.3",
                                               facecolor='lightcoral', edgecolor='darkred',
                                               alpha=0.9), zorder=10,
                                     arrowprops=dict(arrowstyle='->', color='darkred', lw=1))

        info_text = self.build_chart_info()
        if info_text:
            self.ax.text(0.02, 0.98, info_text,
                         transform=self.ax.transAxes,
                         fontsize=8, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white',
                                   edgecolor='gray', alpha=0.9),
                         zorder=12)

        if hasattr(self, 'show_sma') and self.show_sma.get():
            self.ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

        self.canvas.draw()

    def build_chart_info(self):
        """Construir texto de información para mostrar en el gráfico"""
        if self.df is None:
            return ""

        current_price = self.df['Close'].iloc[-1]
        rsi = self.df['RSI'].iloc[-1] if 'RSI' in self.df.columns else 0

        info_lines = []

        if rsi > 70:
            info_lines.append(f"⚠️ RSI: {rsi:.1f} (SOBRECOMPRA)")
        elif rsi < 30:
            info_lines.append(f"✅ RSI: {rsi:.1f} (SOBREVENTA)")
        else:
            info_lines.append(f"📊 RSI: {rsi:.1f}")

        if 'BB_Upper' in self.df.columns:
            bb_upper = self.df['BB_Upper'].iloc[-1]
            bb_lower = self.df['BB_Lower'].iloc[-1]
            bb_middle = self.df['BB_Middle'].iloc[-1]

            if current_price >= bb_upper:
                info_lines.append("🔴 Tocando Banda Superior")
            elif current_price <= bb_lower:
                info_lines.append("🟢 Tocando Banda Inferior")
            else:
                b_percent = (current_price - bb_lower) / (bb_upper - bb_lower) * 100
                info_lines.append(f"📍 %B: {b_percent:.1f}%")

        if self.wave_points:
            current_wave = self.identify_current_wave(self.wave_points, current_price)
            info_lines.append(f"🌊 {current_wave}")

        return '\n'.join(info_lines)


def main():
    root = tk.Tk()
    app = ElliottWaveAnalyzerPro(root)
    root.mainloop()


if __name__ == "__main__":
    main()
