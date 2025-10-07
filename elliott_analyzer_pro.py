#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analizador Avanzado de Ondas de Elliott con Alertas de Trading y Reportes Inteligentes
Autor: Trading Analysis Tool Pro
Versión: 4.0 (Recomendaciones Ponderadas + R:R >= 3:1 + Reportes)
"""

import sys
import os
import subprocess
import importlib.util
from datetime import datetime
import threading
import time
import warnings

warnings.filterwarnings('ignore')

# Lista de paquetes requeridos
REQUIRED_PACKAGES = {
    'yfinance': 'yfinance',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'matplotlib': 'matplotlib',
    'scipy': 'scipy',
}


def check_and_install_packages():
    """Verificar e instalar paquetes faltantes"""
    missing_packages = []

    for package_name, pip_name in REQUIRED_PACKAGES.items():
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            missing_packages.append(pip_name)

    if missing_packages:
        print("=" * 60)
        print("INSTALACIÓN DE DEPENDENCIAS NECESARIAS")
        print("=" * 60)
        print(f"\nPaquetes faltantes detectados: {', '.join(missing_packages)}")

        try:
            response = input("\n¿Deseas instalar automáticamente los paquetes faltantes? (s/n): ")
        except Exception:
            response = 's'

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
            try:
                input("\nPresiona Enter para salir...")
            except Exception:
                pass
            sys.exit(0)
        else:
            print("\n⚠️ Para usar esta aplicación necesitas instalar:")
            print(f"pip install {' '.join(missing_packages)}")
            try:
                input("\nPresiona Enter para salir...")
            except Exception:
                pass
            sys.exit(1)

    return True


# Verificar dependencias antes de importar
if check_and_install_packages():
    print("✅ Todas las dependencias están instaladas")
    print("Iniciando aplicación...\n")

# Ahora importar los módulos
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class TradingSignal:
    """Clase para manejar señales de trading, con soporte para scoring y R:R.
    """

    def __init__(self, tipo, precio, fecha, razon, fuerza,
                 objetivo1, objetivo2, objetivo3, stop_loss,
                 source="Desconocido", atr=None):
        self.tipo = tipo  # 'COMPRA' o 'VENTA'
        self.precio = float(precio)
        self.fecha = fecha
        self.razon = razon
        self.fuerza = int(fuerza)  # 1-5 estrellas
        self.objetivo1 = float(objetivo1)
        self.objetivo2 = float(objetivo2)
        self.objetivo3 = float(objetivo3)
        self.stop_loss = float(stop_loss)
        self.source = source
        self.atr = float(atr) if atr is not None else None

        # Scoring ponderado por indicadores
        self.score_components = {
            'elliott': 0.0,
            'fibonacci': 0.0,
            'bollinger': 0.0,
            'rsi': 0.0,
            'macd': 0.0,
            'trend': 0.0,
        }
        self.weighted_score = 0.0
        self.confidence = 0.0  # 0..1

        # Calcular R:R con el mejor objetivo
        self.risk_reward = self._compute_best_risk_reward()

    def _compute_best_risk_reward(self) -> float:
        try:
            targets = [self.objetivo1, self.objetivo2, self.objetivo3]
            if self.tipo == 'COMPRA':
                risk = max(1e-8, self.precio - self.stop_loss)
                rr_values = [(t - self.precio) / risk for t in targets if t > self.precio]
            else:  # VENTA
                risk = max(1e-8, self.stop_loss - self.precio)
                rr_values = [(self.precio - t) / risk for t in targets if t < self.precio]
            return max(rr_values) if rr_values else 0.0
        except Exception:
            return 0.0


class ElliottWaveAnalyzerPro:
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 Elliott Wave Analyzer PRO - Sistema Automático de Trading v4.0")
        self.root.geometry("1500x950")

        # Variables
        self.df = None
        self.symbol = tk.StringVar(value="AAPL")
        self.period = tk.StringVar(value="6mo")
        self.auto_refresh = tk.BooleanVar(value=False)
        self.alert_enabled = tk.BooleanVar(value=True)
        self.wave_points = []
        self.fibonacci_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.618, 4.236]
        self.trading_signals = []
        self.current_position = None
        self.final_decision = None

        # Configuración ideal y de riesgos
        self.max_profit_config = {
            'entrada_onda2_fib': 0.618,
            'entrada_onda4_fib': 0.382,
            'salida_onda3_fib': 1.618,
            'salida_onda5_fib': 1.0,
            'stop_loss_percent': 3.0,  # usado como respaldo si ATR no disponible
            'position_size_percent': 2.0,
            'min_risk_reward': 3.0,
            'atr_stop_mult': 1.8,  # Stop basado en ATR
        }

        # Pesos del motor de recomendación (Elliott domina)
        self.weights_config = {
            'elliott': 0.45,
            'fibonacci': 0.15,
            'bollinger': 0.10,
            'rsi': 0.12,
            'macd': 0.10,
            'trend': 0.08,
        }

        # Listas predefinidas de activos
        self.setup_asset_lists()

        # UI
        self.setup_styles()
        self.setup_ui()

        # Monitoreo
        self.start_monitoring()

    def setup_styles(self):
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
        self.asset_categories = {
            "📈 ACCIONES POPULARES": [
                "AAPL - Apple", "MSFT - Microsoft", "GOOGL - Google", "AMZN - Amazon", "TSLA - Tesla",
                "META - Meta/Facebook", "NVDA - Nvidia", "JPM - JP Morgan", "V - Visa", "JNJ - Johnson & Johnson",
                "WMT - Walmart", "PG - Procter & Gamble", "DIS - Disney", "NFLX - Netflix", "BA - Boeing"
            ],
            "💰 CRIPTOMONEDAS": [
                "BTC-USD - Bitcoin", "ETH-USD - Ethereum", "BNB-USD - Binance Coin", "ADA-USD - Cardano",
                "XRP-USD - Ripple", "SOL-USD - Solana", "DOT-USD - Polkadot", "DOGE-USD - Dogecoin",
                "AVAX-USD - Avalanche", "MATIC-USD - Polygon", "LINK-USD - Chainlink", "UNI-USD - Uniswap",
                "ATOM-USD - Cosmos", "LTC-USD - Litecoin", "BCH-USD - Bitcoin Cash"
            ],
            "📊 ETFs PRINCIPALES": [
                "SPY - S&P 500", "QQQ - Nasdaq 100", "DIA - Dow Jones", "IWM - Russell 2000",
                "VTI - Total Market", "VOO - Vanguard S&P 500", "GLD - Gold", "SLV - Silver",
                "USO - Oil", "TLT - Treasury Bonds", "VNQ - Real Estate", "ARKK - ARK Innovation",
                "XLF - Financials", "XLK - Technology", "XLE - Energy"
            ],
            "🌎 MERCADOS INTERNACIONALES": [
                "EWJ - Japón", "EWG - Alemania", "EWU - Reino Unido", "EWC - Canadá", "EWA - Australia",
                "EWZ - Brasil", "INDA - India", "FXI - China", "EWT - Taiwan", "EWY - Corea del Sur"
            ]
        }

    def setup_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Tabs
        main_tab = ttk.Frame(notebook)
        notebook.add(main_tab, text="📊 Análisis Principal")

        signals_tab = ttk.Frame(notebook)
        notebook.add(signals_tab, text="🎯 Señales de Trading")

        config_tab = ttk.Frame(notebook)
        notebook.add(config_tab, text="⚙️ Configuración")

        reports_tab = ttk.Frame(notebook)
        notebook.add(reports_tab, text="📄 Reportes Inteligentes")

        # Build tabs
        self.setup_main_tab(main_tab)
        self.setup_signals_tab(signals_tab)
        self.setup_config_tab(config_tab)
        self.setup_reports_tab(reports_tab)

        # Inicializar lista de activos
        self.on_category_change()

    def setup_main_tab(self, parent):
        main_container = ttk.Frame(parent)
        main_container.pack(fill='both', expand=True)

        left_frame = ttk.Frame(main_container, width=260)
        left_frame.pack(side='left', fill='y', padx=5, pady=5)
        left_frame.pack_propagate(False)

        right_frame = ttk.Frame(main_container)
        right_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)

        # Selección activo
        asset_frame = ttk.LabelFrame(left_frame, text="📊 Selección de Activo", padding="10")
        asset_frame.pack(fill='x', pady=5)

        ttk.Label(asset_frame, text="Categoría:", font=('Arial', 9, 'bold')).pack(anchor='w', pady=2)
        self.category_var = tk.StringVar(value="📈 ACCIONES POPULARES")
        category_combo = ttk.Combobox(asset_frame, textvariable=self.category_var,
                                      values=list(self.asset_categories.keys()), width=25, state='readonly')
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
        ttk.Button(action_frame, text="📥 Cargar Datos", command=self.load_data).pack(fill='x', pady=2)
        ttk.Button(action_frame, text="🔄 Análisis Completo", command=self.run_complete_analysis).pack(fill='x', pady=2)
        ttk.Button(action_frame, text="🎯 Detectar Señales", command=self.detect_trading_signals).pack(fill='x', pady=2)
        ttk.Button(action_frame, text="📄 Generar Reporte", command=self.generate_intelligent_report).pack(fill='x', pady=2)
        ttk.Button(action_frame, text="🔍 Zoom Reset", command=self.reset_zoom).pack(fill='x', pady=2)

        options_frame = ttk.LabelFrame(left_frame, text="⚙️ Opciones", padding="10")
        options_frame.pack(fill='x', pady=5)
        ttk.Checkbutton(options_frame, text="🔄 Auto-Actualizar", variable=self.auto_refresh).pack(anchor='w', pady=2)
        ttk.Checkbutton(options_frame, text="🔔 Alertas Activas", variable=self.alert_enabled).pack(anchor='w', pady=2)

        # Right
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
        self.info_text = scrolledtext.ScrolledText(analysis_frame, height=6, wrap=tk.WORD, font=('Consolas', 9))
        self.info_text.pack(fill='both', expand=True, padx=5, pady=5)

        points_frame = ttk.Frame(info_notebook)
        info_notebook.add(points_frame, text="💰 Puntos de Trading")
        self.points_text = scrolledtext.ScrolledText(points_frame, height=6, wrap=tk.WORD, font=('Consolas', 9))
        self.points_text.pack(fill='both', expand=True, padx=5, pady=5)

    def setup_signals_tab(self, parent):
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
                self.signals_tree.column(col, width=90)
            elif col == 'Razón':
                self.signals_tree.column(col, width=260)
            else:
                self.signals_tree.column(col, width=110)
        scrollbar = ttk.Scrollbar(current_frame, orient='vertical', command=self.signals_tree.yview)
        scrollbar.pack(side='right', fill='y')
        self.signals_tree.configure(yscrollcommand=scrollbar.set)

        summary_frame = ttk.LabelFrame(parent, text="📊 Resumen de Trading", padding="10")
        summary_frame.pack(fill='x', padx=5, pady=5)
        self.summary_text = tk.Text(summary_frame, height=6, font=('Consolas', 10))
        self.summary_text.pack(fill='both', expand=True)

    def setup_config_tab(self, parent):
        profit_frame = ttk.LabelFrame(parent, text="💎 Configuración Ideal para Máxima Ganancia", padding="10")
        profit_frame.pack(fill='x', padx=10, pady=10)
        configs = [
            ("Entrada Onda 2 (Fibonacci):", 'entrada_onda2_fib', 0.618),
            ("Entrada Onda 4 (Fibonacci):", 'entrada_onda4_fib', 0.382),
            ("Salida Onda 3 (Fibonacci):", 'salida_onda3_fib', 1.618),
            ("Salida Onda 5 (Fibonacci):", 'salida_onda5_fib', 1.0),
            ("Stop Loss (% respaldo):", 'stop_loss_percent', 3.0),
            ("Tamaño Posición (% capital):", 'position_size_percent', 2.0),
            ("Risk/Reward Mínimo:", 'min_risk_reward', 3.0),
            ("ATR Stop Mult:", 'atr_stop_mult', 1.8),
        ]
        self.config_vars = {}
        for i, (label, key, default) in enumerate(configs):
            ttk.Label(profit_frame, text=label).grid(row=i, column=0, sticky='w', padx=5, pady=5)
            var = tk.DoubleVar(value=self.max_profit_config.get(key, default))
            self.config_vars[key] = var
            entry = ttk.Entry(profit_frame, textvariable=var, width=10)
            entry.grid(row=i, column=1, padx=5, pady=5)
        ttk.Button(profit_frame, text="💾 Guardar Configuración", command=self.save_config).grid(
            row=len(configs), column=0, columnspan=2, pady=10)

        rules_frame = ttk.LabelFrame(parent, text="📋 Checklist de Trading", padding="10")
        rules_frame.pack(fill='both', expand=True, padx=10, pady=10)
        rules_text = """
        ✅ CHECKLIST ANTES DE COMPRAR:

        □ Identificación clara de las 5 ondas de Elliott
        □ Retroceso en zona de Fibonacci (38.2%-61.8%)
        □ Confluencia RSI/MACD/Tendencia a favor
        □ Volumen confirmando el patrón
        □ Ratio Riesgo/Beneficio mínimo 1:3
        □ Stop loss basado en ATR
        □ Tamaño de posición calculado (máx 2% riesgo)

        🎯 PUNTOS IDEALES DE ENTRADA:
        • Final de Onda 2: Retroceso 50%-61.8% ⭐⭐⭐⭐⭐
        • Final de Onda 4: Retroceso 38.2%-50% ⭐⭐⭐⭐
        • Final de Onda C: Corrección completa ⭐⭐⭐⭐

        🔴 PUNTOS DE SALIDA:
        • Final de Onda 3: Venta parcial (50%)
        • Final de Onda 5: Venta total
        • Stop Loss tocado: Salida inmediata
        """
        rules_label = tk.Text(rules_frame, wrap=tk.WORD, font=('Consolas', 10))
        rules_label.insert('1.0', rules_text)
        rules_label.config(state='disabled')
        rules_label.pack(fill='both', expand=True)

    def setup_reports_tab(self, parent):
        container = ttk.Frame(parent)
        container.pack(fill='both', expand=True, padx=10, pady=10)

        buttons = ttk.Frame(container)
        buttons.pack(fill='x')
        ttk.Button(buttons, text="📄 Generar Reporte", command=self.generate_intelligent_report).pack(side='left', padx=5)
        ttk.Button(buttons, text="💾 Exportar (Markdown)", command=self.export_report_markdown).pack(side='left', padx=5)

        self.report_text = scrolledtext.ScrolledText(container, height=28, wrap=tk.WORD, font=('Consolas', 10))
        self.report_text.pack(fill='both', expand=True, pady=8)

    def on_category_change(self, event=None):
        category = self.category_var.get()
        assets = self.asset_categories.get(category, [])
        self.asset_combo['values'] = assets
        if assets:
            self.asset_combo.current(0)
            self.on_asset_change()

    def on_asset_change(self, event=None):
        selected = self.asset_combo.get()
        if selected:
            symbol = selected.split(' - ')[0]
            self.symbol.set(symbol)

    def save_config(self):
        for key, var in self.config_vars.items():
            self.max_profit_config[key] = float(var.get())
        messagebox.showinfo("Configuración", "✅ Configuración guardada exitosamente")

    def start_monitoring(self):
        def monitor():
            while True:
                if self.auto_refresh.get() and self.df is not None:
                    self.root.after(0, self.refresh_data)
                time.sleep(60)
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

    def refresh_data(self):
        try:
            self.load_data(silent=True)
            self.run_complete_analysis()
        except Exception:
            pass

    def load_data(self, silent=False):
        try:
            if not silent:
                self.info_text.delete(1.0, tk.END)
                self.info_text.insert(tk.END, f"⏳ Cargando datos para {self.symbol.get()}...\n")
                self.root.update()

            ticker = yf.Ticker(self.symbol.get())
            self.df = ticker.history(period=self.period.get())

            if self.df is None or self.df.empty:
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
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.df['ATR'] = tr.rolling(window=14).mean()
        low_14 = self.df['Low'].rolling(window=14).min()
        high_14 = self.df['High'].rolling(window=14).max()
        self.df['%K'] = 100 * ((self.df['Close'] - low_14) / (high_14 - low_14))
        self.df['%D'] = self.df['%K'].rolling(window=3).mean()

    def display_basic_info(self):
        current_price = float(self.df['Close'].iloc[-1])
        change = float(self.df['Close'].iloc[-1] - self.df['Close'].iloc[-2])
        change_pct = (change / float(self.df['Close'].iloc[-2])) * 100
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
        if self.df is not None and hasattr(self, 'canvas'):
            self.update_complete_chart()

    def reset_zoom(self):
        if hasattr(self, 'ax'):
            self.ax.autoscale()
            if hasattr(self, 'ax2'):
                self.ax2.autoscale()
            if hasattr(self, 'ax3'):
                self.ax3.autoscale()
            if hasattr(self, 'canvas'):
                self.canvas.draw()

    def plot_basic_chart(self):
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
        for idx in range(len(self.df)):
            date = self.df.index[idx]
            open_price = self.df['Open'].iloc[idx]
            close_price = self.df['Close'].iloc[idx]
            high_price = self.df['High'].iloc[idx]
            low_price = self.df['Low'].iloc[idx]
            color = '#00b050' if close_price >= open_price else '#ff0000'
            ax.plot([date, date], [low_price, high_price], color='black', linewidth=0.5, alpha=0.6)
            height = abs(close_price - open_price)
            bottom = min(close_price, open_price)
            if height > 0:
                ax.bar(date, height, bottom=bottom, width=0.8, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        if hasattr(self, 'show_sma') and self.show_sma.get():
            if 'SMA20' in self.df.columns:
                ax.plot(self.df.index, self.df['SMA20'], label='SMA 20', color='orange', alpha=0.9, linewidth=1.5)
            if 'SMA50' in self.df.columns:
                ax.plot(self.df.index, self.df['SMA50'], label='SMA 50', color='blue', alpha=0.9, linewidth=1.5)
        ax.set_ylabel('Precio ($)', fontsize=10)
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    def plot_volume(self, ax):
        colors = ['#00b050' if self.df['Close'].iloc[i] >= self.df['Open'].iloc[i] else '#ff0000' for i in range(len(self.df))]
        ax.bar(self.df.index, self.df['Volume'], color=colors, alpha=0.6)
        ax.set_ylabel('Volumen', fontsize=9)
        ax.grid(True, alpha=0.2)
        avg_volume = self.df['Volume'].rolling(window=20).mean()
        ax.plot(self.df.index, avg_volume, color='purple', alpha=0.5, linewidth=1, label='Vol. Promedio')
        ax.legend(loc='upper left', fontsize=7)

    def plot_rsi(self, ax):
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
        ax.plot(self.df.index, self.df['MACD'], 'b-', label='MACD', linewidth=1.2)
        ax.plot(self.df.index, self.df['Signal'], 'r-', label='Signal', linewidth=1.2)
        colors = ['#00b050' if val >= 0 else '#ff0000' for val in self.df['MACD_Histogram']]
        ax.bar(self.df.index, self.df['MACD_Histogram'], color=colors, alpha=0.3, label='Histograma')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=0.5)
        ax.set_xlabel('Fecha', fontsize=9)
        ax.set_ylabel('MACD', fontsize=9)
        ax.legend(loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.2)

    # ------------------------
    # Lógica de análisis
    # ------------------------
    def run_complete_analysis(self):
        if self.df is None:
            messagebox.showwarning("Advertencia", "Primero cargue los datos")
            return
        try:
            self.info_text.delete(1.0, tk.END)
            self.points_text.delete(1.0, tk.END)
            self.trading_signals.clear()
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
                extrema.append((int(idx), float(prices[idx]), 'max'))
            for idx in local_minima:
                extrema.append((int(idx), float(prices[idx]), 'min'))
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
        if len(extrema) < 5:
            return None
        best_pattern = None
        best_score = -1
        for start in range(len(extrema) - 4):
            pattern = extrema[start:start + 5]
            types = [p[2] for p in pattern]
            if types == ['min', 'max', 'min', 'max', 'min'] or types == ['max', 'min', 'max', 'min', 'max']:
                score = self.score_elliott_pattern(pattern)
                if score > best_score:
                    best_score = score
                    best_pattern = pattern
        # Extend with ABC if available
        if best_pattern:
            idx_map = {tuple(p): i for i, p in enumerate(extrema)}
            last_idx = idx_map.get(tuple(best_pattern[-1]))
            if last_idx is not None and last_idx + 3 <= len(extrema) - 1:
                best_pattern = best_pattern + extrema[last_idx + 1:last_idx + 4]
        return best_pattern

    def score_elliott_pattern(self, pattern):
        score = 0
        prices = [p[1] for p in pattern]
        wave1_length = abs(prices[1] - prices[0])
        wave3_length = abs(prices[2] - prices[1])
        wave5_length = abs(prices[4] - prices[3]) if len(prices) > 4 else 0
        if wave3_length >= min(wave1_length, wave5_length):
            score += 40
        wave2_retrace = abs(prices[2] - prices[1]) / max(1e-8, abs(prices[1] - prices[0]))
        if 0.382 <= wave2_retrace <= 0.618:
            score += 30
        # onda 4 no invade 1 (aprox)
        if len(prices) > 3 and prices[3] > min(prices[0], prices[1]):
            score += 30
        return score

    def display_wave_analysis(self, wave_pattern):
        self.info_text.insert(tk.END, "\n" + "=" * 50 + "\n")
        self.info_text.insert(tk.END, "     ONDAS DE ELLIOTT DETECTADAS\n")
        self.info_text.insert(tk.END, "=" * 50 + "\n")
        wave_labels = ['1', '2', '3', '4', '5', 'A', 'B', 'C']
        for i, (idx, price, type_) in enumerate(wave_pattern[:8]):
            if i < len(wave_labels):
                date = self.df.index[idx]
                emoji = "📈" if type_ == 'max' else "📉"
                self.info_text.insert(tk.END, f"{emoji} Onda {wave_labels[i]}: ${price:.2f} - {date.strftime('%Y-%m-%d')}\n")

    def calculate_fibonacci_levels(self):
        if self.df is None:
            return
        high = float(self.df['High'].max())
        low = float(self.df['Low'].min())
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
        if 'BB_Upper' not in self.df.columns:
            return
        current_price = float(self.df['Close'].iloc[-1])
        bb_upper = float(self.df['BB_Upper'].iloc[-1])
        bb_lower = float(self.df['BB_Lower'].iloc[-1])
        b_percent = (current_price - bb_lower) / max(1e-8, (bb_upper - bb_lower))
        self.info_text.insert(tk.END, "\n" + "=" * 50 + "\n")
        self.info_text.insert(tk.END, "     BANDAS DE BOLLINGER\n")
        self.info_text.insert(tk.END, "=" * 50 + "\n")
        self.info_text.insert(tk.END, f"Banda Superior: ${bb_upper:.2f}\n")
        self.info_text.insert(tk.END, f"Banda Inferior: ${bb_lower:.2f}\n")
        self.info_text.insert(tk.END, f"Indicador %B: {b_percent:.2%}\n")

    def identify_buy_sell_points(self, wave_pattern):
        if not wave_pattern:
            return
        self.points_text.insert(tk.END, "=" * 60 + "\n")
        self.points_text.insert(tk.END, "   PUNTOS ÓPTIMOS DE COMPRA Y VENTA\n")
        self.points_text.insert(tk.END, "=" * 60 + "\n\n")
        current_price = float(self.df['Close'].iloc[-1])
        prices = [p[1] for p in wave_pattern]
        current_wave = self.identify_current_wave(wave_pattern, current_price)
        self.points_text.insert(tk.END, "🟢 PUNTOS DE COMPRA ÓPTIMOS:\n" + "-" * 40 + "\n")
        if len(prices) > 2:
            wave1_length = abs(prices[1] - prices[0])
            buy_zone_start = prices[0] + (wave1_length * 0.5)
            buy_zone_end = prices[0] + (wave1_length * 0.618)
            sl = prices[0] * (1 - self.max_profit_config['stop_loss_percent'] / 100.0)
            lines_onda2 = [
                "📍 FINAL DE ONDA 2 ⭐⭐⭐⭐⭐ (MEJOR ENTRADA)",
                f"   • Zona de compra: ${buy_zone_start:.2f} - ${buy_zone_end:.2f}",
                f"   • Stop Loss (respaldo): ${sl:.2f}",
                f"   • Objetivo 1: ${prices[1] + (wave1_length * 0.618):.2f}",
                f"   • Objetivo 2: ${prices[1] + (wave1_length * 1.618):.2f}",
                f"   • Objetivo 3: ${prices[1] + (wave1_length * 2.618):.2f}",
                "",
            ]
            self.points_text.insert(tk.END, "\n".join(lines_onda2))
        if len(prices) > 4:
            wave3_length = abs(prices[2] - prices[1])
            wave4_retrace_382 = prices[2] - (wave3_length * 0.382)
            wave4_retrace_50 = prices[2] - (wave3_length * 0.5)
            sl4 = prices[3] * (1 - self.max_profit_config['stop_loss_percent'] / 100.0)
            lines_onda4 = [
                "📍 FINAL DE ONDA 4 ⭐⭐⭐⭐ (MUY BUENA ENTRADA)",
                f"   • Zona de compra: ${wave4_retrace_50:.2f} - ${wave4_retrace_382:.2f}",
                f"   • Stop Loss (respaldo): ${sl4:.2f}",
                f"   • Objetivo (Onda 5): ${prices[2] + (prices[1] - prices[0]):.2f}",
                "",
            ]
            self.points_text.insert(tk.END, "\n".join(lines_onda4))
        self.points_text.insert(tk.END, "\n🔴 PUNTOS DE VENTA ÓPTIMOS:\n" + "-" * 40 + "\n")
        if len(prices) > 2:
            wave1_length = abs(prices[1] - prices[0])
            target_wave3_1 = prices[0] + (wave1_length * 1.618)
            target_wave3_2 = prices[0] + (wave1_length * 2.618)
            lines_onda3 = [
                "📍 FINAL DE ONDA 3 (VENTA PARCIAL 50%)",
                f"   • Objetivo 1: ${target_wave3_1:.2f} (1.618 × Onda 1)",
                f"   • Objetivo 2: ${target_wave3_2:.2f} (2.618 × Onda 1)",
                "",
            ]
            self.points_text.insert(tk.END, "\n".join(lines_onda3))
        if len(prices) > 4:
            target_wave5 = prices[2] + (prices[1] - prices[0])
            lines_onda5 = [
                "📍 FINAL DE ONDA 5 (VENTA TOTAL)",
                f"   • Objetivo: ${target_wave5:.2f} (Onda 5 = Onda 1)",
                "",
            ]
            self.points_text.insert(tk.END, "\n".join(lines_onda5))
        self.points_text.insert(tk.END, "\n" + "=" * 60 + "\n")
        self.points_text.insert(tk.END, "   SITUACIÓN ACTUAL Y RECOMENDACIÓN\n")
        self.points_text.insert(tk.END, "=" * 60 + "\n\n")
        self.points_text.insert(tk.END, f"💹 Precio actual: ${current_price:.2f}\n")
        self.points_text.insert(tk.END, f"📍 Onda actual estimada: {current_wave}\n\n")
        recommendation = self.get_trading_recommendation(current_wave, current_price, wave_pattern)
        self.points_text.insert(tk.END, recommendation)

    def identify_current_wave(self, wave_pattern, current_price):
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
        rsi = float(self.df['RSI'].iloc[-1]) if 'RSI' in self.df.columns else 50
        recommendation = "🎯 RECOMENDACIÓN:\n" + "-" * 40 + "\n"
        if "2" in current_wave and "3" in current_wave:
            recommendation += """
✅ SEÑAL DE COMPRA FUERTE
• Zona óptima de entrada (final Onda 2)
• Esperar confirmación de reversión alcista
• Stop loss basado en ATR
• Objetivos: Niveles de Fibonacci 1.618 y 2.618
• Risk/Reward objetivo: ≥ 1:5
            """
        elif "4" in current_wave and "5" in current_wave:
            recommendation += """
✅ SEÑAL DE COMPRA MODERADA
• Confirmar que Onda 4 no invade Onda 1
• Stop loss basado en ATR
• Objetivo: Onda 5 = Onda 1
• Risk/Reward objetivo: ≥ 1:3
            """
        elif "3" in current_wave:
            if rsi > 70:
                recommendation += """
⚠️ SEÑAL DE VENTA PARCIAL
• Onda 3 en zona de sobrecompra
• Considerar tomar ganancias parciales
• Mover stop loss a breakeven
            """
            else:
                recommendation += """
➡️ MANTENER POSICIÓN
• Onda 3 en desarrollo; dejar correr ganancias
• Ajustar stop loss progresivamente
            """
        elif "5" in current_wave:
            recommendation += """
🔴 SEÑAL DE VENTA FUERTE
• Final de ciclo alcista
• Cerrar posiciones largas
• Preparar posible corrección ABC
            """
        elif "C" in current_wave:
            recommendation += """
✅ PREPARAR NUEVA ENTRADA
• Corrección ABC cerca de completarse
• Buscar divergencia alcista en RSI
            """
        else:
            recommendation += """
⏳ ESPERAR Y OBSERVAR
• Patrón en desarrollo; sin señal clara
            """
        recommendation += f"\n\n📊 RSI Actual: {rsi:.1f}"
        if rsi > 70:
            recommendation += " (SOBRECOMPRA ⚠️)"
        elif rsi < 30:
            recommendation += " (SOBREVENTA ✅)"
        else:
            recommendation += " (NEUTRAL)"
        return recommendation

    # ------------------------
    # Señales y motor de recomendación
    # ------------------------
    def detect_trading_signals(self):
        if self.df is None:
            return
        current_price = float(self.df['Close'].iloc[-1])
        # Limpiar UI
        self.trading_signals.clear()
        for item in self.signals_tree.get_children():
            self.signals_tree.delete(item)
        signals = []
        s1 = self.check_elliott_signal(current_price)
        if s1:
            signals.append(s1)
        s2 = self.check_fibonacci_signal(current_price)
        if s2:
            signals.append(s2)
        s3 = self.check_bollinger_signal(current_price)
        if s3:
            signals.append(s3)
        s4 = self.check_rsi_signal()
        if s4:
            signals.append(s4)
        s5 = self.check_macd_signal()
        if s5:
            signals.append(s5)
        self.evaluate_signals(signals)
        self.show_signals_summary()
        self.check_alerts()

    # Utilidades para scoring
    def _trend_support(self, tipo):
        price = float(self.df['Close'].iloc[-1])
        sma20 = float(self.df['SMA20'].iloc[-1]) if not np.isnan(self.df['SMA20'].iloc[-1]) else price
        sma50 = float(self.df['SMA50'].iloc[-1]) if not np.isnan(self.df['SMA50'].iloc[-1]) else price
        if tipo == 'COMPRA':
            return 1.0 if price >= sma20 and price >= sma50 else 0.0
        return 1.0 if price <= sma20 and price <= sma50 else 0.0

    def _fibonacci_support(self):
        high = float(self.df['High'].max())
        low = float(self.df['Low'].min())
        diff = max(1e-8, high - low)
        current_price = float(self.df['Close'].iloc[-1])
        key_levels = [0.382, 0.5, 0.618]
        min_dist = 1.0
        for level in key_levels:
            fib_price = low + diff * level
            dist = abs(current_price - fib_price) / fib_price
            min_dist = min(min_dist, dist)
        if min_dist <= 0.01:
            return 1.0
        if min_dist <= 0.02:
            return 0.6
        return 0.0

    def _bollinger_support(self, tipo):
        if 'BB_Upper' not in self.df.columns:
            return 0.0
        price = float(self.df['Close'].iloc[-1])
        upper = float(self.df['BB_Upper'].iloc[-1])
        lower = float(self.df['BB_Lower'].iloc[-1])
        if tipo == 'COMPRA':
            if price <= lower:
                return 1.0
            if price <= lower * 1.01:
                return 0.6
            return 0.0
        else:
            if price >= upper:
                return 1.0
            if price >= upper * 0.99:
                return 0.6
            return 0.0

    def _rsi_support(self, tipo):
        rsi = float(self.df['RSI'].iloc[-1]) if 'RSI' in self.df.columns else 50
        if tipo == 'COMPRA':
            if rsi <= 30:
                return 1.0
            if rsi <= 40:
                return 0.6
            return 0.0
        else:
            if rsi >= 70:
                return 1.0
            if rsi >= 60:
                return 0.6
            return 0.0

    def _macd_support(self, tipo):
        macd = float(self.df['MACD'].iloc[-1])
        signal = float(self.df['Signal'].iloc[-1])
        macd_prev = float(self.df['MACD'].iloc[-2])
        signal_prev = float(self.df['Signal'].iloc[-2])
        if tipo == 'COMPRA':
            return 1.0 if (macd > signal and macd_prev <= signal_prev) or macd > 0 else 0.0
        else:
            return 1.0 if (macd < signal and macd_prev >= signal_prev) or macd < 0 else 0.0

    def _elliott_support(self):
        return 1.0 if self.wave_points else 0.0

    def _compute_weighted_score(self, signal: TradingSignal) -> None:
        comps = signal.score_components
        comps['elliott'] = self._elliott_support()
        comps['fibonacci'] = self._fibonacci_support()
        comps['bollinger'] = self._bollinger_support(signal.tipo)
        comps['rsi'] = self._rsi_support(signal.tipo)
        comps['macd'] = self._macd_support(signal.tipo)
        comps['trend'] = self._trend_support(signal.tipo)
        total_weight = sum(self.weights_config.values())
        weighted_sum = 0.0
        for k, v in comps.items():
            weighted_sum += v * self.weights_config.get(k, 0.0)
        signal.weighted_score = weighted_sum / max(1e-8, total_weight)
        signal.confidence = signal.weighted_score

    def _atr_stop(self, entry_price: float, tipo: str) -> float:
        atr = float(self.df['ATR'].iloc[-1]) if 'ATR' in self.df.columns else None
        mult = float(self.max_profit_config.get('atr_stop_mult', 1.8))
        if atr is None or np.isnan(atr) or atr <= 0:
            # fallback porcentaje
            pct = float(self.max_profit_config.get('stop_loss_percent', 3.0)) / 100.0
            return entry_price * (1 - pct) if tipo == 'COMPRA' else entry_price * (1 + pct)
        return entry_price - mult * atr if tipo == 'COMPRA' else entry_price + mult * atr

    # Generación de señales individuales
    def check_elliott_signal(self, current_price):
        if not self.wave_points:
            return None
        prices = [p[1] for p in self.wave_points]
        current_wave = self.identify_current_wave(self.wave_points, current_price)
        atr = float(self.df['ATR'].iloc[-1]) if 'ATR' in self.df.columns else None
        # Onda 2 -> compra fuerte
        if "2" in current_wave and len(prices) > 2:
            wave1_length = abs(prices[1] - prices[0])
            target = prices[0] + (wave1_length * 0.618)
            if abs(current_price - target) / max(1e-8, target) < 0.03:
                stop = self._atr_stop(current_price, 'COMPRA')
                sig = TradingSignal(
                    tipo="COMPRA",
                    precio=current_price,
                    fecha=self.df.index[-1],
                    razon="Elliott: Final de Onda 2 (61.8%)",
                    fuerza=5,
                    objetivo1=prices[1] + (wave1_length * 0.618),
                    objetivo2=prices[1] + (wave1_length * 1.618),
                    objetivo3=prices[1] + (wave1_length * 2.618),
                    stop_loss=stop,
                    source="Elliott",
                    atr=atr,
                )
                if sig.risk_reward >= float(self.max_profit_config['min_risk_reward']):
                    self._compute_weighted_score(sig)
                    return sig
                return None
        # Onda 5 -> venta
        if "5" in current_wave and len(prices) > 4:
            stop = self._atr_stop(current_price, 'VENTA')
            sig = TradingSignal(
                tipo="VENTA",
                precio=current_price,
                fecha=self.df.index[-1],
                razon="Elliott: Final de Onda 5 (fin impulso)",
                fuerza=4,
                objetivo1=current_price * 0.97,
                objetivo2=current_price * 0.94,
                objetivo3=current_price * 0.90,
                stop_loss=stop,
                source="Elliott",
                atr=atr,
            )
            if sig.risk_reward >= float(self.max_profit_config['min_risk_reward']):
                self._compute_weighted_score(sig)
                return sig
        return None

    def check_fibonacci_signal(self, current_price):
        high = float(self.df['High'].max())
        low = float(self.df['Low'].min())
        diff = max(1e-8, high - low)
        for level in [0.382, 0.5, 0.618]:
            fib_price = low + (diff * level)
            if abs(current_price - fib_price) / fib_price < 0.01:  # 1%
                stop = self._atr_stop(current_price, 'COMPRA') if current_price >= fib_price else self._atr_stop(current_price, 'VENTA')
                tipo = 'COMPRA' if current_price >= fib_price else 'VENTA'
                # Objetivos aproximados hacia banda media/superior o inferior
                if tipo == 'COMPRA':
                    objetivo1 = low + diff * (min(1.0, level + 0.236))
                    objetivo2 = low + diff * (min(1.0, level + 0.382))
                    objetivo3 = low + diff * (min(1.0, level + 0.618))
                else:
                    objetivo1 = low + diff * (max(0.0, level - 0.236))
                    objetivo2 = low + diff * (max(0.0, level - 0.382))
                    objetivo3 = low + diff * (max(0.0, level - 0.618))
                sig = TradingSignal(
                    tipo=tipo,
                    precio=current_price,
                    fecha=self.df.index[-1],
                    razon=f"Fibonacci {level:.1%} con reacción",
                    fuerza=3,
                    objetivo1=objetivo1,
                    objetivo2=objetivo2,
                    objetivo3=objetivo3,
                    stop_loss=stop,
                    source="Fibonacci",
                    atr=float(self.df['ATR'].iloc[-1]) if 'ATR' in self.df.columns else None,
                )
                if sig.risk_reward >= float(self.max_profit_config['min_risk_reward']):
                    self._compute_weighted_score(sig)
                    return sig
        return None

    def check_bollinger_signal(self, current_price):
        if 'BB_Upper' not in self.df.columns:
            return None
        upper = float(self.df['BB_Upper'].iloc[-1])
        lower = float(self.df['BB_Lower'].iloc[-1])
        middle = float(self.df['BB_Middle'].iloc[-1])
        atr = float(self.df['ATR'].iloc[-1]) if 'ATR' in self.df.columns else None
        if current_price <= lower:
            stop = self._atr_stop(current_price, 'COMPRA')
            sig = TradingSignal(
                tipo="COMPRA",
                precio=current_price,
                fecha=self.df.index[-1],
                razon="Bollinger: toque banda inferior",
                fuerza=3,
                objetivo1=middle,
                objetivo2=(middle + upper) / 2,
                objetivo3=upper,
                stop_loss=stop,
                source="Bollinger",
                atr=atr,
            )
            if sig.risk_reward >= float(self.max_profit_config['min_risk_reward']):
                self._compute_weighted_score(sig)
                return sig
        elif current_price >= upper:
            stop = self._atr_stop(current_price, 'VENTA')
            sig = TradingSignal(
                tipo="VENTA",
                precio=current_price,
                fecha=self.df.index[-1],
                razon="Bollinger: toque banda superior",
                fuerza=3,
                objetivo1=middle,
                objetivo2=(middle + lower) / 2,
                objetivo3=lower,
                stop_loss=stop,
                source="Bollinger",
                atr=atr,
            )
            if sig.risk_reward >= float(self.max_profit_config['min_risk_reward']):
                self._compute_weighted_score(sig)
                return sig
        return None

    def check_rsi_signal(self):
        if 'RSI' not in self.df.columns:
            return None
        rsi = float(self.df['RSI'].iloc[-1])
        rsi_prev = float(self.df['RSI'].iloc[-2])
        atr = float(self.df['ATR'].iloc[-1]) if 'ATR' in self.df.columns else None
        if rsi < 30 and rsi > rsi_prev:
            price = float(self.df['Close'].iloc[-1])
            stop = self._atr_stop(price, 'COMPRA')
            sig = TradingSignal(
                tipo="COMPRA",
                precio=price,
                fecha=self.df.index[-1],
                razon="RSI sobreventa con reversión",
                fuerza=4,
                objetivo1=price * 1.03,
                objetivo2=price * 1.05,
                objetivo3=price * 1.08,
                stop_loss=stop,
                source="RSI",
                atr=atr,
            )
            if sig.risk_reward >= float(self.max_profit_config['min_risk_reward']):
                self._compute_weighted_score(sig)
                return sig
        elif rsi > 70 and rsi < rsi_prev:
            price = float(self.df['Close'].iloc[-1])
            stop = self._atr_stop(price, 'VENTA')
            sig = TradingSignal(
                tipo="VENTA",
                precio=price,
                fecha=self.df.index[-1],
                razon="RSI sobrecompra con reversión",
                fuerza=3,
                objetivo1=price * 0.97,
                objetivo2=price * 0.95,
                objetivo3=price * 0.92,
                stop_loss=stop,
                source="RSI",
                atr=atr,
            )
            if sig.risk_reward >= float(self.max_profit_config['min_risk_reward']):
                self._compute_weighted_score(sig)
                return sig
        return None

    def check_macd_signal(self):
        if 'MACD' not in self.df.columns:
            return None
        macd = float(self.df['MACD'].iloc[-1])
        signal = float(self.df['Signal'].iloc[-1])
        macd_prev = float(self.df['MACD'].iloc[-2])
        signal_prev = float(self.df['Signal'].iloc[-2])
        atr = float(self.df['ATR'].iloc[-1]) if 'ATR' in self.df.columns else None
        price = float(self.df['Close'].iloc[-1])
        if macd > signal and macd_prev <= signal_prev:
            stop = self._atr_stop(price, 'COMPRA')
            sig = TradingSignal(
                tipo="COMPRA",
                precio=price,
                fecha=self.df.index[-1],
                razon="MACD cruce alcista",
                fuerza=3,
                objetivo1=price * 1.02,
                objetivo2=price * 1.04,
                objetivo3=price * 1.06,
                stop_loss=stop,
                source="MACD",
                atr=atr,
            )
            if sig.risk_reward >= float(self.max_profit_config['min_risk_reward']):
                self._compute_weighted_score(sig)
                return sig
        elif macd < signal and macd_prev >= signal_prev:
            stop = self._atr_stop(price, 'VENTA')
            sig = TradingSignal(
                tipo="VENTA",
                precio=price,
                fecha=self.df.index[-1],
                razon="MACD cruce bajista",
                fuerza=3,
                objetivo1=price * 0.98,
                objetivo2=price * 0.96,
                objetivo3=price * 0.94,
                stop_loss=stop,
                source="MACD",
                atr=atr,
            )
            if sig.risk_reward >= float(self.max_profit_config['min_risk_reward']):
                self._compute_weighted_score(sig)
                return sig
        return None

    def evaluate_signals(self, signals):
        if not signals:
            self.final_decision = {
                'action': 'ESPERAR', 'strength': '—', 'score': 0.0, 'signal': None
            }
            return
        # Filtrar por R:R
        min_rr = float(self.max_profit_config['min_risk_reward'])
        signals = [s for s in signals if s and s.risk_reward >= min_rr]
        if not signals:
            self.final_decision = {
                'action': 'ESPERAR', 'strength': '—', 'score': 0.0, 'signal': None
            }
            self.alert_label.config(text=f"Sin señales que cumplan R:R ≥ {min_rr}:1")
            return
        # Ordenar por score ponderado (computado al crear cada señal)
        for s in signals:
            # Asegurar score calculado
            if s.weighted_score == 0.0:
                self._compute_weighted_score(s)
        signals.sort(key=lambda x: (x.weighted_score, x.fuerza, x.risk_reward), reverse=True)
        # Mostrar señales en tabla
        idx = 1
        for s in signals:
            tipo_desc = f"{s.tipo} {'FUERTE' if s.weighted_score >= 0.65 else ('MODERADA' if s.weighted_score >= 0.5 else 'DÉBIL')}"
            self.add_signal_to_tree(s, tipo_desc + f" [Score {s.weighted_score:.2f}]")
            self.trading_signals.append(s)
            idx += 1
        best = signals[0]
        strength = 'FUERTE' if best.weighted_score >= 0.65 and best.score_components.get('elliott', 0.0) >= 1.0 else (
            'MODERADA' if best.weighted_score >= 0.5 else 'DÉBIL'
        )
        action = best.tipo if strength != 'DÉBIL' else 'ESPERAR'
        self.final_decision = {
            'action': action,
            'strength': strength,
            'score': float(best.weighted_score),
            'signal': best,
        }

    def add_signal_to_tree(self, signal, tipo_desc):
        tag = 'buy' if signal.tipo == "COMPRA" else 'sell'
        self.signals_tree.insert('', 'end',
                                 values=(
                                     signal.fecha.strftime('%Y-%m-%d %H:%M'),
                                     tipo_desc,
                                     f"${signal.precio:.2f}",
                                     f"${signal.stop_loss:.2f}",
                                     f"${signal.objetivo1:.2f}",
                                     f"${signal.objetivo2:.2f}",
                                     f"${signal.objetivo3:.2f}",
                                     f"{signal.risk_reward:.1f}:1",
                                     "⭐" * signal.fuerza,
                                     f"{signal.razon} (src: {signal.source})",
                                 ),
                                 tags=(tag,)
                                 )
        self.signals_tree.tag_configure('buy', foreground='green')
        self.signals_tree.tag_configure('sell', foreground='red')

    def show_signals_summary(self):
        self.summary_text.delete(1.0, tk.END)
        if not self.trading_signals:
            self.summary_text.insert(tk.END, "No hay señales activas actualmente\n")
            return
        buy_count = sum(1 for s in self.trading_signals if s.tipo == "COMPRA")
        sell_count = sum(1 for s in self.trading_signals if s.tipo == "VENTA")
        best_rr = max(self.trading_signals, key=lambda s: s.risk_reward).risk_reward
        best_score = max(self.trading_signals, key=lambda s: s.weighted_score).weighted_score
        decision_text = ""
        if self.final_decision and self.final_decision['signal']:
            fd = self.final_decision
            decision_text = f"\nDECISIÓN: {fd['action']} {fd['strength']} (Score {fd['score']:.2f})\n"
        summary = f"""
📊 RESUMEN DE SEÑALES
═══════════════════════════════════
🟢 Señales de COMPRA: {buy_count}
🔴 Señales de VENTA: {sell_count}
💹 Mejor R:R: {best_rr:.1f}:1
📈 Score más alto: {best_score:.2f}
{decision_text}
        """
        self.summary_text.insert(tk.END, summary)

    def check_alerts(self):
        if not self.alert_enabled.get():
            return
        if not self.final_decision or not self.final_decision.get('signal'):
            self.alert_label.config(text="Señales débiles o sin consenso. Esperar.")
            return
        sig = self.final_decision['signal']
        if self.final_decision['action'] in ("COMPRA", "VENTA") and self.final_decision['strength'] in ("FUERTE", "MODERADA"):
            alert_text = f"""
🚨 ALERTA DE TRADING {self.final_decision['strength']} 🚨
Tipo: {sig.tipo}
Precio: ${sig.precio:.2f}
Razón: {sig.razon}
Score: {sig.weighted_score:.2f}
R:R: {sig.risk_reward:.1f}:1  |  Stop: ${sig.stop_loss:.2f}
Obj1: ${sig.objetivo1:.2f}  Obj2: ${sig.objetivo2:.2f}  Obj3: ${sig.objetivo3:.2f}
            """
            self.alert_label.config(text=alert_text, style='Alert.TLabel')
        else:
            self.alert_label.config(text="Señales débiles - Mantener vigilancia")

    def update_complete_chart(self):
        if not hasattr(self, 'ax') or self.df is None:
            return
        self.plot_basic_chart()
        # Dibujar señales
        if hasattr(self, 'show_signals') and self.show_signals.get() and self.trading_signals:
            for signal in self.trading_signals:
                if signal.tipo == "COMPRA":
                    self.ax.scatter(signal.fecha, signal.precio, marker='^', s=250, color='lime',
                                    edgecolor='darkgreen', linewidth=2, zorder=9)
                    self.ax.annotate('BUY', (signal.fecha, signal.precio), xytext=(0, -20), textcoords='offset points',
                                     fontsize=10, fontweight='bold', color='darkgreen',
                                     bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', edgecolor='darkgreen',
                                               alpha=0.9), zorder=10,
                                     arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1))
                    self.ax.axhline(y=signal.objetivo1, color='green', linestyle=':', alpha=0.3, linewidth=0.8)
                    self.ax.axhline(y=signal.stop_loss, color='red', linestyle=':', alpha=0.3, linewidth=0.8)
                else:
                    self.ax.scatter(signal.fecha, signal.precio, marker='v', s=250, color='red',
                                    edgecolor='darkred', linewidth=2, zorder=9)
                    self.ax.annotate('SELL', (signal.fecha, signal.precio), xytext=(0, 20), textcoords='offset points',
                                     fontsize=10, fontweight='bold', color='darkred',
                                     bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', edgecolor='darkred',
                                               alpha=0.9), zorder=10,
                                     arrowprops=dict(arrowstyle='->', color='darkred', lw=1))
        # Info panel
        info_text = self.build_chart_info()
        if info_text:
            self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, fontsize=8, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9), zorder=12)
        if hasattr(self, 'show_sma') and self.show_sma.get():
            self.ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        self.canvas.draw()

    def build_chart_info(self):
        if self.df is None:
            return ""
        current_price = float(self.df['Close'].iloc[-1])
        rsi = float(self.df['RSI'].iloc[-1]) if 'RSI' in self.df.columns else 0
        info_lines = []
        if rsi > 70:
            info_lines.append(f"⚠️ RSI: {rsi:.1f} (SOBRECOMPRA)")
        elif rsi < 30:
            info_lines.append(f"✅ RSI: {rsi:.1f} (SOBREVENTA)")
        else:
            info_lines.append(f"📊 RSI: {rsi:.1f}")
        if 'BB_Upper' in self.df.columns:
            bb_upper = float(self.df['BB_Upper'].iloc[-1])
            bb_lower = float(self.df['BB_Lower'].iloc[-1])
            b_percent = (current_price - bb_lower) / max(1e-8, (bb_upper - bb_lower)) * 100
            info_lines.append(f"📍 %B: {b_percent:.1f}%")
        if self.wave_points:
            current_wave = self.identify_current_wave(self.wave_points, current_price)
            info_lines.append(f"🌊 {current_wave}")
        if self.final_decision and self.final_decision.get('signal'):
            fd = self.final_decision
            info_lines.append(f"🎯 {fd['action']} {fd['strength']} ({fd['score']:.2f})")
        return '\n'.join(info_lines)

    # ------------------------
    # Reportes inteligentes
    # ------------------------
    def generate_intelligent_report(self):
        self.report_text.delete(1.0, tk.END)
        if self.df is None:
            self.report_text.insert(tk.END, "Cargue datos primero.\n")
            return
        now = datetime.now().strftime('%Y-%m-%d %H:%M')
        current_price = float(self.df['Close'].iloc[-1])
        lines = []
        lines.append(f"# Reporte Inteligente - {self.symbol.get()} ({self.period.get()})")
        lines.append(f"Fecha: {now}")
        lines.append("")
        if not self.trading_signals:
            lines.append("No hay señales activas. Intente 'Detectar Señales'.")
        else:
            fd = self.final_decision
            if fd and fd.get('signal'):
                s = fd['signal']
                lines.append(f"## Decisión")
                lines.append(f"- Acción: {fd['action']} {fd['strength']}")
                lines.append(f"- Score ponderado: {fd['score']:.2f}")
                lines.append(f"- R:R: {s.risk_reward:.1f}:1 (mín. requerido {self.max_profit_config['min_risk_reward']:.1f}:1)")
                lines.append(f"- Entrada: ${s.precio:.2f} | Stop: ${s.stop_loss:.2f}")
                lines.append(f"- Objetivos: ${s.objetivo1:.2f} / ${s.objetivo2:.2f} / ${s.objetivo3:.2f}")
                lines.append(f"- Motivo principal: {s.razon} (src: {s.source})")
                lines.append("")
                lines.append("## Confluencias (ponderación)")
                for k, v in s.score_components.items():
                    lines.append(f"- {k}: {v:.2f} × peso {self.weights_config.get(k, 0.0):.2f}")
                lines.append("")
            lines.append("## Señales evaluadas (filtradas por R:R)")
            for s in self.trading_signals:
                lines.append(f"- {s.tipo} | {s.source} | Score {s.weighted_score:.2f} | R:R {s.risk_reward:.1f}:1 | {s.razon}")
        lines.append("")
        lines.append("Nota: Este reporte es informativo y no constituye asesoría financiera.")
        self.report_text.insert(tk.END, '\n'.join(lines))

    def export_report_markdown(self):
        content = self.report_text.get(1.0, tk.END).strip()
        if not content:
            messagebox.showinfo("Exportar", "Genere un reporte antes de exportar.")
            return
        reports_dir = os.path.join(os.getcwd(), 'reports')
        try:
            os.makedirs(reports_dir, exist_ok=True)
        except Exception:
            pass
        fname = f"reporte_{self.symbol.get()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        fpath = os.path.join(reports_dir, fname)
        try:
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(content)
            messagebox.showinfo("Exportar", f"✅ Reporte exportado en:\n{fpath}")
        except Exception as e:
            messagebox.showerror("Exportar", f"Error al exportar: {e}")


def main():
    root = tk.Tk()
    app = ElliottWaveAnalyzerPro(root)
    root.mainloop()


if __name__ == "__main__":
    main()
