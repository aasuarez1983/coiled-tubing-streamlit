import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import r2_score

st.set_page_config(layout="wide")
st.title("Visualización de Patrones de Coiled Tubing: Tendencias de Velocidad (Hasta 50 pozos)")

# Sidebar para configurar tamaño de zona y profundidad inicial de tendencia
st.sidebar.header("Configuración de zonas y tendencias")
zona_range = st.sidebar.number_input(
    "Rango de zona (metros)", min_value=50, max_value=2000, value=250, step=50,
    help="Ajusta el tamaño de cada zona para la tabla y los promedios"
)
profundidad_tendencia = st.sidebar.number_input(
    "Profundidad de inicio para análisis de tendencia (m)",
    min_value=0, max_value=10000, value=3250, step=50,
    help="Solo se analiza la tendencia a partir de esta profundidad"
)

st.subheader("Carga de Archivos (hasta 50 pozos)")
archivos = st.file_uploader(
    "Subí hasta 50 archivos de carreras (.xlsx)", type=["xlsx"], accept_multiple_files=True
)

def polinomio_info(x, y, grado):
    p = np.polyfit(x, y, grado)
    y_pred = np.polyval(p, x)
    r2 = r2_score(y, y_pred)
    if grado == 2:
        eq = f"y={p[0]:.2e}x²+{p[1]:.2e}x+{p[2]:.2f}"
    else:
        eq = f"Polinomio grado {grado}"
    return y_pred, r2, eq

def exponencial_info(x, y):
    x = np.array(x)
    y = np.array(y)
    mask = y > 0  # Solo valores positivos
    if np.sum(mask) < 3:
        return np.full_like(y, np.nan), np.nan, "Exponencial: Insuf. datos"
    x, y = x[mask], y[mask]
    p = np.polyfit(x, np.log(y), 1)
    y_pred = np.exp(np.polyval(p, x))
    r2 = r2_score(y, y_pred)
    eq = f"y=exp({p[0]:.3g}x+{p[1]:.2f})"
    x_full = np.array(sorted(x))
    y_pred_total = np.exp(np.polyval(p, x_full))
    return y_pred_total, r2, eq

def logaritmica_info(x, y):
    x = np.array(x)
    y = np.array(y)
    mask = x > 0  # Solo x positivos
    if np.sum(mask) < 3:
        return np.full_like(y, np.nan), np.nan, "Log: Insuf. datos"
    x, y = x[mask], y[mask]
    p = np.polyfit(np.log(x), y, 1)
    y_pred = np.polyval(p, np.log(x))
    r2 = r2_score(y, y_pred)
    eq = f"y={p[0]:.3g}ln(x)+{p[1]:.2f}"
    x_full = np.array(sorted(x))
    y_pred_total = np.polyval(p, np.log(x_full))
    return y_pred_total, r2, eq

if archivos:
    df_total = []
    colores = ['blue', 'gray', 'red', 'black', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'lime', 'deepskyblue', 'gold', 'navy', 'crimson', 'darkgreen', 'magenta', 'slategray', 'chocolate', 'khaki', 'orchid', 'teal', 'firebrick', 'olive', 'steelblue', 'tomato', 'indigo', 'coral', 'tan', 'springgreen', 'royalblue', 'chartreuse', 'peru', 'midnightblue', 'lawngreen', 'slateblue', 'saddlebrown', 'plum', 'maroon', 'deepskyblue', 'goldenrod', 'darkviolet', 'palegreen', 'turquoise', 'sienna', 'lightcoral', 'mediumslateblue', 'hotpink', 'cadetblue', 'burlywood', 'aqua']
    for archivo in archivos:
        xls = pd.ExcelFile(archivo)
        for hoja in xls.sheet_names:
            df = xls.parse(hoja)
            df["run_id"] = f"{archivo.name} - {hoja}"
            df["DateTime"] = pd.to_datetime(df["DateTime"])
            df["profundidad_m"] = df["CT-Profundidad(m)"].abs()
            df["tension_lb"] = df["CT-Tension Tuberia(lb)"].astype(float)
            df["velocidad_ftmin"] = df["CT-Velocidad de Viaje(ft/min)"].astype(float)
            df["caudal_bblmin"] = df["CT-Caudal Bombeo Liquido(bbl/min)"].astype(float)

            # Filtrar solo bajada
            df_bajada = df[df["velocidad_ftmin"] > 0].copy()
            if df_bajada.empty:
                continue

            # Hasta máxima profundidad en tiempo
            max_prof = df_bajada["profundidad_m"].max()
            idx_max = df_bajada[df_bajada["profundidad_m"] == max_prof].index[0]
            df_bajada = df_bajada.loc[:idx_max]

            # Quitar duplicados por profundidad y ordenar
            df_bajada = df_bajada.drop_duplicates(subset="profundidad_m", keep="first")
            df_bajada = df_bajada.sort_values("profundidad_m")

            # Suavizado Gaussiano
            df_bajada["tension_filtrada"] = gaussian_filter1d(df_bajada["tension_lb"], sigma=5)
            df_bajada["velocidad_filtrada"] = gaussian_filter1d(df_bajada["velocidad_ftmin"], sigma=5)
            df_bajada["caudal_filtrado"] = gaussian_filter1d(df_bajada["caudal_bblmin"], sigma=5)

            df_total.append(df_bajada)

    if df_total:
        df_concat = pd.concat(df_total)
        run_ids = df_concat["run_id"].unique()

        # -------- GRAFICOS ORIGINALES -----------
        st.subheader("Tensión vs Profundidad (todas las curvas)")
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        for i, run_id in enumerate(run_ids):
            grupo = df_concat[df_concat['run_id'] == run_id]
            ax1.plot(grupo["profundidad_m"], grupo["tension_filtrada"], label=run_id, color=colores[i % len(colores)])
        ax1.set_xlabel("Profundidad (m)")
        ax1.set_ylabel("Tensión Filtrada (lb)")
        ax1.set_title("Tensión vs Profundidad")
        ax1.legend(fontsize=7)
        ax1.grid(True)
        st.pyplot(fig1)

        st.subheader("Velocidad vs Profundidad (todas las curvas)")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        for i, run_id in enumerate(run_ids):
            grupo = df_concat[df_concat['run_id'] == run_id]
            ax2.plot(grupo["profundidad_m"], grupo["velocidad_filtrada"], label=run_id, color=colores[i % len(colores)])
        ax2.set_xlabel("Profundidad (m)")
        ax2.set_ylabel("Velocidad Filtrada (ft/min)")
        ax2.set_title("Velocidad vs Profundidad")
        ax2.legend(fontsize=7)
        ax2.grid(True)
        st.pyplot(fig2)

        st.subheader("Caudal vs Profundidad (todas las curvas)")
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        for i, run_id in enumerate(run_ids):
            grupo = df_concat[df_concat['run_id'] == run_id]
            ax3.plot(grupo["profundidad_m"], grupo["caudal_filtrado"], label=run_id, color=colores[i % len(colores)])
        ax3.set_xlabel("Profundidad (m)")
        ax3.set_ylabel("Caudal Filtrado (bbl/min)")
        ax3.set_title("Caudal vs Profundidad")
        ax3.legend(fontsize=7)
        ax3.grid(True)
        st.pyplot(fig3)

        # -------- GRAFICO VELOCIDAD PROMEDIO vs PROFUNDIDAD + TENDENCIAS SOLO DESDE PROFUNDIDAD SELECCIONADA --------
        profundidad_inicio = 2400
        profundidad_final = int(np.ceil(df_concat["profundidad_m"].max() / zona_range) * zona_range)
        zonas = []
        for start in range(profundidad_inicio, profundidad_final, int(zona_range)):
            end = start + int(zona_range)
            zonas.append((start, end))

        vel_plot_x = []
        vel_plot_y = []
        for start, end in zonas:
            mask = (df_concat["profundidad_m"] >= start) & (df_concat["profundidad_m"] < end)
            if mask.sum() > 0:
                vel_prom = df_concat.loc[mask, "velocidad_ftmin"].mean()
                vel_plot_x.append((start + end) / 2)
                vel_plot_y.append(vel_prom)
        st.subheader(f"Velocidad promedio vs profundidad (zonas desde {profundidad_inicio}m)")
        fig_vel, ax_vel = plt.subplots(figsize=(10, 5))
        ax_vel.plot(vel_plot_x, vel_plot_y, marker='o', color='indianred', label="Promedio")

        # --- TENDENCIAS SOLO DESDE PROFUNDIDAD SELECCIONADA ---
        mask_tend_vel = np.array(vel_plot_x) >= profundidad_tendencia
        x_tend_vel = np.array(vel_plot_x)[mask_tend_vel]
        y_tend_vel = np.array(vel_plot_y)[mask_tend_vel]

        if len(x_tend_vel) > 3:
            # Cuadrática
            y_quad, r2_quad, eq_quad = polinomio_info(x_tend_vel, y_tend_vel, 2)
            ax_vel.plot(x_tend_vel, y_quad, "--", color="firebrick", label=f"Cuadrática\n{eq_quad}\n$R^2$={r2_quad:.3f}")
            # Exponencial
            y_exp, r2_exp, eq_exp = exponencial_info(x_tend_vel, y_tend_vel)
            if not np.isnan(r2_exp):
                ax_vel.plot(np.sort(x_tend_vel), y_exp, "--", color="goldenrod", label=f"Exponencial\n{eq_exp}\n$R^2$={r2_exp:.3f}")
            # Logarítmica
            y_log, r2_log, eq_log = logaritmica_info(x_tend_vel, y_tend_vel)
            if not np.isnan(r2_log):
                ax_vel.plot(np.sort(x_tend_vel), y_log, "--", color="purple", label=f"Logarítmica\n{eq_log}\n$R^2$={r2_log:.3f}")

        ax_vel.set_xlabel("Profundidad (m)")
        ax_vel.set_ylabel("Velocidad Promedio (ft/min)")
        ax_vel.set_title(f"Velocidad promedio por zona (todas las carreras)")
        ax_vel.grid(True)
        ax_vel.legend(fontsize=9, loc="best")
        st.pyplot(fig_vel)

        # -------- TABLA DE PROMEDIOS POR ZONA (DESDE 2400m) --------
        st.subheader(f"Tabla de promedios por zona (cada {zona_range} m, desde {profundidad_inicio}m)")
        promedios_lista = []
        for start, end in zonas:
            mask = (df_concat["profundidad_m"] >= start) & (df_concat["profundidad_m"] < end)
            if mask.sum() > 0:
                promedio_vel = df_concat.loc[mask, "velocidad_ftmin"].mean()
                promedio_caudal = df_concat.loc[mask, "caudal_bblmin"].mean()
                promedios_lista.append({
                    "Zona": f"{start} - {end} m",
                    "Promedio Velocidad (ft/min)": round(promedio_vel, 2),
                    "Promedio Caudal (bbl/min)": round(promedio_caudal, 2)
                })

        df_promedios = pd.DataFrame(promedios_lista)
        st.dataframe(df_promedios, hide_index=True)

        # Descargar la tabla
        csv = df_promedios.to_csv(index=False).encode()
        st.download_button(
            "Descargar tabla CSV",
            data=csv,
            file_name=f"promedios_zonas_{zona_range}m_desde_{profundidad_inicio}m.csv",
            mime="text/csv"
        )