import streamlit as st
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import r2_score
import plotly.graph_objs as go

st.set_page_config(layout="wide")
st.title("Visualización de Patrones de Coiled Tubing: Comparativo Multi-Pozos (Plotly Interactivo)")

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
    mask = y > 0
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
    mask = x > 0
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
    colores = [
        "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#637939",
        "#8c6d31", "#843c39", "#7b4173", "#a55194", "#ce6dbd", "#6b6ecf",
        "#e7ba52", "#b5cf6b", "#ad494a", "#9c9ede", "#bd9e39", "#5254a3",
        "#e7969c", "#31a354", "#756bb1", "#636363", "#bcbddc", "#fdae6b",
        "#c7e9c0", "#fd8d3c", "#6baed6", "#b15928", "#ffff99", "#b2df8a",
        "#a6cee3", "#cab2d6", "#fb9a99", "#fdbf6f", "#e41a1c", "#377eb8",
        "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf", "#999999"
    ]
    for archivo in archivos:
        xls = pd.ExcelFile(archivo)
        for hoja in xls.sheet_names:
            df = xls.parse(hoja)
            df["run_id"] = hoja  # Solo el nombre de la hoja/tab
            df["DateTime"] = pd.to_datetime(df["DateTime"])
            df["profundidad_m"] = df["CT-Profundidad(m)"].abs()
            df["tension_lb"] = df["CT-Tension Tuberia(lb)"].astype(float)
            df["velocidad_ftmin"] = df["CT-Velocidad de Viaje(ft/min)"].astype(float)
            df["caudal_bblmin"] = df["CT-Caudal Bombeo Liquido(bbl/min)"].astype(float)

            # Filtrar solo bajada
            df_bajada = df[df["velocidad_ftmin"] > 0].copy()
            if df_bajada.empty:
                continue
            max_prof = df_bajada["profundidad_m"].max()
            idx_max = df_bajada[df_bajada["profundidad_m"] == max_prof].index[0]
            df_bajada = df_bajada.loc[:idx_max]
            df_bajada = df_bajada.drop_duplicates(subset="profundidad_m", keep="first")
            df_bajada = df_bajada.sort_values("profundidad_m")
            df_bajada["tension_filtrada"] = gaussian_filter1d(df_bajada["tension_lb"], sigma=5)
            df_bajada["velocidad_filtrada"] = gaussian_filter1d(df_bajada["velocidad_ftmin"], sigma=5)
            df_bajada["caudal_filtrado"] = gaussian_filter1d(df_bajada["caudal_bblmin"], sigma=5)
            df_total.append(df_bajada)

    if df_total:
        df_concat = pd.concat(df_total)
        run_ids = df_concat["run_id"].unique()

        # ------- Selector de pozos para visualización ------
        st.subheader("Selección de pozos a visualizar")
        pozos_a_ver = st.multiselect(
            "Selecciona los pozos a mostrar:",
            options=list(run_ids),
            default=list(run_ids)[:3]
        )

        # --------- GRAFICO TENSIÓN vs PROFUNDIDAD (Plotly con doble eje X) ---------
        st.subheader("Tensión vs Profundidad (selección de pozos)")
        fig_tension = go.Figure()
        for i, run_id in enumerate(pozos_a_ver):
            grupo = df_concat[df_concat['run_id'] == run_id]
            fig_tension.add_trace(go.Scatter(
                x=grupo["profundidad_m"], y=grupo["tension_filtrada"],
                name=run_id,
                mode="lines",
                line=dict(color=colores[i % len(colores)]),
                hovertemplate=run_id + "<br>Profundidad: %{x:.1f} m<br>Tensión: %{y:.1f} lb"
            ))
        fig_tension.update_layout(
            xaxis=dict(title="Profundidad (m)", side="bottom"),
            xaxis2=dict(
                title="Profundidad (m)", side="top", overlaying="x", showgrid=False, showline=True, zeroline=False
            ),
            yaxis=dict(title="Tensión Filtrada (lb)"),
            legend=dict(font=dict(size=8)),
            width=950, height=500,
            margin=dict(l=60, r=40, t=60, b=60),
        )
        st.plotly_chart(fig_tension, use_container_width=True)

        # --------- GRAFICO TENSIÓN vs PROFUNDIDAD con ZOOM por input box ---------
        st.subheader("Zoom lateral: Tensión vs Profundidad (definido por rango)")
        min_prof = float(df_concat["profundidad_m"].min())
        max_prof = float(df_concat["profundidad_m"].max())

        st.sidebar.markdown("### Rango para zoom de Tensión vs Profundidad")
        zoom_start = st.sidebar.number_input(
            "Profundidad inicial (m) para zoom", min_value=int(min_prof), max_value=int(max_prof-1),
            value=int(min_prof), step=10
        )
        zoom_end = st.sidebar.number_input(
            "Profundidad final (m) para zoom", min_value=zoom_start+1, max_value=int(max_prof),
            value=min(int(zoom_start+1000), int(max_prof)), step=10
        )
        # Ajustar si el rango supera 1000m
        if zoom_end - zoom_start > 1000:
            zoom_end = zoom_start + 1000
            st.sidebar.warning("El rango de zoom no puede ser mayor a 1000 m. Se ajustó automáticamente.")

        fig_tension_zoom = go.Figure()
        for i, run_id in enumerate(pozos_a_ver):
            grupo = df_concat[df_concat['run_id'] == run_id]
            grupo_zoom = grupo[(grupo["profundidad_m"] >= zoom_start) & (grupo["profundidad_m"] <= zoom_end)]
            fig_tension_zoom.add_trace(go.Scatter(
                x=grupo_zoom["profundidad_m"], y=grupo_zoom["tension_filtrada"],
                name=run_id,
                mode="lines",
                line=dict(color=colores[i % len(colores)]),
                hovertemplate=run_id + "<br>Profundidad: %{x:.1f} m<br>Tensión: %{y:.1f} lb"
            ))
        # Ajusta eje Y automáticamente para ese rango
        all_y = []
        for run_id in pozos_a_ver:
            grupo = df_concat[df_concat['run_id'] == run_id]
            grupo_zoom = grupo[(grupo["profundidad_m"] >= zoom_start) & (grupo["profundidad_m"] <= zoom_end)]
            all_y.extend(grupo_zoom["tension_filtrada"].values)
        if all_y:
            y_min = min(all_y)
            y_max = max(all_y)
        else:
            y_min, y_max = 0, 1
        fig_tension_zoom.update_layout(
            xaxis=dict(title="Profundidad (m)", range=[zoom_start, zoom_end]),
            yaxis=dict(title="Tensión Filtrada (lb)", range=[y_min, y_max]),
            legend=dict(font=dict(size=8)),
            width=950, height=400,
            margin=dict(l=60, r=40, t=60, b=60),
        )
        st.plotly_chart(fig_tension_zoom, use_container_width=True)

        # --------- GRAFICO VELOCIDAD vs PROFUNDIDAD (Plotly con doble eje X) ---------
        st.subheader("Velocidad vs Profundidad (selección de pozos)")
        fig_vel = go.Figure()
        for i, run_id in enumerate(pozos_a_ver):
            grupo = df_concat[df_concat['run_id'] == run_id]
            fig_vel.add_trace(go.Scatter(
                x=grupo["profundidad_m"], y=grupo["velocidad_filtrada"],
                name=run_id,
                mode="lines",
                line=dict(color=colores[i % len(colores)]),
                hovertemplate=run_id + "<br>Profundidad: %{x:.1f} m<br>Velocidad: %{y:.2f} ft/min"
            ))
        fig_vel.update_layout(
            xaxis=dict(title="Profundidad (m)", side="bottom"),
            xaxis2=dict(
                title="Profundidad (m)", side="top", overlaying="x", showgrid=False, showline=True, zeroline=False
            ),
            yaxis=dict(title="Velocidad Filtrada (ft/min)"),
            legend=dict(font=dict(size=8)),
            width=950, height=500,
            margin=dict(l=60, r=40, t=60, b=60),
        )
        st.plotly_chart(fig_vel, use_container_width=True)

        # --------- GRAFICO CAUDAL vs PROFUNDIDAD (Plotly con doble eje X) ---------
        st.subheader("Caudal vs Profundidad (selección de pozos)")
        fig_caudal = go.Figure()
        for i, run_id in enumerate(pozos_a_ver):
            grupo = df_concat[df_concat['run_id'] == run_id]
            fig_caudal.add_trace(go.Scatter(
                x=grupo["profundidad_m"], y=grupo["caudal_filtrado"],
                name=run_id,
                mode="lines",
                line=dict(color=colores[i % len(colores)]),
                hovertemplate=run_id + "<br>Profundidad: %{x:.1f} m<br>Caudal: %{y:.2f} bbl/min"
            ))
        fig_caudal.update_layout(
            xaxis=dict(title="Profundidad (m)", side="bottom"),
            xaxis2=dict(
                title="Profundidad (m)", side="top", overlaying="x", showgrid=False, showline=True, zeroline=False
            ),
            yaxis=dict(title="Caudal Filtrado (bbl/min)"),
            legend=dict(font=dict(size=8)),
            width=950, height=500,
            margin=dict(l=60, r=40, t=60, b=60),
        )
        st.plotly_chart(fig_caudal, use_container_width=True)

        # --- PROMEDIOS POR ZONA Y GRAFICOS DE COMPARACION Y TENDENCIA ---
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
        x_min = profundidad_inicio
        x_max = df_concat["profundidad_m"].max()
        y_min = min(vel_plot_y) if vel_plot_y else 0
        y_max = max(vel_plot_y) if vel_plot_y else 10

        # --------- GRAFICO PROMEDIO vs VARIOS POZOS SELECCIONADOS ---------
        st.subheader("Comparación de velocidad promedio vs pozos seleccionados (zonas)")
        all_options = ["Promedio"] + list(run_ids)
        pozos_sel = st.multiselect(
            "Selecciona pozos a comparar (puedes incluir 'Promedio'):",
            options=all_options,
            default=["Promedio"]
        )
        fig_comp = go.Figure()
        color_map = {"Promedio": 'dodgerblue'}
        for idx, sel in enumerate(pozos_sel):
            if sel == "Promedio":
                fig_comp.add_trace(go.Scatter(
                    x=vel_plot_x, y=vel_plot_y, mode='lines+markers',
                    name='Promedio',
                    marker=dict(size=8, color=color_map["Promedio"]),
                    line=dict(color=color_map["Promedio"], width=2),
                    hovertemplate="Promedio<br>Profundidad: %{x:.1f} m<br>Velocidad: %{y:.2f} ft/min"
                ))
            else:
                grupo = df_concat[df_concat['run_id'] == sel]
                vel_pozo_x = []
                vel_pozo_y = []
                for start, end in zonas:
                    mask = (grupo["profundidad_m"] >= start) & (grupo["profundidad_m"] < end)
                    if mask.sum() > 0:
                        vel_prom = grupo.loc[mask, "velocidad_ftmin"].mean()
                        vel_pozo_x.append((start + end) / 2)
                        vel_pozo_y.append(vel_prom)
                fig_comp.add_trace(go.Scatter(
                    x=vel_pozo_x, y=vel_pozo_y, mode='lines+markers',
                    name=sel,
                    marker=dict(size=8, color=colores[(idx+1) % len(colores)]),
                    line=dict(color=colores[(idx+1) % len(colores)], width=2),
                    hovertemplate=f"{sel}<br>Profundidad: "+"%{x:.1f} m<br>Velocidad: %{y:.2f} ft/min"
                ))
        fig_comp.update_layout(
            xaxis=dict(title="Profundidad (m)", side="bottom", range=[x_min, x_max], dtick=zona_range),
            xaxis2=dict(title="Profundidad (m)", side="top", overlaying="x", showgrid=False, showline=True, zeroline=False),
            yaxis=dict(title="Velocidad (ft/min)", range=[y_min-1, y_max+1], dtick=2),
            legend=dict(font=dict(size=10)),
            width=900, height=500
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # --------- GRAFICO PROMEDIO + TENDENCIA SELECCIONABLE ---------
        st.subheader("Velocidad promedio vs profundidad (zonas) con tendencia seleccionable")
        tipo_tendencia = st.selectbox(
            "Elige el tipo de tendencia a mostrar:",
            options=["Cuadrática", "Exponencial", "Logarítmica"],
            index=0
        )
        mask_tend_vel = np.array(vel_plot_x) >= profundidad_tendencia
        x_tend_vel = np.array(vel_plot_x)[mask_tend_vel]
        y_tend_vel = np.array(vel_plot_y)[mask_tend_vel]
        fig_tend = go.Figure()
        fig_tend.add_trace(go.Scatter(
            x=vel_plot_x, y=vel_plot_y, mode='lines+markers',
            name='Promedio',
            marker=dict(size=8, color='indianred'),
            line=dict(color='indianred', width=2),
            hovertemplate="Promedio<br>Profundidad: %{x:.1f} m<br>Velocidad: %{y:.2f} ft/min"
        ))
        if len(x_tend_vel) > 3:
            if tipo_tendencia == "Cuadrática":
                y_tend, r2, eq = polinomio_info(x_tend_vel, y_tend_vel, 2)
                color = 'firebrick'
            elif tipo_tendencia == "Exponencial":
                y_tend, r2, eq = exponencial_info(x_tend_vel, y_tend_vel)
                color = 'goldenrod'
            elif tipo_tendencia == "Logarítmica":
                y_tend, r2, eq = logaritmica_info(x_tend_vel, y_tend_vel)
                color = 'purple'
            fig_tend.add_trace(go.Scatter(
                x=x_tend_vel, y=y_tend, mode='lines',
                name=f"{tipo_tendencia} <br>{eq}<br>R²={r2:.3f}",
                line=dict(dash='dash', color=color)
            ))
        fig_tend.update_layout(
            xaxis=dict(title="Profundidad (m)", side="bottom", range=[x_min, x_max], dtick=zona_range),
            xaxis2=dict(title="Profundidad (m)", side="top", overlaying="x", showgrid=False, showline=True, zeroline=False),
            yaxis=dict(title="Velocidad Promedio (ft/min)", range=[y_min-1, y_max+1], dtick=2),
            legend=dict(font=dict(size=10)),
            width=900, height=500
        )
        st.plotly_chart(fig_tend, use_container_width=True)

        # --------- TABLA DE PROMEDIOS POR ZONA ---------
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
        csv = df_promedios.to_csv(index=False).encode()
        st.download_button(
            "Descargar tabla CSV",
            data=csv,
            file_name=f"promedios_zonas_{zona_range}m_desde_{profundidad_inicio}m.csv",
            mime="text/csv"
        )
