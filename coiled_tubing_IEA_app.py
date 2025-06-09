import streamlit as st
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import r2_score
import plotly.graph_objs as go
import unidecode
import re

st.set_page_config(layout="wide")
st.title("Visualización de Patrones de Coiled Tubing: Comparativo Multi-Pozos (Plotly Interactivo)")

# --- Diccionario de nombres equivalentes de columnas ---
NOMBRES_EQUIVALENTES = {
    "CT-Tension Tuberia(lb)": "CT-Tension Tuberia(lb)",
    "CT-Peso Tuberia(lb)": "CT-Tension Tuberia(lb)",
    "CT-Profundidad(m)": "CT-Profundidad(m)",
    "Measured Depth (m)": "CT-Profundidad(m)",
    # Puedes agregar más equivalentes aquí si surgen más casos
}

def limpiar_decimal_universal(col_serie):
    def corrige(valor):
        if pd.isnull(valor) or str(valor).strip() == '':
            return np.nan
        s = str(valor).strip().replace(" ", "")
        s = re.sub(r"[^0-9\-,\.eE]", "", s)
        if s.count(',') and s.count('.'):
            if s.rfind(',') > s.rfind('.'):
                s = s.replace('.', '').replace(',', '.')
            else:
                s = s.replace(',', '')
        elif s.count(','):
            if len(s.split(',')[-1]) <= 3:
                s = s.replace(',', '.')
            else:
                s = s.replace(',', '')
        try:
            return float(s)
        except Exception:
            return np.nan
    return col_serie.apply(corrige)

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

st.subheader("Carga de Archivos Fecha, Profundidad, Peso, Caudal (hasta 50 pozos)")
archivos = st.file_uploader(
    "Subí hasta 50 archivos de carreras (.xlsx)", type=["xlsx"], accept_multiple_files=True
)

st.subheader("Carga archivo de DoglegSeverity y Tortuosidad (opcional, .xlsx, 1 archivo, varias hojas posible ) (Nota: EL NOMBRE DE LOS POZOS DEBE SER EXACTAMENTE IGUAL)")
archivo_dogleg = st.file_uploader(
    "Carga archivo de Dogleg/Tortuosidad (.xlsx)", type=["xlsx"], accept_multiple_files=False
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
            df = xls.parse(hoja, dtype=str)
            # --- Normalizar nombres de columnas ---
            df.columns = [NOMBRES_EQUIVALENTES.get(c, c) for c in df.columns]

            # Listado de columnas clave
            columnas_ct = [
                "CT-Profundidad(m)", "CT-Tension Tuberia(lb)",
                "CT-Velocidad de Viaje(ft/min)", "CT-Caudal Bombeo Liquido(bbl/min)"
            ]

            # Verifica presencia de cada columna
            for col in columnas_ct:
                if col not in df.columns:
                    st.error(
                        f"❌ En la hoja **'{hoja}'** falta la columna obligatoria '**{col}**'.\n"
                        f"Columnas disponibles: {list(df.columns)}"
                    )
                    # Salta el procesamiento de esta hoja
                    break
            else:
                # Si no faltó ninguna columna, continúa procesando normalmente:
                for col in columnas_ct:
                    df[col] = limpiar_decimal_universal(df[col])

                df["run_id"] = hoja
                if "DateTime" in df.columns:
                    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
                df["profundidad_m"] = pd.to_numeric(df["CT-Profundidad(m)"], errors='coerce').abs()
                df["tension_lb"] = pd.to_numeric(df["CT-Tension Tuberia(lb)"], errors='coerce')
                df["velocidad_ftmin"] = pd.to_numeric(df["CT-Velocidad de Viaje(ft/min)"], errors='coerce')
                df["caudal_bblmin"] = pd.to_numeric(df["CT-Caudal Bombeo Liquido(bbl/min)"], errors='coerce')
                # ... el resto de tu lógica ...
                df_bajada = df[df["velocidad_ftmin"] > 0].copy()
                if df_bajada.empty:
                    continue
                max_prof = df_bajada["profundidad_m"].max()
                idx_max = df_bajada[df_bajada["profundidad_m"] == max_prof].index[0]
                df_bajada = df_bajada.loc[:idx_max]
                df_bajada = df_bajada.drop_duplicates(subset="profundidad_m", keep="first")
                df_bajada = df_bajada.sort_values("profundidad_m")
                df_bajada["tension_filtrada"] = gaussian_filter1d(df_bajada["tension_lb"].fillna(0), sigma=5)
                df_bajada["velocidad_filtrada"] = gaussian_filter1d(df_bajada["velocidad_ftmin"].fillna(0), sigma=5)
                df_bajada["caudal_filtrado"] = gaussian_filter1d(df_bajada["caudal_bblmin"].fillna(0), sigma=5)
                df_total.append(df_bajada)

    # --- Carga de datos de Dogleg/Tortuosidad (opcional) ---
    dogleg_data_dict = {}
    if archivo_dogleg is not None:
        xls_dogleg = pd.ExcelFile(archivo_dogleg)
        for hoja in xls_dogleg.sheet_names:
            df_dogleg = xls_dogleg.parse(hoja, dtype=str)
            # --- Normalizar nombres de columnas ---
            df_dogleg.columns = [NOMBRES_EQUIVALENTES.get(c, c) for c in df_dogleg.columns]
            # --- Normalizar columnas con unidecode ---
            df_dogleg.columns = [unidecode.unidecode(str(c).strip().replace(" ", "_").lower()) for c in df_dogleg.columns]
            prof_cols = [c for c in df_dogleg.columns if "profun" in c]
            dogleg_cols = [c for c in df_dogleg.columns if "dogleg" in c]
            tort_cols = [c for c in df_dogleg.columns if "tortuo" in c]
            if not (prof_cols and dogleg_cols and tort_cols):
                st.warning(
                    f"No se encontraron todas las columnas requeridas en la hoja '{hoja}'.\n"
                    f"Columnas encontradas: {df_dogleg.columns.tolist()}"
                )
                continue
            for col in [prof_cols[0], dogleg_cols[0], tort_cols[0]]:
                if col in df_dogleg.columns:
                    df_dogleg[col] = limpiar_decimal_universal(df_dogleg[col])
            df_dogleg = df_dogleg[
                pd.to_numeric(df_dogleg[prof_cols[0]], errors='coerce').notnull()
            ].copy()
            df_dogleg[prof_cols[0]] = pd.to_numeric(df_dogleg[prof_cols[0]], errors='coerce')
            df_dogleg["profundidad_m"] = df_dogleg[prof_cols[0]].abs()
            df_dogleg[dogleg_cols[0]] = pd.to_numeric(df_dogleg[dogleg_cols[0]], errors='coerce')
            df_dogleg[tort_cols[0]] = pd.to_numeric(df_dogleg[tort_cols[0]], errors='coerce')
            dogleg_data_dict[hoja] = df_dogleg[["profundidad_m", dogleg_cols[0], tort_cols[0]]].rename(
                columns={dogleg_cols[0]: "dogleg", tort_cols[0]: "tortuosidad"}
            )


    if df_total:
        df_concat = pd.concat(df_total)
        run_ids = df_concat["run_id"].unique()

        st.subheader("Selección de pozos a visualizar")
        pozos_a_ver = st.multiselect(
            "Selecciona los pozos a mostrar:",
            options=list(run_ids),
            default=list(run_ids)[:3]
        )

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

        # ==== GRAFICOS INDIVIDUALES DUALES POR POZO: VELOCIDAD, DOGLEG Y TORTUOSIDAD ====
        st.subheader(
            "Gráficos individuales por pozo: Velocidad, DoglegSeverity, Tortuosidad e Índices de Eficiencia vs Profundidad (Ejes independientes, curvas opcionales)")

        for idx, run_id in enumerate(pozos_a_ver):
            st.markdown(f"**Pozo: {run_id}**")
            grupo = df_concat[df_concat['run_id'] == run_id].copy()

            # Checkboxes para mostrar/ocultar curvas
            # Por defecto: IEA1 (mecánico) DESHABILITADO, resto habilitados
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                show_tortuosidad = st.checkbox("Mostrar Tortuosidad", value=True, key=f"tort_{run_id}")
            with col2:
                show_dogleg = st.checkbox("Mostrar Dogleg", value=True, key=f"dog_{run_id}")
            with col3:
                show_iea1 = st.checkbox("Mostrar IEA1 (mecánico)", value=False, key=f"iea1_{run_id}")
            with col4:
                show_iea2 = st.checkbox("Mostrar IEA2 (geométrico)", value=True, key=f"iea2_{run_id}")

            # Merge dogleg y tortuosidad si están disponibles
            if dogleg_data_dict and run_id in dogleg_data_dict:
                dogleg_df = dogleg_data_dict[run_id].copy()
                grupo = pd.merge_asof(
                    grupo.sort_values('profundidad_m'),
                    dogleg_df.sort_values('profundidad_m'),
                    on='profundidad_m', direction='nearest', tolerance=5
                )
                grupo["dogleg"].fillna(0, inplace=True)
                grupo["tortuosidad"].fillna(0, inplace=True)
            else:
                grupo["dogleg"] = 0
                grupo["tortuosidad"] = 0

            # Índices de eficiencia
            grupo["indice_eficiencia_mecanica"] = grupo["velocidad_filtrada"] / (
                        grupo["tension_filtrada"].abs() * (1 + grupo["dogleg"]))
            grupo["indice_eficiencia_geom"] = grupo["velocidad_filtrada"] / (1 + grupo["dogleg"] + grupo["tortuosidad"])

            fig = go.Figure()

            # Siempre se muestra la velocidad
            fig.add_trace(go.Scatter(
                x=grupo["profundidad_m"], y=grupo["velocidad_filtrada"],
                name="Velocidad (ft/min)",
                mode="lines",
                line=dict(color="dodgerblue", width=2),
                hovertemplate="Velocidad<br>Profundidad: %{x} m<br>Vel: %{y} ft/min",
                yaxis="y1"
            ))

            # IEA1 (mecánico)
            if show_iea1:
                fig.add_trace(go.Scatter(
                    x=grupo["profundidad_m"], y=grupo["indice_eficiencia_mecanica"],
                    name="Índice Eficiencia (IEA1: mecánico)",
                    mode="lines",
                    line=dict(color="red", width=3, dash="dot"),
                    hovertemplate="IEA1<br>Profundidad: %{x} m<br>Índice: %{y}",
                    yaxis="y2"
                ))

            # IEA2 (geométrico)
            if show_iea2:
                fig.add_trace(go.Scatter(
                    x=grupo["profundidad_m"], y=grupo["indice_eficiencia_geom"],
                    name="Índice Eficiencia (IEA2: geométrico)",
                    mode="lines",
                    line=dict(color="black", width=3, dash="dash"),
                    hovertemplate="IEA2<br>Profundidad: %{x} m<br>Índice: %{y}",
                    yaxis="y3"
                ))

            # Dogleg
            if show_dogleg:
                fig.add_trace(go.Scatter(
                    x=grupo["profundidad_m"], y=grupo["dogleg"],
                    name="DoglegSeverity (°/30m)",
                    mode="lines",
                    line=dict(color="firebrick", width=2, dash="dash"),
                    hovertemplate="Dogleg<br>Profundidad: %{x} m<br>Dogleg: %{y} °/30m",
                    yaxis="y4"
                ))

            # Tortuosidad
            if show_tortuosidad:
                fig.add_trace(go.Scatter(
                    x=grupo["profundidad_m"], y=grupo["tortuosidad"],
                    name="Tortuosidad (°/30m)",
                    mode="lines",
                    line=dict(color="green", width=2, dash="dot"),
                    hovertemplate="Tortuosidad<br>Profundidad: %{x} m<br>Tortuosidad: %{y} °/30m",
                    yaxis="y5"
                ))

            fig.update_layout(
                xaxis=dict(title="Profundidad (m)"),
                yaxis=dict(
                    title="Velocidad (ft/min)", color="dodgerblue", side="left", position=0.0
                ),
                yaxis2=dict(
                    title="IEA1 (mecánico)",
                    color="red",
                    anchor="x",
                    overlaying="y",
                    side="right",
                    position=1.0
                ),
                yaxis3=dict(
                    title="IEA2 (geométrico)",
                    color="black",
                    anchor="free",
                    overlaying="y",
                    side="right",
                    position=0.95,
                    showgrid=False
                ),
                yaxis4=dict(
                    title="DoglegSeverity (°/30m)",
                    color="firebrick",
                    anchor="free",
                    overlaying="y",
                    side="right",
                    position=0.91,
                    range=[0, max(15, grupo["dogleg"].max() * 1.1)],
                    showgrid=False
                ),
                yaxis5=dict(
                    title="Tortuosidad (°/30m)",
                    color="green",
                    anchor="free",
                    overlaying="y",
                    side="right",
                    position=0.87,
                    range=[0, max(2, grupo["tortuosidad"].max() * 1.1)],
                    showgrid=False
                ),
                legend=dict(font=dict(size=20)),  # Doble del tamaño original
                width=1100, height=500,
                margin=dict(l=80, r=80, t=60, b=60),
            )

            st.plotly_chart(fig, use_container_width=True)
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

# ============================ ## ============================ ## ============================ ## ============================ ## ============================ #
# ============================ ## ============================ ##   MÓDULO DE MACHINE LEARNING ## ============================ ## ============================ #
# ============================ ## ============================ ## ============================ ## ============================ ## ============================ #

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import pickle
import io
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.markdown("---")
st.header("🔮 Módulo de Predicción Global de Velocidad con Machine Learning")

if "df_concat" in locals() and not df_concat.empty:
    # Merge dogleg y tortuosidad para todos los pozos
    df_ml = df_concat.copy()
    if dogleg_data_dict:
        frames = []
        for run_id in df_ml["run_id"].unique():
            df_p = df_ml[df_ml["run_id"] == run_id].copy()
            if run_id in dogleg_data_dict:
                df_dog = dogleg_data_dict[run_id].copy()
                df_merged = pd.merge_asof(
                    df_p.sort_values("profundidad_m"),
                    df_dog.sort_values("profundidad_m"),
                    on="profundidad_m", direction="nearest", tolerance=5
                )
            else:
                df_p["dogleg"] = 0
                df_p["tortuosidad"] = 0
                df_merged = df_p
            frames.append(df_merged)
        df_ml = pd.concat(frames, ignore_index=True)
    else:
        df_ml["dogleg"] = 0
        df_ml["tortuosidad"] = 0

    features = ["profundidad_m", "dogleg", "tortuosidad"]
    target = "velocidad_filtrada" if "velocidad_filtrada" in df_ml else "velocidad_ftmin"

    df_train = df_ml.dropna(subset=features + [target]).copy()
    if len(df_train) < 30:
        st.warning("No hay suficientes datos en la base total para entrenar un modelo robusto.")
    else:
        X = df_train[features]
        y = df_train[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

        # Entrenamiento del modelo global
        model = RandomForestRegressor(n_estimators=120, random_state=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        st.write(f"**Score R² global:** {r2:.3f} | **MSE:** {mse:.2f} | Muestras de entrenamiento: {len(df_train)}")

        # Exportar modelo entrenado
        buf = io.BytesIO()
        pickle.dump(model, buf)
        buf.seek(0)
        st.download_button(
            label="⬇️ Descargar modelo entrenado (.pkl)",
            data=buf,
            file_name="modelo_coiled_tubing_rf.pkl",
            mime="application/octet-stream"
        )

        # Importancia de variables
        importancias = model.feature_importances_
        import matplotlib.pyplot as plt
        fig_import = plt.figure(figsize=(4, 2.5))
        plt.barh(features, importancias, color='teal')
        plt.title("Importancia de variables en el modelo")
        plt.tight_layout()
        st.pyplot(fig_import)
        st.info(
            f"Tortuosidad representa el {importancias[2] * 100:.1f}% de la importancia para predecir la velocidad."
            if importancias[2] > max(importancias[0], importancias[1]) else
            f"Tortuosidad NO es la principal variable (importancia: {importancias[2] * 100:.1f}%)."
        )

        # Comparar curva real vs predicha de cualquier pozo cargado
        st.subheader("Comparar curva real y predicha de velocidad para un pozo cargado")
        pozo_sel = st.selectbox("Selecciona un pozo para comparar real vs predicho:", df_ml["run_id"].unique())
        df_comp = df_ml[df_ml["run_id"] == pozo_sel].dropna(subset=features)
        if not df_comp.empty:
            y_pred_comp = model.predict(df_comp[features])
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(
                x=df_comp["profundidad_m"], y=df_comp[target],
                mode="lines+markers", name="Velocidad real", marker=dict(color="dodgerblue")
            ))
            fig_comp.add_trace(go.Scatter(
                x=df_comp["profundidad_m"], y=y_pred_comp,
                mode="lines+markers", name="Velocidad predicha", marker=dict(color="orange")
            ))
            fig_comp.update_layout(
                xaxis_title="Profundidad (m)",
                yaxis_title="Velocidad (ft/min)",
                title=f"Comparación velocidad real vs predicha en pozo '{pozo_sel}'",
                width=800, height=400
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.warning(f"No hay datos completos para el pozo seleccionado ({pozo_sel}).")

        # Curva IEW
        st.subheader("Curva de IEW (Índice Exploratorio de Velocidad)")
        df_train["IEW"] = df_train[target] / (1 + df_train["tortuosidad"] + df_train["dogleg"])
        fig_iew = go.Figure()
        fig_iew.add_trace(go.Scatter(
            x=df_train["profundidad_m"], y=df_train["IEW"],
            mode='markers', name="IEW",
            marker=dict(color='mediumvioletred', size=3),
            hovertemplate="Profundidad: %{x:.1f} m<br>IEW: %{y:.2f}"
        ))
        fig_iew.update_layout(
            xaxis_title="Profundidad (m)",
            yaxis_title="IEW",
            title="IEW vs Profundidad (base global)",
            width=800, height=350
        )
        st.plotly_chart(fig_iew, use_container_width=True)

        st.caption(
            "IEW = Velocidad / (1 + Tortuosidad + Dogleg). Si la curva IEW se mantiene estable o revela correlación, "
            "puede ser útil como predictor adicional de eficiencia o fricción geométrica."
        )

        # -----------------------------------------------------
        #   APLICAR MODELO SOBRE DATA TEÓRICA (ARCHIVO ADICIONAL)
        # -----------------------------------------------------
        st.markdown("### 🧪 Predicción de velocidad sobre datos teóricos cargados por el usuario")
        archivo_teorico = st.file_uploader(
            "Cargar archivo con datos teóricos de Profundidad, Tortuosidad y Dogleg (.xlsx, .csv)",
            type=["xlsx", "csv"], key="teorico"
        )
        if archivo_teorico is not None:
            if archivo_teorico.name.endswith(".xlsx"):
                df_teorico = pd.read_excel(archivo_teorico)
            else:
                df_teorico = pd.read_csv(archivo_teorico)
            # Mostrar columnas detectadas
            st.write("Columnas en el archivo teórico:", list(df_teorico.columns))
            columnas_teorico = [str(col).strip().lower() for col in df_teorico.columns]
            cols_dict = {}
            for i, c in enumerate(columnas_teorico):
                if "profun" in c:
                    cols_dict["profundidad_m"] = df_teorico.columns[i]
                if "tortuo" in c:
                    cols_dict["tortuosidad"] = df_teorico.columns[i]
                if "dogleg" in c:
                    cols_dict["dogleg"] = df_teorico.columns[i]
            st.write("Columnas equivalentes encontradas:", cols_dict)
            faltantes = [c for c in features if c not in cols_dict]
            if not faltantes:
                df_teorico_pred = df_teorico.rename(columns={v: k for k, v in cols_dict.items()})
                df_teorico_pred = df_teorico_pred[features]
                # Predecir
                y_pred_teo = model.predict(df_teorico_pred)
                df_teorico_pred["velocidad_predicha"] = y_pred_teo

                # Score ponderado respecto al set de entrenamiento
                mean_vel = y.mean()
                std_vel = y.std()
                df_teorico_pred['score_normalizado'] = (df_teorico_pred['velocidad_predicha'] - mean_vel) / std_vel
                df_teorico_pred['ponderacion_prediccion'] = norm.cdf(df_teorico_pred['score_normalizado'])
                st.info(
                    "La ponderación de predicción ('ponderacion_prediccion') indica qué tan 'típica' es la velocidad predicha respecto al historial. "
                    "Valores cercanos a 0.5 son muy habituales, extremos (<0.1 o >0.9) indican valores atípicos respecto al set de entrenamiento."
                )

                # Gráfico de predicción
                fig_teo = go.Figure()
                fig_teo.add_trace(go.Scatter(
                    x=df_teorico_pred["profundidad_m"], y=df_teorico_pred["velocidad_predicha"],
                    mode='lines+markers', name="Velocidad Predicha",
                    marker=dict(color='royalblue')
                ))
                fig_teo.update_layout(
                    title="Predicción de Velocidad sobre data teórica",
                    xaxis_title="Profundidad (m)",
                    yaxis_title="Velocidad predicha (ft/min)",
                    width=800, height=350
                )
                st.plotly_chart(fig_teo, use_container_width=True)
                st.dataframe(df_teorico_pred)

                # --- Cálculo optimizado del tiempo de RIH usando integración por tramos (solo donde velocidad > 1 ft/min) ---

                velocidad_min_ftmin = 1.0  # Umbral físico de velocidad mínima aceptable (puedes ajustar)
                prof_column = "profundidad_m"
                vel_column = "velocidad_predicha"

                # Ordena el dataframe por profundidad (por si acaso)
                df_teorico_pred = df_teorico_pred.sort_values(prof_column).reset_index(drop=True)

                # Convierte velocidades a metros/min si está en ft/min (ajusta si tus unidades ya son m/min)
                ft_to_m = 0.3048
                df_teorico_pred["vel_pred_m_min"] = df_teorico_pred[vel_column] * ft_to_m

                # Calcula diferencias de profundidad
                df_teorico_pred["delta_prof"] = df_teorico_pred[prof_column].diff().fillna(0)

                # Para evitar divisiones por cero o por velocidades irreales
                df_teorico_pred["vel_pred_m_min_filtrado"] = df_teorico_pred["vel_pred_m_min"].where(
                    df_teorico_pred["vel_pred_m_min"] > 0.5, np.nan
                    # Solo considera velocidades > 0.5 m/min (~1.6 ft/min)
                )

                # Tiempo incremental solo donde velocidad es aceptable
                df_teorico_pred["tiempo_min"] = df_teorico_pred["delta_prof"] / df_teorico_pred[
                    "vel_pred_m_min_filtrado"]

                # Suma el tiempo total
                tiempo_total_min = df_teorico_pred["tiempo_min"].sum(skipna=True)
                tiempo_total_horas = tiempo_total_min / 60.0
                tiempo_total_dias = tiempo_total_horas / 24.0

                # Muestra el resultado en negrita y azul
                st.markdown(
                    f"""<div style='background:#1565c0;color:white;padding:12px 18px;border-radius:8px;font-size:2rem;font-weight:bold;display:inline-block;'>
                    ⏱️ Tiempo estimado de RIH: {tiempo_total_horas:.2f} horas | {tiempo_total_dias:.2f} días
                    </div>""",
                    unsafe_allow_html=True
                )
###-----------------------Reporte de PDF----------------------####
import io
import base64
import tempfile
import pdfkit  # pip install pdfkit
from datetime import datetime

# Utilidad para convertir figura plotly a base64 (PNG en memoria)
def fig_to_base64(fig, width=900, height=400):
    buf = io.BytesIO()
    fig.write_image(buf, format="png", width=width, height=height)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# Utilidad para matplotlib
def mplfig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# Función para armar el HTML del reporte
def get_html_report(fig_dict, resultados_dict, resumen_dict):
    html = f"""
    <html>
    <head>
    <meta charset="utf-8">
    <style>
    body {{ font-family: Arial, sans-serif; font-size: 15px; }}
    h1,h2,h3 {{ color: #1565c0; }}
    .section {{ margin-bottom: 2em; }}
    .img-plot {{ max-width: 100%; border: 1px solid #bbb; margin-bottom: 1em; }}
    .block {{ background: #e3f2fd; padding: 0.7em 1em; border-radius: 7px; }}
    </style>
    </head>
    <body>
    <h1>Reporte de Coiled Tubing: Multi-Pozos</h1>
    <p>Fecha de generación: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
    """
    html += "<div class='section'><h2>Resumen Inicial</h2><div class='block'>"
    for k, v in resumen_dict.items():
        html += f"<b>{k}:</b> {v}<br>"
    html += "</div></div>"
    html += "<div class='section'><h2>Gráficos de Parámetros Operativos</h2>"
    for title, b64 in fig_dict["previo_ml"]:
        html += f"<h3>{title}</h3><img class='img-plot' src='data:image/png;base64,{b64}' />"
    html += "</div>"
    html += "<div class='section'><h2>Resultados de Machine Learning</h2>"
    for k, v in resultados_dict.items():
        html += f"<div class='block'><b>{k}:</b> {v}</div>"
    for title, b64 in fig_dict["ml"]:
        html += f"<h3>{title}</h3><img class='img-plot' src='data:image/png;base64,{b64}' />"
    html += "</div></body></html>"
    return html

# Cuando el usuario pulse el botón, se genera y descarga el PDF:
if st.button("📄 Generar reporte completo en PDF"):
    # Captura gráficos de antes del ML
    figs_previo_ml = [
        ("Tensión vs Profundidad", fig_to_base64(fig_tension)),
        ("Velocidad vs Profundidad", fig_to_base64(fig_vel)),
        ("Caudal vs Profundidad", fig_to_base64(fig_caudal)),
        ("Comparativo de Velocidad Promedio", fig_to_base64(fig_comp)),
        ("Tendencia de Velocidad", fig_to_base64(fig_tend)),
    ]
    # Captura gráficos del módulo ML
    figs_ml = []
    if 'fig_import' in locals():
        figs_ml.append(("Importancia de Variables", mplfig_to_base64(fig_import)))
    if 'fig_comp' in locals():
        figs_ml.append(("Comparación Real vs Predicha", fig_to_base64(fig_comp)))
    if 'fig_iew' in locals():
        figs_ml.append(("Curva IEW", fig_to_base64(fig_iew)))
    if 'fig_teo' in locals():
        figs_ml.append(("Predicción en Pozo Teórico", fig_to_base64(fig_teo)))

    resultados_dict = {
        "Score R² global": f"{r2:.3f}",
        "MSE": f"{mse:.2f}",
        "Muestras entrenamiento": f"{len(df_train)}",
        "Importancia tortuosidad": f"{importancias[2]*100:.2f}%"
    }
    resumen_dict = {
        "Pozos analizados": ", ".join(run_ids),
        "Configuración zonas": f"Rango zona = {zona_range} m",
        "Tendencia usada": tipo_tendencia,
    }
    fig_dict = {"previo_ml": figs_previo_ml, "ml": figs_ml}
    html_report = get_html_report(fig_dict, resultados_dict, resumen_dict)
    # Genera PDF usando pdfkit (debes tener wkhtmltopdf instalado en el sistema)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdfkit.from_string(html_report, tmp_pdf.name)
        tmp_pdf.seek(0)
        st.download_button(
            label="⬇️ Descargar reporte en PDF",
            data=tmp_pdf.read(),
            file_name="reporte_coiled_tubing.pdf",
            mime="application/pdf"
        )