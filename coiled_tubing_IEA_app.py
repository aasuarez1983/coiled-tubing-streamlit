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

# --- LIMPIEZA UNIVERSAL DE FORMATO DECIMAL Y TEXTOS RAROS ---
def limpiar_decimal_universal(col_serie):
    def corrige(valor):
        if pd.isnull(valor) or str(valor).strip() == '':
            return np.nan
        s = str(valor).strip().replace(" ", "")
        # Elimina cualquier caracter que no sea número, punto, coma, e, E, signo menos (excepto separador de miles y decimal)
        s = re.sub(r"[^0-9\-,\.eE]", "", s)
        # Corrige miles/decimales según el último separador
        if s.count(',') and s.count('.'):
            if s.rfind(',') > s.rfind('.'):
                s = s.replace('.', '').replace(',', '.')
            else:
                s = s.replace(',', '')
        elif s.count(','):
            # Si hay solo una coma y menos de 4 caracteres luego, es decimal: 1234,5 → 1234.5
            if len(s.split(',')[-1]) <= 3:
                s = s.replace(',', '.')
            else:
                s = s.replace(',', '')
        elif s.count('.'):
            # Si hay solo un punto y menos de 4 caracteres luego, es decimal: 1234.5 → 1234.5
            pass # ya está OK
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

# --- Carga de datos de pozos ---
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
            columnas_ct = [
                "CT-Profundidad(m)", "CT-Peso Tuberia(lb)",
                "CT-Velocidad de Viaje(ft/min)", "CT-Caudal Bombeo Liquido(bbl/min)"
            ]
            for col in columnas_ct:
                if col in df.columns:
                    df[col] = limpiar_decimal_universal(df[col])
            df["run_id"] = hoja
            if "DateTime" in df.columns:
                df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
            df["profundidad_m"] = pd.to_numeric(df["CT-Profundidad(m)"], errors='coerce').abs()
            df["Peso_lb"] = pd.to_numeric(df["CT-Peso Tuberia(lb)"], errors='coerce')
            df["velocidad_ftmin"] = pd.to_numeric(df["CT-Velocidad de Viaje(ft/min)"], errors='coerce')
            df["caudal_bblmin"] = pd.to_numeric(df["CT-Caudal Bombeo Liquido(bbl/min)"], errors='coerce')
            df_bajada = df[df["velocidad_ftmin"] > 0].copy()
            if df_bajada.empty:
                continue
            max_prof = df_bajada["profundidad_m"].max()
            idx_max = df_bajada[df_bajada["profundidad_m"] == max_prof].index[0]
            df_bajada = df_bajada.loc[:idx_max]
            df_bajada = df_bajada.drop_duplicates(subset="profundidad_m", keep="first")
            df_bajada = df_bajada.sort_values("profundidad_m")
            df_bajada["Peso_filtrada"] = gaussian_filter1d(df_bajada["Peso_lb"].fillna(0), sigma=5)
            df_bajada["velocidad_filtrada"] = gaussian_filter1d(df_bajada["velocidad_ftmin"].fillna(0), sigma=5)
            df_bajada["caudal_filtrado"] = gaussian_filter1d(df_bajada["caudal_bblmin"].fillna(0), sigma=5)
            df_total.append(df_bajada)

    # --- Carga de datos de Dogleg/Tortuosidad (opcional) ---
    dogleg_data_dict = {}
    if archivo_dogleg is not None:
        xls_dogleg = pd.ExcelFile(archivo_dogleg)
        for hoja in xls_dogleg.sheet_names:
            df_dogleg = xls_dogleg.parse(hoja, dtype=str)
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

        st.subheader("Peso vs Profundidad (selección de pozos)")
        fig_Peso = go.Figure()
        for i, run_id in enumerate(pozos_a_ver):
            grupo = df_concat[df_concat['run_id'] == run_id]
            fig_Peso.add_trace(go.Scatter(
                x=grupo["profundidad_m"], y=grupo["Peso_filtrada"],
                name=run_id,
                mode="lines",
                line=dict(color=colores[i % len(colores)]),
                hovertemplate=run_id + "<br>Profundidad: %{x:.1f} m<br>Peso: %{y:.1f} lb"
            ))
        fig_Peso.update_layout(
            xaxis=dict(title="Profundidad (m)", side="bottom"),
            xaxis2=dict(
                title="Profundidad (m)", side="top", overlaying="x", showgrid=False, showline=True, zeroline=False
            ),
            yaxis=dict(title="Peso Filtrada (lb)"),
            legend=dict(font=dict(size=8)),
            width=950, height=500,
            margin=dict(l=60, r=40, t=60, b=60),
        )
        st.plotly_chart(fig_Peso, use_container_width=True)

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
                        grupo["Peso_filtrada"].abs() * (1 + grupo["dogleg"]))
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
