import pandas as pd
import streamlit as st
import numpy as np
from pymcdm.methods import PROMETHEE_II
from pymcdm.methods import TOPSIS
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import random

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
PERIODIC_TABLE_URL = "download.webp"
ELEMENT_COLUMNS = [f"Element_{i}" for i in range(1, 8)]
FILTER_OPTIONS = [
    "Reserve (ton)", "Production (ton)", "HHI (USGS)",
    "ESG Score", "CO2 footprint max (kg/kg)",
    "Embodied energy max (MJ/kg)", "Water usage max (l/kg)",
    "Toxicity", "Max Companionability",
]
CRITERIA_OPTIONS = {
    "Reserve (ton)": 1, "Production (ton)": 1, "HHI (USGS)": -1,
    "ESG Score": -1, "CO2 footprint max (kg/kg)": -1,
    "Embodied energy max (MJ/kg)": -1, "Water usage max (l/kg)": -1,
    "Toxicity": -1, "Max Companionability": -1,
}

# ---------------------------------------------------------------------------
# STYLE  (injected once per session)
# ---------------------------------------------------------------------------
def set_custom_style():
    if st.session_state.get("_style_injected"):
        return
    st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #f8f9fa !important;
            border-right: 1px solid #e0e0e0;
        }
        [data-testid="stSidebar"] .stRadio > label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] .stMarkdown {
            color: #333333 !important;
        }
        [data-testid="stSidebar"] .stRadio > div:hover {
            background-color: #e9ecef;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)
    st.session_state["_style_injected"] = True

# ---------------------------------------------------------------------------
# DATA LOADING  (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_composition_database() -> pd.DataFrame:
    try:
        return pd.read_parquet("7_materials_properties.parquet")
    except:
        df = pd.read_excel("7_materials_properties.xlsx")
        df.to_parquet("7_materials_properties.parquet")
    return df

# ---------------------------------------------------------------------------
# DATE PARSING  (cached — avoids re-parsing on every rerender)
# ---------------------------------------------------------------------------
@st.cache_data
def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

# ---------------------------------------------------------------------------
# ELEMENT HELPERS  (module-level, cached)
# ---------------------------------------------------------------------------
@st.cache_data
def get_all_elements_df1(df: pd.DataFrame) -> list:
    """Return sorted list of all unique elements appearing in Element_1..7."""
    elements: set = set()
    for col in ELEMENT_COLUMNS:
        if col in df.columns:
            elements.update(df[col].dropna().unique())
    return sorted(e for e in elements if e and str(e).strip())


def filter_df1_by_included_elements(
    df: pd.DataFrame, included_list: list, element_cols: list
) -> pd.DataFrame:
    """Keep rows whose element slots contain only elements in included_list."""
    if not included_list:
        return df.iloc[0:0]
    if not element_cols:
        return df

    included_set = set(included_list)

    def _is_empty(x):
        return pd.isna(x) or str(x).strip() == ""

    mask = pd.Series(True, index=df.index)
    for c in element_cols:
        if c in df.columns:
            mask &= df[c].isin(included_set) | df[c].apply(_is_empty)
    return df[mask]


def filter_by_excluded_elements(df: pd.DataFrame, excluded_elements: list) -> pd.DataFrame:
    """Remove rows that contain any of the excluded elements."""
    if not excluded_elements:
        return df

    excluded_set = {str(e).strip() for e in excluded_elements}
    valid_cols = [c for c in ELEMENT_COLUMNS if c in df.columns]

    mask = pd.concat(
        [df[c].fillna("").astype(str).str.strip().isin(excluded_set) for c in valid_cols],
        axis=1,
    ).any(axis=1)
    return df[~mask]


# ---------------------------------------------------------------------------
# GENERAL FILTER
# ---------------------------------------------------------------------------
def filter_dataframe(
    df: pd.DataFrame, filters: dict, selected_names=None
) -> pd.DataFrame:
    """Filter df by numeric ranges and optional name list."""
    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in filters.items():
        if col in df.columns:
            mask &= df[col].between(lo, hi, inclusive="both")

    if selected_names is not None:
        mask &= df["Name"].isin(selected_names)

    return df[mask]


# ---------------------------------------------------------------------------
# MCDM RUNNERS
# ---------------------------------------------------------------------------
def run_topsis(matrix, weights, criteria_types):
    return TOPSIS()(matrix, weights, criteria_types)

def run_promethee(matrix, weights, criteria_types):
    return PROMETHEE_II("usual")(matrix, weights, criteria_types)

# ---------------------------------------------------------------------------
# PLOT HELPERS  (module-level)
# ---------------------------------------------------------------------------
def pick_palette(n: int) -> list:
    colors = px.colors.qualitative.Set3 if n <= 12 else px.colors.qualitative.Light24
    return colors[:n]


@st.cache_data
def prepare_plot_data(
    df: pd.DataFrame, x_col: str, y_col: str,
    log_x: bool = False, log_y: bool = False,
) -> pd.DataFrame:
    df_plot = df.copy()
    if log_x:
        df_plot[x_col] = np.log10(df_plot[x_col].clip(lower=1e-10))
    if log_y:
        df_plot[y_col] = np.log10(df_plot[y_col].clip(lower=1e-10))
    return df_plot


def create_professional_plot(
    df: pd.DataFrame, x_col: str, y_col: str,
    title: str, x_label: str, y_label: str,
    log_x: bool = False, log_y: bool = False,
) -> go.Figure:
    """Create a professional Plotly scatter plot."""
    primary_color = "#3498db"
    highlight_color = "#c5301f"

    df_plot = df.copy()
    if log_x:
        df_plot[x_col] = df_plot[x_col].clip(lower=1e-10)
    if log_y:
        df_plot[y_col] = df_plot[y_col].clip(lower=1e-10)

    num_highlight = min(10, len(df_plot))
    highlight_indices = np.random.choice(len(df_plot), num_highlight, replace=False)
    df_plot["is_highlight"] = False
    df_plot.iloc[highlight_indices, df_plot.columns.get_loc("is_highlight")] = True

    fig = go.Figure()

    df_regular = df_plot[~df_plot["is_highlight"]]
    fig.add_trace(go.Scatter(
        x=df_regular[x_col], y=df_regular[y_col],
        mode="markers", name="All Materials",
        marker=dict(size=8, color=primary_color, opacity=0.6, line=dict(width=0)),
        text=df_regular["Name"],
        hovertemplate=f"<b>%{{text}}</b><br>{x_label}: %{{x}}<br>{y_label}: %{{y}}<extra></extra>",
    ))

    df_highlight = df_plot[df_plot["is_highlight"]]
    fig.add_trace(go.Scatter(
        x=df_highlight[x_col], y=df_highlight[y_col],
        mode="markers+text", name="Highlighted Materials",
        marker=dict(size=12, color=highlight_color, opacity=1.0, line=dict(width=0)),
        text=df_highlight["Name"],
        textposition="top center",
        textfont=dict(color=highlight_color, size=10),
        hovertemplate=f"<b>%{{text}}</b><br>{x_label}: %{{x}}<br>{y_label}: %{{y}}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis_title=f"{'log(' + x_label + ')' if log_x else x_label}",
        yaxis_title=f"{'log(' + y_label + ')' if log_y else y_label}",
        xaxis_type="log" if log_x else "linear",
        yaxis_type="log" if log_y else "linear",
        hovermode="closest",
        template="plotly_white",
        width=900, height=550,
        showlegend=True,
        legend=dict(x=0.99, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray",
                            tickfont=dict(color="black"), title_font=dict(color="black"))
    fig.update_yaxes(showgrid=False,
                            tickfont=dict(color="black"), title_font=dict(color="black"))
    return fig


# ---------------------------------------------------------------------------
# UTILITY HELPERS  (module-level)
# ---------------------------------------------------------------------------
def format_tons(value: float) -> str:
    if value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.1f}TB tons"
    elif value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B tons"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M tons"
    elif value >= 1_000:
        return f"{value / 1_000:.1f}K tons"
    return f"{value:.0f} tons"


@st.cache_data
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Cached CSV serialisation — avoids re-encoding on every rerender."""
    return df.to_csv(index=False).encode("utf-8")


@st.cache_data
def create_full_output(
    filtered_df: pd.DataFrame,
    results_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    filter_items: tuple,          # serialisable replacement for session_state
) -> bytes:
    """Create Excel output with all MCDM analysis results."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        full_data = filtered_df.copy()
        results_reset = results_df.reset_index()

        if "Score" in results_reset.columns:
            score_map = dict(zip(results_reset["Material"], results_reset["Score"]))
            rank_map  = dict(zip(results_reset["Material"], results_reset["Rank"]))
            full_data["TOPSIS_Score"] = full_data["Name"].map(score_map)
            full_data["TOPSIS_Rank"]  = full_data["Name"].map(rank_map)
        else:
            flow_map = dict(zip(results_reset["Material"], results_reset["Net Flow"]))
            rank_map = dict(zip(results_reset["Material"], results_reset["Rank"]))
            full_data["PROMETHEE_Net_Flow"] = full_data["Name"].map(flow_map)
            full_data["PROMETHEE_Rank"]     = full_data["Name"].map(rank_map)

        full_data.to_excel(writer, sheet_name="Full Data", index=False)
        results_reset.to_excel(writer, sheet_name="Rankings", index=False)
        weights_df.reset_index().to_excel(writer, sheet_name="Weights", index=False)

        if filter_items:
            filter_settings = pd.DataFrame(
                [{"Filter": k, "Min": v[0], "Max": v[1]} for k, v in filter_items]
            )
            filter_settings.to_excel(writer, sheet_name="Filter Settings", index=False)

    return output.getvalue()


# ---------------------------------------------------------------------------
# PAGE: HOME
# ---------------------------------------------------------------------------
def page_home(df1: pd.DataFrame) -> None:
    st.title("Semiconductor Database")

    cols = st.columns(2)
    with cols[0]:
        st.markdown("""
        ### 🔍 About This Tool
        This interactive platform enables comprehensive analysis of environmental impacts
        and sustainability of semiconductors with:
        - **Extensive database** on ESG scores, CO₂ footprints, and more
        - **Visualizations** to explore relationships between parameters
        - **Multi-criteria** decision making tools (TOPSIS, PROMETHEE)
        - **Export capabilities** for further analysis
        """)
    with cols[1]:
        st.markdown("""
        ### 🚀 Getting Started
        1. Select an analysis page from the sidebar
        2. Configure your filters and parameters
        3. Visualize the relationships
        4. Download results for further use

        **Pro Tip:** Use the MCDM analysis for ranking the most promising semiconductors.
        """)

    st.markdown("---")
    st.markdown("### 📚 Database Information")
    cols = st.columns(2)
    with cols[0]:
        st.metric("Total Materials", len(df1))
        prod_min = df1["Production (ton)"].min()
        prod_max = df1["Production (ton)"].max()
        st.metric("Production Range", f"{format_tons(prod_min)} - {format_tons(prod_max)}")
    with cols[1]:
        st.metric("Bandgap Range", f"{df1['Bandgap'].min():.1f} - {df1['Bandgap'].max():.1f} eV")


# ---------------------------------------------------------------------------
# PAGE: BANDGAP INFORMATION
# ---------------------------------------------------------------------------
def page_bandgap(df1: pd.DataFrame) -> None:
    st.title("Bandgap Information")
    st.markdown("Most commonly researched semiconductors and their band gap range.")

    # ---- session state defaults ----
    if "included_elements" not in st.session_state:
        st.session_state.included_elements = []
    if "filters_applied" not in st.session_state:
        st.session_state.filters_applied = False

    # ---- element selection ----
    st.markdown("### Element Inclusion")
    st.image(PERIODIC_TABLE_URL, caption="Periodic Table of Elements", use_container_width=True)

    if st.session_state.included_elements:
        df1_filtered = filter_df1_by_included_elements(
            df1, st.session_state.included_elements, ELEMENT_COLUMNS
        )
        st.info(
            f"🔬 Element filter active: Included "
            f"{', '.join(sorted(st.session_state.included_elements))} | "
            f"Showing {len(df1_filtered)} of {len(df1)} materials"
        )
    else:
        df1_filtered = df1.iloc[0:0]   # empty until elements chosen

    st.markdown("**Enter element symbols to include (separated by commas)**")
    element_text_input = st.text_input(
        "Element symbols:",
        value=", ".join(st.session_state.included_elements) if st.session_state.included_elements else "",
        key="element_text_input",
        placeholder="e.g., Ti, O, Zn",
        help="Enter element symbols separated by commas.",
    )

    if element_text_input.strip():
        selected_elements = [e.strip() for e in element_text_input.split(",") if e.strip()]
    else:
        selected_elements = []

    if st.session_state.included_elements:
        st.markdown(f"**Currently Included Elements:** {len(st.session_state.included_elements)}")

    # Preview only when selection changed
    if selected_elements != st.session_state.included_elements:
        preview = filter_df1_by_included_elements(df1, selected_elements, ELEMENT_COLUMNS)
        if len(preview) > 0:
            st.success(f"✅ Preview: {len(preview)} materials ({len(preview)/len(df1)*100:.1f}%)")
        else:
            st.warning("⚠️ Preview: No materials contain only these elements.")

    st.session_state.included_elements = selected_elements

    colA, _ = st.columns([1, 3])
    with colA:
        if st.button("Apply Filters", key="apply_initial_filters"):
            st.session_state.filters_applied = True
            st.rerun()

    # Reuse already-computed df1_filtered (no extra call)
    df_filtered = filter_df1_by_included_elements(
        df1, st.session_state.included_elements, ELEMENT_COLUMNS
    )

    bandgap_col = next(
        (c for c in ["Bandgap", "bandgap", "Band_gap", "band_gap", "Value", "value", "BandGap"]
         if c in df_filtered.columns),
        None,
    )

    # ---- aggregation ----
    df_agg = (
        df_filtered.groupby("Name")
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
        .head(9)
    )
    st.session_state.df_agg = df_agg
    top_names = df_agg["Name"].head(9).tolist()

    # ---- Bandgap scatter ----
    st.markdown("""
    <h3 style='font-size:20px;font-weight:600;color:#222;margin-top:10px;
               margin-bottom:20px;font-family:Arial;'>
    Exploratory Data Analysis of Database
    </h3>""", unsafe_allow_html=True)
    st.markdown("***The scatterplot provides a broad visual overview of the relative bandgap ranges across the materials.***")

    if bandgap_col is None or not top_names:
        st.warning("⚠️ No bandgap column found or no top names available.")
    else:
        filtered_top = df_filtered[df_filtered["Name"].isin(top_names)]
        if filtered_top.empty:
            st.info("No data to display for the current selection of elements.")
        else:
            fig = px.scatter(
                filtered_top, x=bandgap_col, y="Name", color="Name",
                color_discrete_sequence=px.colors.qualitative.Bold,
                title="Bandgap Distribution by Semiconductor",
                labels={bandgap_col: "Bandgap (eV)", "Name": "Semiconductor"},
                height=500, hover_data={bandgap_col: ":.2f"},
            )
            fig.update_traces(marker=dict(size=10, opacity=0.9))
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray",
                            tickfont=dict(color="black"), title_font=dict(color="black"))
            fig.update_yaxes(showgrid=False,
                            tickfont=dict(color="black"), title_font=dict(color="black"))
            fig.update_layout(template="plotly_white", hovermode="closest")
            st.plotly_chart(fig, use_container_width=True, key="bandgap_scatter")

    # ---- Histogram grid ----
    st.markdown("***Histogram plot shows the frequency distribution of bandgaps.***")

    if bandgap_col is None:
        st.warning("⚠️ No bandgap column found.")
    else:
        df_hist = df_filtered[df_filtered["Name"].isin(top_names)].copy()
        df_hist = df_hist[pd.to_numeric(df_hist[bandgap_col], errors="coerce").notna()]
        df_hist[bandgap_col] = df_hist[bandgap_col].astype(float)

        if df_hist.empty:
            st.info("No data to plot after filtering.")
        else:
            x_min = df_hist[bandgap_col].min()
            top9 = top_names[:9]
            fig_hist, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 10), sharex=True)
            axes = axes.flatten()

            for i, name in enumerate(top9):
                ax = axes[i]
                sub = df_hist.loc[df_hist["Name"] == name, bandgap_col].dropna().values
                if sub.size >= 1:
                    ax.hist(sub, bins="auto", density=False, alpha=0.85)
                    if sub.size >= 2:
                        ax.axvline(np.median(sub), linestyle="--", linewidth=1)
                    if sub.size >= 50:
                        ax.set_yscale("log")
                else:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, alpha=0.7)
                ax.set_title(f"{name} (n={sub.size})", fontsize=11)
                ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
                ax.set_xlim(x_min, 10)
                if i % 3 != 0:
                    ax.set_ylabel("")
                if i < 6:
                    ax.set_xlabel("")

            for j in range(len(top9), 9):
                axes[j].axis("off")

            fig_hist.supylabel("Count (log scale)")
            fig_hist.supxlabel("Bandgap (eV)")
            fig_hist.tight_layout(rect=[0, 0.02, 1, 0.95])
            st.pyplot(fig_hist, clear_figure=True)
            plt.close(fig_hist)   # ← explicit close to free memory

    # ---- Temporal scatter ----
    st.markdown("***The scatter plot visualizes the trends of recently researched materials and their corresponding bandgap values.***")

    unique_list = df_agg["Name"].dropna().unique().tolist()

    # Derive the Elsevier publication subset directly from df1
    df_elsevier = df1[
        (df1["Publisher"].str.strip().str.lower() == "elsevier") &
        df1["Date"].notna()
    ].copy()

    if df_elsevier.empty or "Name" not in df_elsevier.columns:
        st.warning("No Elsevier publication records with dates found.")
    else:
        df2_dated = parse_dates(df_elsevier)

        df2_doi = (
            df2_dated[df2_dated["Name"].isin(unique_list)]
            [["Name", "Bandgap", "DOI", "Date"]]
        )
        df2_filtered = (
            df2_doi.groupby(["Date", "Name", "Bandgap"])
            .size()
            .reset_index(name="Frequency")
            .reset_index()
        )

        required_cols = {"Date", "Bandgap", "Frequency", "Name"}
        missing_cols = required_cols - set(df2_filtered.columns)
        if missing_cols:
            st.warning(f"Missing required columns: {', '.join(sorted(missing_cols))}")
            st.stop()

        if df2_filtered.empty:
            st.info("No materials found for the selected names.")
        else:
            df2_filtered["Date"]      = pd.to_datetime(df2_filtered["Date"],     errors="coerce")
            df2_filtered["Bandgap"]   = pd.to_numeric(df2_filtered["Bandgap"],   errors="coerce")
            df2_filtered["Frequency"] = pd.to_numeric(df2_filtered["Frequency"], errors="coerce")
            df2_filtered = df2_filtered.dropna(subset=["Date", "Bandgap", "Frequency", "Name"])

            df2_plot = df2_filtered.sort_values("Date")

            y_max = df2_plot["Bandgap"].max()
            fig_scatter = go.Figure()

            # Background band shading — increased opacity for contrast
            for y0, y1, color, label in [
                (0,    1.6,    "rgba(255,200,0,0.18)",   "Infrared (0–1.6 eV)"),
                (1.6,  3.26,   "rgba(0,180,0,0.18)",     "Visible (1.6–3.26 eV)"),
                (3.26, y_max,  "rgba(220,0,0,0.18)",     "Ultraviolet (3.26+ eV)"),
            ]:
                fig_scatter.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0,
                                      annotation_text=label, annotation_position="top left",
                                      annotation_font_size=11)

            # High-contrast qualitative palette
            contrast_palette = px.colors.qualitative.Bold
            n_groups = df2_plot["Name"].nunique()
            palette = (contrast_palette * ((n_groups // len(contrast_palette)) + 1))[:n_groups]

            for idx, g in enumerate(df2_plot["Name"].unique()):
                gdata = df2_plot[df2_plot["Name"] == g]
                fig_scatter.add_trace(go.Scatter(
                    x=gdata["Date"], y=gdata["Bandgap"],
                    mode="markers", name=g,
                    marker=dict(
                        size=(gdata["Frequency"].clip(lower=1) * 5).clip(upper=35),
                        color=palette[idx],
                        opacity=0.92,
                        line=dict(width=1, color="white"),
                        symbol="circle",
                    ),
                    hovertemplate="<b>%{text}</b><br>Date: %{x}<br>Bandgap: %{y:.2f} eV<extra></extra>",
                    text=gdata["Name"],
                ))

            fig_scatter.update_layout(
                xaxis_title="Publication Date",
                yaxis_title="Bandgap Energy (eV)",
                template="plotly_white",
                hovermode="closest",
                legend=dict(title="Material", orientation="v", x=1.01, y=1),
                height=520,
                xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
                paper_bgcolor="rgba(255,255,255,1)",
                plot_bgcolor="rgba(255,255,255,1)",
            )
            fig_scatter.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray",
                            tickfont=dict(color="black"), title_font=dict(color="black"))
            fig_scatter.update_yaxes(showgrid=False,
                            tickfont=dict(color="black"), title_font=dict(color="black"))
            st.plotly_chart(fig_scatter, use_container_width=True, key="temporal_scatter")

            st.markdown("***The table displays ten(10) sampled journals relating to the filtered semiconductors.***")

            n = min(10, len(df2_doi))
            if "sample_seed" not in st.session_state:
                st.session_state.sample_seed = 42
            if st.button("🔀 Shuffle sample"):
                st.session_state.sample_seed = random.randint(0, 10**9)

            sample = df2_doi.sample(n=n, random_state=st.session_state.sample_seed).copy()
            sample["Date"] = sample["Date"].dt.strftime("%m/%Y")
            st.dataframe(sample, use_container_width=True)

            csv_bytes = to_csv_bytes(df2_doi)
            st.download_button(
                label="⬇️ Download excel file as CSV",
                data=csv_bytes,
                file_name="bandgap-filtered.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------------------
# PAGE: DECISION-MAKING ASSISTANT
# ---------------------------------------------------------------------------
@st.fragment
def page_decision(df1: pd.DataFrame) -> None:
    st.title("Decision-making Assistant")
    st.markdown("Facilitate semiconductor selection with advanced filtering and visualization")

    # ---- session state defaults ----
    for key, default in [
        ("filters", {}),
        ("initial_filter_name", None),
        ("initial_filters_only", {}),
        ("plot_x_col", "Bandgap"),
        ("plot_y_col", "Reserve (ton)"),
        ("excluded_elements", []),
        ("additional_dynamic_filters", []),
        ("filters_applied", False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ---- element exclusion UI ----
    st.markdown("### 1. Element Exclusion")
    st.image(PERIODIC_TABLE_URL, caption="Periodic Table of Elements", use_container_width=True)

    element_text_input = st.text_input(
        "Element symbols:",
        value=", ".join(st.session_state.excluded_elements) if st.session_state.excluded_elements else "",
        key="element_text_input",
        placeholder="e.g., Au, Ag, Si, Pb",
        help="Enter element symbols separated by commas.",
    )

    selected_elements = (
        [e.strip() for e in element_text_input.split(",") if e.strip()]
        if element_text_input.strip() else []
    )

    if st.session_state.excluded_elements:
        st.markdown(f"**Currently Excluded Elements:** {len(st.session_state.excluded_elements)}")

    # Compute exclusion once; reuse for all downstream consumers
    df_after_exclusion = filter_by_excluded_elements(df1, st.session_state.excluded_elements)

    if st.session_state.excluded_elements:
        removed_count = len(df1) - len(df_after_exclusion)
        st.info(
            f"🔬 Element filter active: Excluded "
            f"{', '.join(sorted(st.session_state.excluded_elements))} | "
            f"Removed {removed_count} materials | Showing {len(df_after_exclusion)} of {len(df1)} materials"
        )

    # Preview (only when selection differs from stored state)
    if selected_elements != st.session_state.excluded_elements:
        preview_filtered = filter_by_excluded_elements(df1, selected_elements)
        would_remove = len(df1) - len(preview_filtered)
        if would_remove > 0:
            st.warning(
                f"⚠️ Preview: This will remove {would_remove} materials "
                f"({would_remove / len(df1) * 100:.1f}%) from the dataset."
            )

    # ---- initial filters ----
    st.markdown("### 2. Initial Filters")
    cols = st.columns(2)

    with cols[0]:
        st.markdown("#### Bandgap Selection")
        c1, c2 = st.columns(2)
        with c1:
            bandgap_min = st.number_input("Min (eV)", min_value=0.0, max_value=35.0,
                                          value=0.0, step=0.1, key="bandgap_min")
        with c2:
            bandgap_max = st.number_input("Max (eV)", min_value=0.0, max_value=35.0,
                                          value=3.0, step=0.1, key="bandgap_max")
        if bandgap_min > bandgap_max:
            st.error("Minimum bandgap must be less than or equal to maximum bandgap")
        bandgap_range = (bandgap_min, bandgap_max)

    with cols[1]:
        st.markdown("#### Additional Filter")
        selected_filter = st.selectbox("Choose a filter", FILTER_OPTIONS, key="selected_filter")

        if selected_filter:
            # Reuse already-computed df_after_exclusion for min/max bounds
            temp_filtered = df_after_exclusion
            filter_min = float(temp_filtered[selected_filter].min())
            filter_max = float(temp_filtered[selected_filter].max())

            if selected_filter in ("Production (ton)", "Reserve (ton)"):
                filter_min_input = st.number_input(
                    "Minimum Requirement (tonnes)",
                    min_value=filter_min, max_value=filter_max, value=filter_min,
                    step=1000.0, format="%.2f", key="filter_min_input",
                )
                filter_range = (filter_min_input, filter_max)
                st.caption(f"**Minimum Required:** {format_tons(filter_min_input)}")
            elif selected_filter == "Toxicity":
                filter_range = st.slider(
                    f"{selected_filter} Range",
                    int(filter_min), int(filter_max),
                    (int(filter_min), int(filter_max)),
                    step=1, key="initial_filter_slider",
                )
            else:
                filter_range = st.slider(
                    f"{selected_filter} Range",
                    filter_min, filter_max, (filter_min, filter_max),
                    key="initial_filter_slider",
                )
        else:
            filter_range = None

    # ---- dynamic extra filters ----
    st.markdown("### 3. Additional Filters (Optional)")
    available_for_dynamic = [f for f in FILTER_OPTIONS if f != selected_filter]

    col_add, col_info = st.columns([1, 3])
    with col_add:
        if st.button("➕ Add Filter", key="add_dynamic_filter"):
            if len(st.session_state.additional_dynamic_filters) < len(available_for_dynamic):
                st.session_state.additional_dynamic_filters.append(
                    {"filter_name": None, "filter_range": None}
                )
                st.rerun()
    with col_info:
        st.caption(f"You can add up to {len(available_for_dynamic)} additional filters")

    dynamic_filter_values: dict = {}
    filters_to_remove: list = []

    for idx, filter_config in enumerate(st.session_state.additional_dynamic_filters):
        st.markdown(f"#### Filter #{idx + 2}")
        col1, col2, col3 = st.columns([2, 3, 1])

        with col1:
            used_filters = (
                [selected_filter]
                + [f["filter_name"] for f in st.session_state.additional_dynamic_filters if f["filter_name"]]
            )
            available_options = [
                f for f in available_for_dynamic
                if f not in used_filters or f == filter_config.get("filter_name")
            ]
            if available_options:
                dynamic_filter_name = st.selectbox(
                    "Select filter", options=available_options,
                    index=available_options.index(filter_config["filter_name"])
                          if filter_config.get("filter_name") in available_options else 0,
                    key=f"dynamic_filter_name_{idx}",
                )
                filter_config["filter_name"] = dynamic_filter_name
            else:
                st.warning("No more filters available")
                dynamic_filter_name = None

        with col2:
            if dynamic_filter_name:
                # Reuse already-computed df_after_exclusion for min/max bounds
                dyn_min = float(df_after_exclusion[dynamic_filter_name].min())
                dyn_max = float(df_after_exclusion[dynamic_filter_name].max())

                if dynamic_filter_name == "Toxicity":
                    dyn_range = st.slider(
                        f"{dynamic_filter_name} Range",
                        int(dyn_min), int(dyn_max), (int(dyn_min), int(dyn_max)),
                        step=1, key=f"dynamic_filter_range_{idx}",
                    )
                elif dynamic_filter_name in ("Production (ton)", "Reserve (ton)"):
                    dyn_range = st.slider(
                        f"{dynamic_filter_name} Range",
                        dyn_min, dyn_max, (dyn_min, dyn_max),
                        format="", key=f"dynamic_filter_range_{idx}",
                    )
                    st.caption(f"**Range:** {format_tons(dyn_range[0])} to {format_tons(dyn_range[1])}")
                else:
                    dyn_range = st.slider(
                        f"{dynamic_filter_name} Range",
                        dyn_min, dyn_max, (dyn_min, dyn_max),
                        key=f"dynamic_filter_range_{idx}",
                    )
                dynamic_filter_values[dynamic_filter_name] = dyn_range

        with col3:
            if st.button("🗑️", key=f"remove_filter_{idx}", help="Remove this filter"):
                filters_to_remove.append(idx)

    if filters_to_remove:
        for idx in sorted(filters_to_remove, reverse=True):
            st.session_state.additional_dynamic_filters.pop(idx)
        st.rerun()

    # ---- apply button ----
    if st.button("Apply Filters", key="apply_all_filters", type="primary"):
        if filter_range is not None:
            st.session_state.excluded_elements = selected_elements
            all_filters = {"Bandgap": bandgap_range, selected_filter: filter_range}
            all_filters.update(dynamic_filter_values)
            st.session_state.filters = all_filters
            st.session_state.initial_filters_only = all_filters.copy()
            st.session_state.initial_filter_name = selected_filter
            st.session_state.plot_x_col = "Bandgap"
            st.session_state.plot_y_col = selected_filter
            st.session_state.filters_applied = True
            st.success(f"✅ {len(all_filters)} filter(s) applied successfully!")
            st.rerun()
        else:
            st.warning("Please select an additional filter and set its range.")

    # ---- filtered results ----
    st.subheader("Filtered Results")

    if st.session_state.initial_filters_only:
        df_filtered = filter_dataframe(df_after_exclusion, st.session_state.initial_filters_only)
        filter_summary = ", ".join(st.session_state.initial_filters_only.keys())
        st.info(
            f"📊 Showing {len(df_filtered)} materials | Filters applied: {filter_summary} | "
            f"Available: {len(df_after_exclusion)} (after element exclusion)"
        )
    else:
        df_filtered = df_after_exclusion
        st.info(f"📈 Showing all {len(df_after_exclusion)} available materials (after element exclusion)")

    x_col = st.session_state.plot_x_col
    y_col = st.session_state.plot_y_col

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        st.write(f"**X-axis:** {x_col}")
    with c2:
        st.write(f"**Y-axis:** {y_col}")
    with c3:
        log_y = st.checkbox("Log Y-axis", key="log_y_main")

    if not df_filtered.empty:
        fig_main = create_professional_plot(
            df_filtered, x_col, y_col, f"{x_col} vs {y_col}", x_col, y_col, False, log_y
        )
        st.plotly_chart(fig_main, use_container_width=True, key="decision_making_scatter_plot")
    else:
        st.warning("⚠️ No materials match the current filters")

    # ---- MCDM section ----
    if not st.session_state.filters_applied or df_filtered.empty:
        return

    st.markdown("---")
    st.subheader("4. Multi-Criteria Decision Making")
    st.info(f"Analyze the {len(df_filtered)} filtered materials using TOPSIS or PROMETHEE methods")

    cols_mcdm = st.columns(2)
    with cols_mcdm[0]:
        mcdm_method = st.selectbox(
            "Method", ["TOPSIS", "PROMETHEE"],
            help=(
                "TOPSIS: Technique for Order Preference by Similarity to Ideal Solution\n"
                "PROMETHEE: Preference Ranking Organization Method for Enrichment Evaluation"
            ),
            key="mcdm_method_custom",
        )
    with cols_mcdm[1]:
        weighting_method = st.radio(
            "Weighting", ["Entropy Weighting", "Manual Weights"],
            horizontal=True, key="mcdm_weighting_custom",
        )

    available_criteria = {k: v for k, v in CRITERIA_OPTIONS.items() if k in df_filtered.columns}

    if not available_criteria:
        st.error("❌ No criteria columns found in filtered data.")
        st.stop()

    # ---- drop N/A rows once — shared by entropy weights and run button ----
    crit_cols = list(available_criteria.keys())
    nan_rows  = df_filtered[crit_cols].isna().any(axis=1)
    n_dropped = int(nan_rows.sum())
    df_mcdm   = df_filtered[~nan_rows].copy()

    if n_dropped > 0:
        st.warning(f"⚠️ {n_dropped} material(s) with N/A in selected criteria excluded from MCDM.")

    if df_mcdm.empty:
        st.error("❌ No materials remain after removing N/A criteria rows.")
        st.stop()

    # ---- weight calculation ----
    weights = None

    if weighting_method == "Entropy Weighting":
        if len(df_mcdm) < 20:
            st.warning(f"⚠️ Warning: Only {len(df_mcdm)} materials available.")

        try:
            matrix_for_entropy = df_mcdm[crit_cols].values

            if np.isnan(matrix_for_entropy).any():
                st.error(f"❌ Cannot calculate entropy weights: {np.isnan(matrix_for_entropy).sum()} missing values found.")
            elif np.any(matrix_for_entropy < 0):
                st.error("❌ Cannot calculate entropy weights: Negative values found.")
            else:
                n_rows, m_cols = matrix_for_entropy.shape
                col_sums = matrix_for_entropy.sum(axis=0)
                prob_matrix = np.where(
                    col_sums > 1e-10,
                    matrix_for_entropy / col_sums,
                    1.0 / n_rows,
                )

                p_safe = np.where(prob_matrix > 1e-10, prob_matrix, 1e-10)
                entropy = -np.sum(p_safe * np.log(p_safe), axis=0) / np.log(n_rows)
                diversities = 1 - entropy

                diversity_sum = diversities.sum()
                weights = diversities / diversity_sum if diversity_sum > 1e-10 else np.ones(m_cols) / m_cols

                if np.isnan(weights).any() or np.isinf(weights).any():
                    weights = np.ones(len(available_criteria)) / len(available_criteria)
                    st.success(f"✅ Using equal weights: {1/len(available_criteria):.2%}")
                else:
                    st.success("✅ Entropy weights calculated successfully")

        except Exception as e:
            st.error(f"❌ Error calculating entropy weights: {e}")
            weights = np.ones(len(available_criteria)) / len(available_criteria)
            st.success(f"✅ Using equal weights: {1/len(available_criteria):.2%}")

    else:  # Manual Weights
        st.markdown("**📊 Criteria Weights** - Assign importance (0–5 scale):")

        if "preset_weights" not in st.session_state:
            st.session_state.preset_weights = {col: 3 for col in available_criteria}

        preset_cols = st.columns(3)
        with preset_cols[0]:
            if st.button("Balanced", key="preset_balanced"):
                st.session_state.preset_weights = {col: 3 for col in available_criteria}
                st.rerun()
        with preset_cols[1]:
            if st.button("Long-term goal", key="preset_long_term"):
                st.session_state.preset_weights = {
                    col: 5 if col in ("ESG Score", "Toxicity", "Max Companionability", "Reserve (ton)") else 1
                    for col in available_criteria
                }
                st.rerun()
        with preset_cols[2]:
            if st.button("Short-term goal", key="preset_short_term"):
                st.session_state.preset_weights = {
                    col: 5 if col in ("Production (ton)", "HHI (USGS)", "CO2 footprint max (kg/kg)",
                                      "Water usage max (l/kg)", "Embodied energy max (MJ/kg)") else 1
                    for col in available_criteria
                }
                st.rerun()

        st.markdown("##### Adjust Individual Weights")
        raw_weights = []
        criteria_list = list(available_criteria.items())
        mid_point = (len(criteria_list) + 1) // 2

        cols_row1 = st.columns(mid_point)
        for i, (col, _) in enumerate(criteria_list[:mid_point]):
            with cols_row1[i]:
                w = st.slider(col, 0, 5,
                              value=st.session_state.preset_weights.get(col, 3),
                              key=f"weight_custom_{col}")
                raw_weights.append(w)

        if len(criteria_list) > mid_point:
            cols_row2 = st.columns(len(criteria_list) - mid_point)
            for i, (col, _) in enumerate(criteria_list[mid_point:]):
                with cols_row2[i]:
                    w = st.slider(col, 0, 5,
                                  value=st.session_state.preset_weights.get(col, 3),
                                  key=f"weight_custom_{col}")
                    raw_weights.append(w)

        if sum(raw_weights) == 0:
            st.warning("All weights set to 0 - using equal weights")
            weights = np.ones(len(raw_weights)) / len(raw_weights)
        else:
            weights = np.array(raw_weights) / sum(raw_weights)

    # ---- weights table ----
    weights_df = pd.DataFrame({
        "Criterion": list(available_criteria.keys()),
        "Weight": weights if weights is not None else np.zeros(len(available_criteria)),
        "Direction": ["Maximize" if d == 1 else "Minimize" for d in available_criteria.values()],
    }).sort_values("Weight", ascending=False).reset_index(drop=True)
    weights_df.index     = weights_df.index + 1
    weights_df.index.name = "Rank"

    st.subheader("Criteria Weights")
    if weights is None:
        st.error("❌ Error: Weights are None.")
    elif len(weights) == 0:
        st.error("❌ Error: No weights calculated.")
    elif np.isnan(weights).any():
        st.error("❌ Error: Some weights are NaN.")
        st.dataframe(weights_df)
    else:
        st.dataframe(
            weights_df.style.format({"Weight": "{:.2%}"}),
            use_container_width=True,
        )

    # ---- run analysis ----
    if st.button("🚀 Run MCDM Analysis", type="primary", key="run_mcdm_custom"):
        with st.spinner("Performing analysis..."):
            matrix = df_mcdm[crit_cols].values
            types  = np.array(list(available_criteria.values()))

            if np.isnan(matrix).any():
                st.error(f"❌ Error: Found {np.isnan(matrix).sum()} missing values.")
                st.stop()
            if weights is None or len(weights) == 0:
                st.error("❌ Error: Weights are not defined.")
                st.stop()
            if np.isnan(weights).any():
                st.error("❌ Error: Weights contain NaN values.")
                st.stop()
            if not np.isclose(np.sum(weights), 1.0):
                st.warning("⚠️ Normalizing weights to 1.0")
                weights = weights / np.sum(weights)

            try:
                if mcdm_method == "TOPSIS":
                    scores = run_topsis(matrix, weights, types)
                    if np.isnan(scores).any():
                        st.error("❌ TOPSIS returned NaN scores.")
                        st.stop()
                    results = pd.DataFrame({
                        "Material":     df_mcdm["Name"].values,
                        "Bandgap (eV)": df_mcdm["Bandgap"].values,
                        "DOI":          df_mcdm["DOI"].values,
                        "Score":        scores,
                    }).sort_values("Score", ascending=False).reset_index(drop=True)
                else:
                    flows = run_promethee(matrix, weights, types)
                    if np.isnan(flows).any():
                        st.error("❌ PROMETHEE returned NaN flows.")
                        st.stop()
                    results = pd.DataFrame({
                        "Material":     df_mcdm["Name"].values,
                        "Bandgap (eV)": df_mcdm["Bandgap"].values,
                        "Net Flow":     flows,
                    }).sort_values("Net Flow", ascending=False).reset_index(drop=True)

                results.index      = results.index + 1
                results.index.name = "Rank"

            except Exception as e:
                st.error(f"❌ Error running {mcdm_method}: {e}")
                st.stop()

        # ---- display results ----
        st.subheader("MCDM Results")
        top_n = 100
        st.write(f"Showing top {top_n} results (out of {len(results)} total)")

        score_col    = "Score" if "Score" in results.columns else "Net Flow"
        display_cols = ["Material", "Bandgap (eV)", "DOI", score_col]
        format_dict  = {"Bandgap (eV)": "{:.2f}", score_col: "{:.4f}"}

        st.dataframe(
            results[display_cols].head(top_n).style.format(format_dict),
            use_container_width=True,
        )

        st.subheader("🏆 Top Materials")
        top3 = results.drop_duplicates(subset=["Material"], keep="first").head(3)

        if len(top3) > 0:
            cols_top = st.columns(len(top3))
            for i in range(len(top3)):
                with cols_top[i]:
                    st.metric(
                        label=f"Rank #{top3.index[i]}",
                        value=top3.iloc[i]["Material"],
                    )
        else:
            st.info("No materials to display")

        # Serialise filter state to a hashable tuple for cache_data
        filter_items = tuple(
            (k, v) for k, v in st.session_state.get("filters", {}).items()
        )
        excel_data = create_full_output(df_mcdm, results, weights_df, filter_items)
        st.download_button(
            label="📥 Download Full MCDM Report",
            data=excel_data,
            file_name=f"mcdm_analysis_{mcdm_method}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_mcdm_custom",
        )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    set_custom_style()

    df1 = load_composition_database()

    st.sidebar.title("Material Analysis")
    st.sidebar.markdown("---")
    selected_page = st.sidebar.radio(
        "Navigation Menu",
        ["Home", "Bandgap Information", "Decision-making Assistant"],
        captions=["Welcome page", "Commonly researched semiconductors", "Multi-criteria decision making tool"],
    )

    st.markdown("""
    <div class="footer">
        Semiconductor Database © 2026 | v5.0 | Developed by HERAWS
    </div>
    """, unsafe_allow_html=True)

    if selected_page == "Home":
        page_home(df1)
    elif selected_page == "Bandgap Information":
        page_bandgap(df1)
    elif selected_page == "Decision-making Assistant":
        page_decision(df1)


if __name__ == "__main__":
    main()
