#!/usr/bin/env python3
"""Dash app to browse WaveQLab3D station time series."""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, TypedDict

import plotly.graph_objects as go
from plotly.colors import qualitative as plotly_qual
from dash import ALL, Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# --- 1. SETUP & HELPER CLASSES ---

class TimeSeries(TypedDict):
    t: List[float]
    vx: List[float]
    vy: List[float]
    vz: List[float]

# Try to import ctx
try:
    from dash import ctx
except ImportError:
    ctx = None

# Regex patterns
FNAME_RE = re.compile(
    r"^(?P<dataset>.+?)_(?P<q>[^_]+)_(?P<r>[^_]+)_(?P<s>[^_]+)_(?P<block>block[^.]+)\.dat$",
    flags=re.IGNORECASE,
)
STATION_NAME_RE = re.compile(
    r"^(?P<dataset>.+?)_station_(?P<station>[A-Za-z0-9]+)\.dat$",
    flags=re.IGNORECASE,
)
NUM_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?"
XYZ_RE = re.compile(
    rf"^(?P<dataset>.+?)_(?P<x>{NUM_RE})_(?P<y>{NUM_RE})_(?P<z>{NUM_RE})\.dat$",
    flags=re.IGNORECASE,
)

@dataclass(frozen=True)
class DatasetInfo:
    path: Path
    dataset: str
    station: str

VARIANT_ELASTIC = "elastic"
VARIANT_ANELASTIC_C2 = "anelastic_c2"

# --- 2. HELPER FUNCTIONS ---

def dataset_base_and_variant(dataset_name: str) -> Tuple[str, Optional[str]]:
    name = dataset_name
    if name.endswith('.dat'):
        name = name[:-4]
    parts = name.split('_')
    if len(parts) > 3 and all(re.match(r'^-?\d+(\.\d+)?$', p) for p in parts[-3:]):
        name = '_'.join(parts[:-3])

    lowered = name.lower()
    variant: Optional[str] = None
    if re.search(r'[_-]anelastic[-_]c2', lowered):
        variant = VARIANT_ANELASTIC_C2
        base = re.sub(r'[_-]anelastic[-_]c2', '', name)
    elif re.search(r'[_-]elastic', lowered):
        variant = VARIANT_ELASTIC
        base = re.sub(r'[_-]elastic', '', name)
    else:
        base = name
    base = re.sub(r'__', '_', base)
    base = base.rstrip('_-')
    base = re.sub(r'pml-off_A_?', '', base)
    if not base:
        base = dataset_name
    return base, variant

def iter_dataset_files(data_dir: Path) -> List[Path]:
    patterns = ["*.dat"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(Path(p).resolve() for p in glob.glob(str(data_dir / pat)))
    files = [p for p in files if p.is_file()]
    files.sort(key=lambda p: p.name)
    return files

def parse_dataset_info(path: Path) -> Optional[DatasetInfo]:
    if path.suffix.lower() != ".dat":
        return None
    stem = path.stem
    parts = stem.split('_')
    if len(parts) >= 3 and all(re.match(r'^-?\d+(\.\d+)?$', p) for p in parts[-3:]):
        station = f"{parts[-3]}_{parts[-2]}_{parts[-1]}"
        dataset = '_'.join(parts[:-3])
        return DatasetInfo(path=path, dataset=dataset, station=station)

    m2 = STATION_NAME_RE.match(path.name)
    if m2:
        dataset = m2.group("dataset")
        station = f"station_{m2.group('station')}"
        return DatasetInfo(path=path, dataset=dataset, station=station)

    m3 = XYZ_RE.match(path.name)
    if m3:
        dataset = m3.group("dataset")
        x, y, z = m3.group("x"), m3.group("y"), m3.group("z")
        station = f"{x}_{y}_{z}"
        return DatasetInfo(path=path, dataset=dataset, station=station)

    if len(parts) >= 5:
        dataset = "_".join(parts[:-4])
        q, r, s, block = parts[-4], parts[-3], parts[-2], parts[-1]
        if dataset and block.lower().startswith("block"):
            station = f"{q}_{r}_{s}_{block}"
            return DatasetInfo(path=path, dataset=dataset, station=station)
    return None

def load_timeseries(path: Path) -> TimeSeries:
    t, vx, vy, vz = [], [], [], []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line: continue
            parts = line.split()
            if len(parts) < 4: continue
            try:
                tt, vxx, vyy, vzz = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
            except ValueError: continue
            t.append(tt); vx.append(vxx); vy.append(vyy); vz.append(vzz)
    return {"t": t, "vx": vx, "vy": vy, "vz": vz}

def make_figure(station: str, selected: List[DatasetInfo], plot: str, properties: Dict[str, Dict[str, float]], height: int = 400) -> go.Figure:
    plot_meta = {
        "vx": (f"particle velocity in the x-direction at station {station}", "vx"),
        "vy": (f"particle velocity in the y-direction at station {station}", "vy"),
        "vz": (f"particle velocity in the z-direction at station {station}", "vz"),
    }
    if plot not in plot_meta:
        fig = go.Figure()
        fig.add_annotation(text="Invalid plot", showarrow=False)
        return fig

    title, y_col = plot_meta[plot]
    fig = go.Figure()
    datasets_in_order = sorted({i.dataset for i in selected})
    palette = list(plotly_qual.Plotly) or ["#1f77b4"]
    dataset_to_color = {ds: palette[idx % len(palette)] for idx, ds in enumerate(datasets_in_order)}

    for info in selected:
        df = load_timeseries(info.path)
        label = info.dataset
        color = dataset_to_color.get(label)
        props = properties.get(label, {})
        width = props.get('width', 2)
        opacity = props.get('opacity', 1)
        fig.add_trace(go.Scatter(x=df["t"], y=df[y_col], mode="lines", name=label,
                                 line=dict(color=color, width=width), opacity=opacity, showlegend=True))

    fig.update_layout(title=title, height=height, margin=dict(l=40, r=20, t=60, b=40),
                      xaxis_title="time (s)", yaxis_title="particle velocity (m/s)")
    return fig

def build_index(data_dir: Path) -> Tuple[List[DatasetInfo], List[str]]:
    infos: List[DatasetInfo] = []
    for path in iter_dataset_files(data_dir):
        info = parse_dataset_info(path)
        if info is not None:
            infos.append(info)
    stations = sorted({i.station for i in infos})
    return infos, stations

def group_by_station(infos: Iterable[DatasetInfo]) -> Dict[str, List[DatasetInfo]]:
    out: Dict[str, List[DatasetInfo]] = {}
    for i in infos:
        out.setdefault(i.station, []).append(i)
    for st in out:
        out[st].sort(key=lambda x: x.dataset)
    return out

# --- 3. GLOBAL INITIALIZATION (Executed on Import) ---

# Initialize App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# CRITICAL: Expose server for Gunicorn
server = app.server

# Locate Data Directory
script_dir = Path(__file__).resolve().parent
# We prioritize a folder named 'data' relative to the script
candidates = [
    Path.cwd() / "data",
    Path.cwd() / "waveqlab3d/simulation/plots",
    script_dir / "waveqlab3d/simulation/plots",
    script_dir,
]
data_dir = next((p.resolve() for p in candidates if p.exists()), candidates[-1].resolve())

# Load Data
all_infos, stations = build_index(data_dir)
by_station = group_by_station(all_infos)

# Setup Defaults
initial_station = stations[0] if stations else ""
initial_infos = by_station.get(initial_station, [])
initial_selected_infos = [initial_infos[0]] if initial_infos else []
initial_plots = ["vx", "vy", "vz"]
initial_figs = [make_figure(initial_station, initial_selected_infos, p, {}, 400) for p in initial_plots]

# --- 4. LAYOUT DEFINITION ---

app.layout = dbc.Container(
    [
        dbc.Badge(f"Data dir: {data_dir}", color="secondary", className="mb-3"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3("Setup A, PML-Off"),
                        html.Label("Station"),
                        dcc.Dropdown(
                            id="station-dropdown",
                            options=[{"label": s, "value": s} for s in stations],
                            value=initial_station,
                            multi=False, clearable=False, placeholder="Choose station",
                        ),
                        html.Hr(),
                        html.Label("Datasets"),
                        html.Button("Clear Dataset Selections", id="clear-dataset-button", style={"marginLeft": "10px"}),
                        dcc.Store(id="dataset-path-map", data={}),
                        dcc.Store(id="dataset-base-order", data=[]),
                        dcc.Store(id="dataset-selection-store", data={}),
                        html.Div(id="dataset-table-container"),
                        html.Hr(),
                        html.Div([
                            html.Label("Select timeseries plot(s):", style={"marginRight": "10px"}),
                            dcc.Checklist(
                                id="plot-checklist",
                                options=[{"label": "vx", "value": "vx"}, {"label": "vy", "value": "vy"}, {"label": "vz", "value": "vz"}],
                                value=initial_plots,
                                labelStyle={"display": "inline", "marginRight": "15px"}, style={"display": "inline"},
                            ),
                        ], style={"display": "flex", "alignItems": "center"}),
                        html.Hr(),
                        html.Label("Adjust line properties:"),
                        dcc.Store(id="line-properties-store", data={}),
                        html.Div(id="line-controls-container"),
                        html.Hr(),
                        html.Label("Adjust plot height:"),
                        dcc.Slider(id="plot-height-slider", min=200, max=800, value=400, step=50, marks={200:'200', 400:'400', 600:'600', 800:'800'}),
                    ],
                    style={"flex": "0 0 20%", "maxWidth": "20%"},
                ),
                dbc.Col(
                    [
                        html.Div(id="plot-container", children=[dcc.Graph(figure=fig) for fig in initial_figs])
                    ],
                    style={"flex": "0 0 80%", "maxWidth": "80%", "overflowY": "auto", "maxHeight": "95vh"},
                ),
            ],
            align="start",
        ),
    ],
    fluid=True,
    className="p-3",
)

# --- 5. CALLBACKS ---

@app.callback(
    [Output("dataset-table-container", "children"),
     Output("dataset-path-map", "data"),
     Output("dataset-base-order", "data")],
    [Input("station-dropdown", "value"),
     Input("dataset-selection-store", "data")]
)
def update_dataset_table(selected_station: str, selection_store: Dict):
    infos = by_station.get(selected_station or "", [])
    stencil_to_variants: Dict[str, Dict[Tuple[str, str], Dict[str, str]]] = {}
    
    for info in infos:
        base, variant = dataset_base_and_variant(info.dataset)
        if variant not in (VARIANT_ELASTIC, VARIANT_ANELASTIC_C2): continue
        parts = base.split('_')
        if len(parts) < 3: continue
        stencil, order, ver = parts[0], parts[1], '_'.join(parts[2:])
        key = (order, ver)
        if stencil not in stencil_to_variants: stencil_to_variants[stencil] = {}
        if key not in stencil_to_variants[stencil]: stencil_to_variants[stencil][key] = {}
        stencil_to_variants[stencil][key].setdefault(variant, str(info.path))

    sorted_stencils = sorted(stencil_to_variants.keys(), key=lambda s: ['traditional', 'upwind', 'upwind-drp'].index(s) if s in ['traditional', 'upwind', 'upwind-drp'] else 999)
    grouped = {}
    base_order = []
    
    for stencil in sorted_stencils:
        for key in sorted(stencil_to_variants[stencil].keys()):
            order, ver = key
            base = f"{stencil}_{order}_{ver}"
            grouped[base] = stencil_to_variants[stencil][key]
            base_order.append(base)

    has_any_selection = False
    for base in base_order:
        saved = selection_store.get(base, {}) if isinstance(selection_store, dict) else {}
        if bool(saved.get(VARIANT_ELASTIC)) or bool(saved.get(VARIANT_ANELASTIC_C2)):
            has_any_selection = True; break

    if not has_any_selection and base_order:
        first_base = base_order[0]
        selection_store = dict(selection_store or {})
        selection_store.setdefault(first_base, {})
        if grouped[first_base].get(VARIANT_ELASTIC): selection_store[first_base][VARIANT_ELASTIC] = True
        elif grouped[first_base].get(VARIANT_ANELASTIC_C2): selection_store[first_base][VARIANT_ANELASTIC_C2] = True

    header = html.Thead(html.Tr([html.Th("Stencil"), html.Th("Order"), html.Th("Ver."), html.Th("Elastic"), html.Th("Anelastic (c=2)")]))
    rows = []
    
    for stencil in sorted_stencils:
        sorted_keys = sorted(stencil_to_variants[stencil].keys())
        for i, key in enumerate(sorted_keys):
            order, ver = key
            base = f"{stencil}_{order}_{ver}"
            variants = grouped[base]
            elastic_path, anelastic_path = variants.get(VARIANT_ELASTIC), variants.get(VARIANT_ANELASTIC_C2)
            saved = selection_store.get(base, {}) if isinstance(selection_store, dict) else {}
            
            elastic_cell = html.Td(dcc.Checklist(id={"type": "dataset-elastic", "base": base}, options=[{"label": "", "value": "on", "disabled": elastic_path is None}], value=["on"] if bool(saved.get(VARIANT_ELASTIC)) and elastic_path else []))
            anelastic_cell = html.Td(dcc.Checklist(id={"type": "dataset-anelastic", "base": base}, options=[{"label": "", "value": "on", "disabled": anelastic_path is None}], value=["on"] if bool(saved.get(VARIANT_ANELASTIC_C2)) and anelastic_path else []))
            
            row_children = [html.Td(stencil, rowSpan=len(sorted_keys))] if i == 0 else []
            row_children.extend([html.Td(order), html.Td(ver), elastic_cell, anelastic_cell])
            rows.append(html.Tr(row_children))

    return dbc.Table([header, html.Tbody(rows)], bordered=True, hover=True, size="sm", responsive=True), grouped, base_order

@app.callback(Output("dataset-selection-store", "data"), Input("clear-dataset-button", "n_clicks"), prevent_initial_call=True)
def clear_dataset_selections(n_clicks): return {}

@app.callback(Output("line-properties-store", "data"), [Input({"type": "width", "dataset": ALL}, "value"), Input({"type": "opacity", "dataset": ALL}, "value")], State("line-properties-store", "data"), prevent_initial_call=True)
def update_line_properties(width_values, opacity_values, current_props):
    if not current_props: current_props = {}
    triggered = ctx.triggered
    if triggered:
        prop_id = triggered[0]['prop_id']
        import json
        id_dict = json.loads(prop_id.split('.')[0])
        ds, typ, val = id_dict['dataset'], id_dict['type'], triggered[0]['value']
        if ds not in current_props: current_props[ds] = {}
        current_props[ds][typ] = val
    return current_props

@app.callback(
    [Output("plot-container", "children"), Output("line-controls-container", "children")],
    [Input("station-dropdown", "value"), Input("plot-checklist", "value"), Input({"type": "dataset-elastic", "base": ALL}, "value"), Input({"type": "dataset-anelastic", "base": ALL}, "value"), Input("plot-height-slider", "value")],
    [State("dataset-path-map", "data"), State("dataset-base-order", "data"), State("line-properties-store", "data")]
)
def update_plot(station, plots_selected, elastic_values, anelastic_values, plot_height, path_map, base_order, properties):
    selected_infos = []
    if base_order and isinstance(path_map, dict):
        for idx, base in enumerate(base_order):
            variants = path_map.get(base, {})
            if idx < len(elastic_values) and elastic_values[idx]:
                if p := variants.get(VARIANT_ELASTIC): 
                    if info := parse_dataset_info(Path(p)): selected_infos.append(info)
            if idx < len(anelastic_values) and anelastic_values[idx]:
                if p := variants.get(VARIANT_ANELASTIC_C2):
                    if info := parse_dataset_info(Path(p)): selected_infos.append(info)

    figs = [make_figure(station or "", selected_infos, p, properties, plot_height) for p in plots_selected or []]
    
    # Create controls table
    selected_datasets = sorted({info.dataset for info in selected_infos})
    if selected_datasets:
        rows = []
        for ds in selected_datasets:
            props = properties.get(ds, {})
            rows.append(html.Tr([
                html.Td(ds),
                html.Td(dcc.Slider(id={"type": "width", "dataset": ds}, min=1, max=5, value=props.get('width', 2), step=1, marks={1:'1',3:'3',5:'5'})),
                html.Td(dcc.Slider(id={"type": "opacity", "dataset": ds}, min=0, max=1, value=props.get('opacity', 1), step=0.1, marks={0:'0',0.5:'0.5',1:'1'}))
            ]))
        controls = dbc.Table([html.Thead(html.Tr([html.Th("Dataset"), html.Th("Line Width"), html.Th("Line Opacity")])), html.Tbody(rows)], bordered=True, size="sm")
    else:
        controls = html.Div()
        
    return [dcc.Graph(figure=fig) for fig in figs], controls

# --- 6. EXECUTION ---

if __name__ == "__main__":
    app.run(debug=True, port=8050)