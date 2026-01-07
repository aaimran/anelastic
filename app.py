#!/usr/bin/env python3
"""Dash app to browse WaveQLab3D station time series.

Supports two filename formats:

1) Legacy (coordinates embedded in filename):
     <dataset>_<q>_<r>_<s>_block<id>.dat
     Station name becomes: <q>_<r>_<s>_block<id>

2) New (named stations):
     <dataset>_station_<name>.dat
     Example: traditional_6_pml-on_elastic_station_A.dat
     Station name becomes: station_<name> (e.g. station_A)

3) New (xyz coordinates, no block in name):
    <dataset>_<x>_<y>_<z>.dat
    Example: traditional_6_pml-on_elastic_14.000_0.000_17.000.dat
    Station name becomes: <x>_<y>_<z>

Each file is assumed to have 4 whitespace-delimited columns:
  t  vx  vy  vz

Run:
  python dash_station_viewer.py --data-dir waveqlab3d/simulation/plots
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, TypedDict


class TimeSeries(TypedDict):
    t: List[float]
    vx: List[float]
    vy: List[float]
    vz: List[float]



import plotly.graph_objects as go
from plotly.colors import qualitative as plotly_qual
from plotly.subplots import make_subplots

from dash import ALL, Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

try:
    from dash import ctx  # Dash >= 2.4
except ImportError:  # pragma: no cover
    ctx = None  # type: ignore[assignment]

try:
    import dash_bootstrap_components as dbc
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'dash-bootstrap-components'. "
        "Install requirements from requirements-dash.txt"
    ) from exc


FNAME_RE = re.compile(
    r"^(?P<dataset>.+?)_"  # dataset prefix (legacy; dataset may contain underscores)
    r"(?P<q>[^_]+)_"  # q
    r"(?P<r>[^_]+)_"  # r
    r"(?P<s>[^_]+)_"  # s
    r"(?P<block>block[^.]+)"  # block...
    r"\.dat$",
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

# Only match exact trailing '_anelastic-c2' or '_elastic' (or with dash)
_RE_TRAILING_ANELASTIC_C2 = re.compile(r"(?i)(?:[_-]anelastic[-_]c2)$")
_RE_TRAILING_ELASTIC = re.compile(r"(?i)(?:[_-]elastic)$")

def dataset_base_and_variant(dataset_name: str) -> Tuple[str, Optional[str]]:
    """
    Split a dataset name into (common_base_name, variant).
    Removes variant suffix ('_anelastic-c2' or '_elastic') and 'pml-off_A_' if present.
    Example:
      traditional_6_pml-off_anelastic-c2_A_mq_0.000_14.000_17.000 -> traditional_6_mq
      traditional_6_pml-off_elastic_A_mq_0.000_14.000_17.000     -> traditional_6_mq
    """
    # Remove .dat extension if present
    name = dataset_name
    if name.endswith('.dat'):
        name = name[:-4]

    # Remove trailing coordinates (last three _number tokens)
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
    base = re.sub(r'__', '_', base)  # Clean double underscores
    base = base.rstrip('_-')
    # Strip 'pml-off_A' plus any following underscore if present
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
    # Only process .dat files
    if path.suffix.lower() != ".dat":
        return None

    # Try to extract station as last three underscore-separated numbers before .dat
    stem = path.stem
    parts = stem.split('_')
    if len(parts) >= 3 and all(re.match(r'^-?\d+(\.\d+)?$', p) for p in parts[-3:]):
        station = f"{parts[-3]}_{parts[-2]}_{parts[-1]}"
        dataset = '_'.join(parts[:-3])
        return DatasetInfo(path=path, dataset=dataset, station=station)

    # Fallback to previous logic for other formats
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

    # Legacy format: <dataset>_<q>_<r>_<s>_block<id>.dat
    if len(parts) >= 5:
        dataset = "_".join(parts[:-4])
        q, r, s, block = parts[-4], parts[-3], parts[-2], parts[-1]
        if dataset and block.lower().startswith("block"):
            station = f"{q}_{r}_{s}_{block}"
            return DatasetInfo(path=path, dataset=dataset, station=station)
    return None


def load_timeseries(path: Path) -> TimeSeries:
    t: List[float] = []
    vx: List[float] = []
    vy: List[float] = []
    vz: List[float] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                tt, vxx, vyy, vzz = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
            except ValueError:
                continue
            t.append(tt)
            vx.append(vxx)
            vy.append(vyy)
            vz.append(vzz)

    return {"t": t, "vx": vx, "vy": vy, "vz": vz}


def make_figure(station: str, selected: List[DatasetInfo], plot: str, properties: Dict[str, Dict[str, float]], height: int = 400) -> go.Figure:
    plot_meta = {
        "vx": (f"particle velocity in the x-direction at station {station}", "vx"),
        "vy": (f"particle velocity in the y-direction at station {station}", "vy"),
        "vz": (f"particle velocity in the z-direction at station {station}", "vz"),
    }
    if plot not in plot_meta:
        fig = go.Figure()
        fig.update_layout(
            title="",
            height=400,
            margin=dict(l=40, r=20, t=60, b=40),
        )
        fig.add_annotation(
            text="Invalid plot selected",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return fig

    title, y_col = plot_meta[plot]
    fig = go.Figure()

    # Stable color per dataset across all selected plots.
    datasets_in_order = sorted({i.dataset for i in selected})
    palette = list(plotly_qual.Plotly)
    if not palette:
        palette = ["#1f77b4"]
    dataset_to_color = {
        ds: palette[idx % len(palette)] for idx, ds in enumerate(datasets_in_order)
    }

    for info in selected:
        df = load_timeseries(info.path)
        label = info.dataset
        color = dataset_to_color.get(label)
        props = properties.get(label, {})
        width = props.get('width', 2)
        opacity = props.get('opacity', 1)
        fig.add_trace(
            go.Scatter(
                x=df["t"],
                y=df[y_col],
                mode="lines",
                name=label,
                line=dict(color=color, width=width),
                opacity=opacity,
                showlegend=True,
            )
        )

    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis_title="time (s)",
        yaxis_title="particle velocity (m/s)",
    )
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing station .dat files",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # For deployment on Render, use environment variables
    host = os.getenv('HOST', args.host)
    port = int(os.getenv('PORT', args.port))

    script_dir = Path(__file__).resolve().parent

    if args.data_dir:
        data_dir = Path(args.data_dir).expanduser().resolve()
    else:
        candidates = [
            Path.cwd() / "data",
            Path.cwd() / "waveqlab3d/simulation/plots",
            script_dir / "waveqlab3d/simulation/plots",
            script_dir,
        ]
        data_dir = next((p.resolve() for p in candidates if p.exists()), candidates[-1].resolve())

    if not data_dir.exists():
        raise SystemExit(
            "Data directory not found. Try: "
            "python dash_station_viewer.py --data-dir /path/to/plots\n"
            f"Tried: {data_dir}"
        )

    all_infos, stations = build_index(data_dir)
    by_station = group_by_station(all_infos)

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    server = app.server 


    initial_station = stations[0] if stations else ""
    initial_infos = by_station.get(initial_station, [])
    initial_selected_infos = [initial_infos[0]] if initial_infos else []
    initial_plots = ["vx", "vy", "vz"]
    initial_figs = [make_figure(initial_station, initial_selected_infos, p, {}, 400) for p in initial_plots]

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
                                multi=False,
                                clearable=False,
                                placeholder="Choose station",
                            ),
                            html.Hr(),
                            html.Label("Datasets"),
                            html.Button("Clear Dataset Selections", id="clear-dataset-button", style={"marginLeft": "10px"}),
                            dcc.Store(id="dataset-path-map", data={}),
                            dcc.Store(id="dataset-base-order", data=[]),
                            dcc.Store(id="dataset-selection-store", data={}),
                            html.Div(id="dataset-table-container"),
                            html.Hr(),
                            html.Div(
                                [
                                    html.Label("Select timeseries plot(s):", style={"marginRight": "10px"}),
                                    dcc.Checklist(
                                        id="plot-checklist",
                                        options=[
                                            {"label": "vx", "value": "vx"},
                                            {"label": "vy", "value": "vy"},
                                            {"label": "vz", "value": "vz"},
                                        ],
                                        value=initial_plots,
                                        labelStyle={"display": "inline", "marginRight": "15px"},
                                        style={"display": "inline"},
                                    ),
                                ],
                                style={"display": "flex", "alignItems": "center"},
                            ),
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

    @app.callback(
        [
            Output("dataset-table-container", "children"),
            Output("dataset-path-map", "data"),
            Output("dataset-base-order", "data"),
        ],
        [
            Input("station-dropdown", "value"),
            Input("dataset-selection-store", "data"),
        ],
    )
    def update_dataset_table(selected_station: str, selection_store: Dict):
        infos = by_station.get(selected_station or "", [])

        stencil_to_variants: Dict[str, Dict[Tuple[str, str], Dict[str, str]]] = {}
        for info in infos:
            base, variant = dataset_base_and_variant(info.dataset)
            if variant not in (VARIANT_ELASTIC, VARIANT_ANELASTIC_C2):
                continue
            parts = base.split('_')
            if len(parts) < 3:
                continue
            stencil = parts[0]
            order = parts[1]
            ver = '_'.join(parts[2:])
            key = (order, ver)
            if stencil not in stencil_to_variants:
                stencil_to_variants[stencil] = {}
            if key not in stencil_to_variants[stencil]:
                stencil_to_variants[stencil][key] = {}
            stencil_to_variants[stencil][key].setdefault(variant, str(info.path))

        # Sort stencils
        sorted_stencils = sorted(stencil_to_variants.keys(), key=lambda s: ['traditional', 'upwind', 'upwind-drp'].index(s) if s in ['traditional', 'upwind', 'upwind-drp'] else 999)

        # Build grouped and base_order
        grouped = {}
        base_order = []
        for stencil in sorted_stencils:
            sorted_keys = sorted(stencil_to_variants[stencil].keys())
            for key in sorted_keys:
                order, ver = key
                base = f"{stencil}_{order}_{ver}"
                grouped[base] = stencil_to_variants[stencil][key]
                base_order.append(base)

        # Decide defaults
        has_any_selection = False
        for base in base_order:
            saved = selection_store.get(base, {}) if isinstance(selection_store, dict) else {}
            if bool(saved.get(VARIANT_ELASTIC)) or bool(saved.get(VARIANT_ANELASTIC_C2)):
                has_any_selection = True
                break

        if not has_any_selection and base_order:
            first_base = base_order[0]
            selection_store = dict(selection_store or {})
            selection_store.setdefault(first_base, {})
            if grouped[first_base].get(VARIANT_ELASTIC):
                selection_store[first_base][VARIANT_ELASTIC] = True
            elif grouped[first_base].get(VARIANT_ANELASTIC_C2):
                selection_store[first_base][VARIANT_ANELASTIC_C2] = True

        # Build table
        header = html.Thead(
            html.Tr(
                [
                    html.Th("Stencil"),
                    html.Th("Order"),
                    html.Th("Ver."),
                    html.Th("Elastic"),
                    html.Th("Anelastic (c=2)"),
                ]
            )
        )

        rows = []
        for stencil in sorted_stencils:
            sorted_keys = sorted(stencil_to_variants[stencil].keys())
            num_rows = len(sorted_keys)
            for i, key in enumerate(sorted_keys):
                order, ver = key
                base = f"{stencil}_{order}_{ver}"
                variants = grouped[base]
                elastic_path = variants.get(VARIANT_ELASTIC)
                anelastic_path = variants.get(VARIANT_ANELASTIC_C2)

                saved = selection_store.get(base, {}) if isinstance(selection_store, dict) else {}
                elastic_checked = bool(saved.get(VARIANT_ELASTIC)) and elastic_path is not None
                anelastic_checked = bool(saved.get(VARIANT_ANELASTIC_C2)) and anelastic_path is not None

                elastic_options = [
                    {
                        "label": "",
                        "value": "on",
                        "disabled": elastic_path is None,
                    }
                ]
                anelastic_options = [
                    {
                        "label": "",
                        "value": "on",
                        "disabled": anelastic_path is None,
                    }
                ]

                order_cell = html.Td(order)
                ver_cell = html.Td(ver)
                elastic_cell = html.Td(
                    dcc.Checklist(
                        id={"type": "dataset-elastic", "base": base},
                        options=elastic_options,
                        value=["on"] if elastic_checked else [],
                    )
                )
                anelastic_cell = html.Td(
                    dcc.Checklist(
                        id={"type": "dataset-anelastic", "base": base},
                        options=anelastic_options,
                        value=["on"] if anelastic_checked else [],
                    )
                )

                if i == 0:
                    row_children = [
                        html.Td(stencil, rowSpan=num_rows),
                        order_cell,
                        ver_cell,
                        elastic_cell,
                        anelastic_cell,
                    ]
                else:
                    row_children = [
                        order_cell,
                        ver_cell,
                        elastic_cell,
                        anelastic_cell,
                    ]

                rows.append(html.Tr(row_children))

        table = dbc.Table(
            [header, html.Tbody(rows)],
            bordered=True,
            hover=True,
            size="sm",
            responsive=True,
        )

        path_map = grouped
        return table, path_map, base_order

    @app.callback(
        Output("dataset-selection-store", "data"),
        Input("clear-dataset-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_dataset_selections(n_clicks):
        return {}

    @app.callback(
        Output("line-properties-store", "data"),
        [
            Input({"type": "width", "dataset": ALL}, "value"),
            Input({"type": "opacity", "dataset": ALL}, "value"),
        ],
        State("line-properties-store", "data"),
        prevent_initial_call=True,
    )
    def update_line_properties(width_values, opacity_values, current_props):
        if not current_props:
            current_props = {}
        # Since ALL, need to match by index, but since ids have dataset, better to use ctx
        triggered = ctx.triggered
        if triggered:
            prop_id = triggered[0]['prop_id']
            # prop_id like '{"type":"width","dataset":"some_ds"}.value'
            import json
            id_dict = json.loads(prop_id.split('.')[0])
            ds = id_dict['dataset']
            typ = id_dict['type']
            val = triggered[0]['value']
            if ds not in current_props:
                current_props[ds] = {}
            current_props[ds][typ] = val
        return current_props

    @app.callback(
        [
            Output("plot-container", "children"),
            Output("line-controls-container", "children"),
        ],
        [
            Input("station-dropdown", "value"),
            Input("plot-checklist", "value"),
            Input({"type": "dataset-elastic", "base": ALL}, "value"),
            Input({"type": "dataset-anelastic", "base": ALL}, "value"),
            Input("plot-height-slider", "value"),
        ],
        [
            State("dataset-path-map", "data"),
            State("dataset-base-order", "data"),
            State("line-properties-store", "data"),
        ],
    )
    def update_plot(
        station: str,
        plots_selected: List[str],
        elastic_values: List[List[str]],
        anelastic_values: List[List[str]],
        plot_height: int,
        path_map: Dict,
        base_order: List[str],
        properties: Dict,
    ):
        selected_infos: List[DatasetInfo] = []

        if not base_order or not isinstance(path_map, dict):
            figs = [make_figure(station or "", selected_infos, p, properties, plot_height) for p in plots_selected or []]
            controls = html.Div()
            return [dcc.Graph(figure=fig) for fig in figs], controls

        for idx, base in enumerate(base_order):
            variants = path_map.get(base, {}) if isinstance(path_map.get(base, {}), dict) else {}
            if idx < len(elastic_values) and elastic_values[idx]:
                p = variants.get(VARIANT_ELASTIC)
                if p:
                    info = parse_dataset_info(Path(p))
                    if info is not None:
                        selected_infos.append(info)

            if idx < len(anelastic_values) and anelastic_values[idx]:
                p = variants.get(VARIANT_ANELASTIC_C2)
                if p:
                    info = parse_dataset_info(Path(p))
                    if info is not None:
                        selected_infos.append(info)

        figs = [make_figure(station or "", selected_infos, p, properties, plot_height) for p in plots_selected or []]

        # Create controls table
        selected_datasets = list(set(info.dataset for info in selected_infos))
        if selected_datasets:
            header = html.Thead(
                html.Tr([
                    html.Th("Dataset"),
                    html.Th("Line Width"),
                    html.Th("Line Opacity"),
                ])
            )
            rows = []
            for ds in sorted(selected_datasets):
                props = properties.get(ds, {})
                rows.append(html.Tr([
                    html.Td(ds),
                    html.Td(dcc.Slider(
                        id={"type": "width", "dataset": ds},
                        min=1, max=5, value=props.get('width', 2), step=1,
                        marks={1: '1', 3: '3', 5: '5'}
                    )),
                    html.Td(dcc.Slider(
                        id={"type": "opacity", "dataset": ds},
                        min=0, max=1, value=props.get('opacity', 1), step=0.1,
                        marks={0: '0', 0.5: '0.5', 1.0: '1.0'}
                    )),
                ]))
            table = dbc.Table([header, html.Tbody(rows)], bordered=True, size="sm")
            controls = table
        else:
            controls = html.Div()

        return [dcc.Graph(figure=fig) for fig in figs], controls

    app.run(host=host, port=port, debug=args.debug)


if __name__ == "__main__":
    main()
