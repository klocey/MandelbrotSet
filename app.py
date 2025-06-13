import numpy as np
import matplotlib.cm as cm
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go

# Configuration
WIDTH, HEIGHT = 1000, 1000
MAX_ITER = 300
INIT_BOUNDS = (-2.5, 1.5, -2.0, 2.0)

# Mandelbrot Calculation
def mandelbrot(xmin, xmax, ymin, ymax, w, h, max_iter):
    xs = np.linspace(xmin, xmax, w)
    ys = np.linspace(ymin, ymax, h)
    C = xs[np.newaxis, :] + 1j * ys[:, np.newaxis]
    Z = np.zeros_like(C)
    M = np.full(C.shape, max_iter, dtype=int)

    for i in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask] ** 2 + C[mask]
        M[mask & (np.abs(Z) > 2)] = i
    return M

# Convert iteration data to RGB image
def generate_rgb(bounds):
    xmin, xmax, ymin, ymax = bounds
    data = mandelbrot(xmin, xmax, ymin, ymax, WIDTH, HEIGHT, MAX_ITER)
    norm = data / MAX_ITER
    rgba = cm.jet_r(norm)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return rgb[::-1], bounds  # Flip vertically for correct display

# Dash app setup
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H2("Explore the Mandelbrot Set", className="title"),
    html.Button("Reset", id="reset", n_clicks=0, className="reset-button"),
    html.Div(id="info", className="info"),
    
    html.Div([
        dcc.Graph(id="mandelbrot", config={"scrollZoom": True}, className="mandelbrot-graph")
    ], className="graph-wrapper"),
    
    dcc.Store(id="bounds", data=INIT_BOUNDS)
], className="main-container")

@app.callback(
    Output("mandelbrot", "figure"),
    Output("info", "children"),
    Output("bounds", "data"),
    Input("mandelbrot", "relayoutData"),
    Input("reset", "n_clicks"),
    State("bounds", "data"),
)
def update(relayout, reset_clicks, bounds):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered == "reset" or relayout is None:
        b = INIT_BOUNDS
    elif "xaxis.range[0]" in relayout:
        b = (
            float(relayout["xaxis.range[0]"]), float(relayout["xaxis.range[1]"]),
            float(relayout["yaxis.range[0]"]), float(relayout["yaxis.range[1]"])
        )
    else:
        b = bounds

    rgb, (xmin, xmax, ymin, ymax) = generate_rgb(b)
    fig = go.Figure(go.Image(
        z=rgb,
        x0=xmin,
        dx=(xmax - xmin) / WIDTH,
        y0=ymax,
        dy=-(ymax - ymin) / HEIGHT
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            scaleanchor="y",
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            showline=False
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            showline=False
        ),
        dragmode="zoom",
        plot_bgcolor='#595959',
        paper_bgcolor='#595959',
    )

    center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
    zoom = (INIT_BOUNDS[1] - INIT_BOUNDS[0]) / (xmax - xmin)
    info = f"Center: ({center_x:.5f}, {center_y:.5f}) | Zoom: {zoom:.2f}×"
    return fig, info, b

# Run the server
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
