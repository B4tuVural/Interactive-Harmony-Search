import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils import HarmonySearch, rosenbrock, michalewicz

# --- Sabitler ---
GRID_SIZE = 60

BENCHMARK_CONFIG = {
    "Logaritmik Rosenbrock": {
        "func": rosenbrock,
        "bounds": [(-5.0, 5.0), (-5.0, 5.0)],
        "z_floor_offset": 0,
        "ideal_text": "**Teorik İdeal Çözüm:** $f_{min} = 0$ konum:&nbsp;(1,&nbsp;1)",
    },
    "Michalewicz": {
        "func": michalewicz,
        "bounds": [(0.0, np.pi), (0.0, np.pi)],
        "z_floor_offset": -0.1,
        "ideal_text": "**Teorik İdeal Çözüm:** $f_{min} \\approx -1.801$ konum:&nbsp;(2.20319,&nbsp;1.57049)",
    },
}


# --- Yardımcı Fonksiyonlar ---
def compute_surface(obj_func, bounds):
    x = np.linspace(bounds[0][0], bounds[0][1], GRID_SIZE)
    y = np.linspace(bounds[1][0], bounds[1][1], GRID_SIZE)
    X, Y = np.meshgrid(x, y)
    Z = np.vectorize(lambda xi, yi: obj_func([xi, yi]))(X, Y)
    return X, Y, Z


def build_figure(X, Y, Z, floor_z, history, show_iter, best_harmony):
    sliced = history[:show_iter]
    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Greys',
        opacity=0.8,
        showscale=False,
        contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True),
    ))
    fig.add_trace(go.Scatter3d(
        x=sliced[:, 0], y=sliced[:, 1],
        z=np.full(len(sliced), floor_z),
        mode='markers',
        marker=dict(size=2, color='green', opacity=0.6),
        name='Arama Geçmişi',
    ))
    fig.add_trace(go.Scatter3d(
        x=[best_harmony[0]], y=[best_harmony[1]],
        z=[floor_z],
        mode='markers',
        marker=dict(size=15, color='red', symbol='cross', line=dict(color='white', width=2)),
        name='Optimum Nokta',
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X Ekseni',
            yaxis_title='Y Ekseni',
            zaxis_title='f(x, y)',
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.2)),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=520,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def needs_run(benchmark_choice):
    if "hs_results" not in st.session_state:
        return True
    return st.session_state.hs_results.get('benchmark') != benchmark_choice


def run_optimization(cfg, benchmark_choice, hms, r_accept, r_pa, bw, max_iter):
    with st.spinner(f'{benchmark_choice} için optimizasyon hesaplanıyor...'):
        best_fitness, best_harmony, history = HarmonySearch(
            cfg["func"], cfg["bounds"], hms, r_accept, r_pa, bw, max_iter
        )
    st.session_state.hs_results = {
        'best_fitness': best_fitness,
        'best_harmony': best_harmony,
        'history': history,
        'benchmark': benchmark_choice,
    }


# --- Sayfa Yapılandırması ---
st.set_page_config(page_title="Harmony Search Optimizasyonu", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background-color: #1E2129; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🎶 Harmony Search (HS) Algoritması Benchmark Testleri")
st.markdown("Bu uygulama, `.mlx` benzeri interaktif kontrollerle **Harmony Search** algoritmasının arama uzayını nasıl taradığını simüle eder.")

# --- Yan Menü ---
with st.sidebar:
    st.header("⚙️ HS Parametreleri")
    benchmark_choice = st.selectbox("Benchmark Fonksiyonu Seçin", list(BENCHMARK_CONFIG.keys()))
    st.markdown("---")
    hms      = st.slider("Harmony Memory Size (HMS)", 10, 100, 20, 5)
    r_accept = st.slider("Accept Rate (r_accept)", 0.1, 1.0, 0.85, 0.05)
    r_pa     = st.slider("Pitch Adjusting Rate (r_pa)", 0.1, 1.0, 0.5, 0.05)
    bw       = st.slider("Bandwidth (bw)", 0.01, 1.0, 0.15, 0.01)
    max_iter = st.slider("Maks. İterasyon", 100, 10000, 5000, 100)
    st.markdown("---")
    run_button = st.button("🚀 Optimizasyonu Başlat / Yenile", use_container_width=True)
    st.markdown("---")
    st.markdown("👀 **Tarama Adımlarını İncele**")
    show_iter = st.slider("Gösterilecek İterasyon Aralığı", 1, max_iter, max_iter, 10)

# --- Optimizasyon Çalıştırma ---
cfg = BENCHMARK_CONFIG[benchmark_choice]
if run_button or needs_run(benchmark_choice):
    run_optimization(cfg, benchmark_choice, hms, r_accept, r_pa, bw, max_iter)

res = st.session_state.hs_results

# --- Görselleştirme ---
st.subheader("🔍 3D Arama Uzayı ve Tarama Geçmişi")
col1, col2 = st.columns([1, 3])

with col1:
    st.success("🏆 **Algoritmanın Bulduğu Sonuç**")
    st.metric("En İyi f(x, y)", f"{res['best_fitness']:.5f}")
    st.metric("Optimum x (x1)", f"{res['best_harmony'][0]:.5f}")
    st.metric("Optimum y (x2)", f"{res['best_harmony'][1]:.5f}")
    st.info(cfg["ideal_text"])

with col2:
    X, Y, Z = compute_surface(cfg["func"], cfg["bounds"])
    floor_z = np.min(Z) + cfg["z_floor_offset"]
    fig = build_figure(X, Y, Z, floor_z, res['history'], show_iter, res['best_harmony'])
    st.plotly_chart(fig, use_container_width=True)
