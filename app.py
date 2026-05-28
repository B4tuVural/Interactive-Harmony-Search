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
        "ideal_text": "**Teorik İdeal Çözüm:** $f_{min} = 0$ &nbsp;→&nbsp; konum: (1, 1)",
    },
    "Michalewicz": {
        "func": michalewicz,
        "bounds": [(0.0, np.pi), (0.0, np.pi)],
        "z_floor_offset": -0.1,
        "ideal_text": "**Teorik İdeal Çözüm:** $f_{min} \\approx -1.801$ &nbsp;→&nbsp; konum: (2.20319, 1.57049)",
    },
}

NO_SCROLL_CSS = """
<style>
/* Genel arka plan */
.stApp { background-color: #1E2129; }

/* Ana içerik alanı — padding sıfırla */
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 0rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 100% !important;
}

/* Başlık — tek satıra sığdır, küçük font */
h1 {
    margin-top: 0 !important;
    margin-bottom: 0.2rem !important;
    font-size: 1.25rem !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
h3 { margin-top: 0.2rem !important; margin-bottom: 0.25rem !important; font-size: 0.95rem !important; }

/* Metrikler */
[data-testid="stMetric"] {
    background: #262730;
    border-radius: 8px;
    padding: 0.35rem 0.5rem !important;
}
[data-testid="stMetricLabel"] { font-size: 0.72rem !important; }
[data-testid="stMetricValue"] { font-size: 1.2rem !important; }

/* Alert/info/success kutuları */
[data-testid="stAlert"] {
    padding: 0.35rem 0.6rem !important;
    margin-bottom: 0.3rem !important;
}

/* Slider */
[data-testid="stSlider"] { margin-bottom: 0 !important; padding-bottom: 0 !important; }
.stSlider > label { font-size: 0.78rem !important; margin-bottom: 0 !important; }

/* Paragraf boşlukları */
p { margin-bottom: 0.15rem !important; }

/* ── SİDEBAR ── */
section[data-testid="stSidebar"] {
    overflow-y: auto;
}
/* Sidebar iç blok */
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.6rem !important;
    padding-bottom: 0.5rem !important;
}
/* Sidebar başlık */
section[data-testid="stSidebar"] h2 {
    font-size: 1rem !important;
    margin-bottom: 0.3rem !important;
    margin-top: 0 !important;
}
/* Sidebar her element arası boşluk */
section[data-testid="stSidebar"] .element-container {
    margin-bottom: 0.15rem !important;
}
/* Sidebar slider label */
section[data-testid="stSidebar"] .stSlider > label {
    font-size: 0.78rem !important;
}
/* Sidebar hr */
section[data-testid="stSidebar"] hr {
    margin-top: 0.4rem !important;
    margin-bottom: 0.4rem !important;
}
/* Sidebar selectbox */
section[data-testid="stSidebar"] [data-testid="stSelectbox"] {
    margin-bottom: 0 !important;
}

/* Plotly grafik */
.js-plotly-plot { margin-top: 0 !important; }

/* Genel element boşlukları (ana alan) */
.block-container > div > .element-container { margin-bottom: 0.25rem !important; }
</style>
"""


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
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='f(x,y)',
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.2)),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=480,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="right", x=1,
            font=dict(size=11),
        ),
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
st.markdown(NO_SCROLL_CSS, unsafe_allow_html=True)

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

# --- Optimizasyon Çalıştırma ---
cfg = BENCHMARK_CONFIG[benchmark_choice]
if run_button or needs_run(benchmark_choice):
    run_optimization(cfg, benchmark_choice, hms, r_accept, r_pa, bw, max_iter)

res = st.session_state.hs_results

# ── Başlık ──────────────────────────────────────────────────────────────
st.title("🎶 Harmony Search (HS) — Benchmark Testleri")

# ── Sol panel (metrikler + bilgi + slider) | Sağ panel (grafik) ─────────
left, right = st.columns([1, 2.6], gap="medium")

with left:
    st.subheader("🏆 Algoritmanın Bulduğu Sonuç")

    m1, m2, m3 = st.columns(3)
    m1.metric("f(x,y)", f"{res['best_fitness']:.5f}")
    m2.metric("x₁", f"{res['best_harmony'][0]:.5f}")
    m3.metric("x₂", f"{res['best_harmony'][1]:.5f}")

    st.info(cfg["ideal_text"])

    st.markdown("---")
    st.markdown("👀 **Tarama Adımlarını İncele**")
    show_iter = st.slider(
        "Gösterilecek İterasyon",
        min_value=1,
        max_value=max_iter,
        value=max_iter,
        step=10,
    )

    st.markdown("---")
    st.caption(
        "Bu uygulama `.mlx` benzeri interaktif kontrollerle "
        "**Harmony Search** algoritmasının arama uzayını nasıl taradığını simüle eder."
    )

with right:
    st.subheader("🔍 3D Arama Uzayı ve Tarama Geçmişi")
    X, Y, Z = compute_surface(cfg["func"], cfg["bounds"])
    floor_z  = np.min(Z) + cfg["z_floor_offset"]
    fig = build_figure(X, Y, Z, floor_z, res['history'], show_iter, res['best_harmony'])
    st.plotly_chart(fig, use_container_width=True)