import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils import (
    HarmonySearch, PSO, GA,
    rosenbrock, michalewicz,
)

# --- Sabitler ---
GRID_SIZE = 60

# =====================================================================
#  BENCHMARK FONKSİYONLARI (tüm algoritmalar ortak kullanır)
# =====================================================================
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

# =====================================================================
#  ALGORİTMA KONFİGÜRASYONLARI
#  Her algoritma için: solver fonksiyonu, parametre tanımları (slider),
#  ve solver'ı çağıran "runner" fonksiyonu.
#  Parametre tanımı: (anahtar, etiket, min, max, varsayılan, adım)
#  Sayı tipleri (int/float) slider tipini belirler.
# =====================================================================
def _run_hs(func, bounds, p):
    return HarmonySearch(func, bounds, p["hms"], p["r_accept"], p["r_pa"], p["bw"], p["max_iter"])


def _run_pso(func, bounds, p):
    return PSO(func, bounds, p["pop_size"], p["max_iter"], p["c1"], p["c2"], p["w"])


def _run_ga(func, bounds, p):
    return GA(func, bounds, p["pop_size"], p["max_iter"], p["crossover_rate"], p["mutation_rate"])


ALGO_CONFIG = {
    "HS": {
        "label": "🎶 Harmony Search",
        "desc": "Müzikal doğaçlamadan esinlenen, hafıza tabanlı bir arama algoritması.",
        "runner": _run_hs,
        "params": [
            ("hms",      "Harmony Memory Size (HMS)",   10,   100,   20,   5),
            ("r_accept", "Accept Rate (r_accept)",      0.1,  1.0,   0.85, 0.05),
            ("r_pa",     "Pitch Adjusting Rate (r_pa)", 0.1,  1.0,   0.5,  0.05),
            ("bw",       "Bandwidth (bw)",              0.01, 1.0,   0.12, 0.01),
            ("max_iter", "Maks. İterasyon",             100,  10000, 5000, 100),
        ],
    },
    "PSO": {
        "label": "🐦 Particle Swarm",
        "desc": "Sürü zekâsı; parçacıklar kişisel ve global en iyiye doğru hareket eder.",
        "runner": _run_pso,
        "params": [
            ("pop_size", "Sürü Boyutu (pop_size)",   10,  100,  50,  5),
            ("max_iter", "Maks. İterasyon",           50,  500,  150, 10),
            ("c1",       "Bilişsel Katsayı (c1)",     0.0, 4.0,  1.5, 0.1),
            ("c2",       "Sosyal Katsayı (c2)",       0.0, 4.0,  1.5, 0.1),
            ("w",        "Atalet Ağırlığı (w)",       0.0, 1.5,  0.7, 0.05),
        ],
    },
    "GA": {
        "label": "🧬 Genetic Algorithm",
        "desc": "Doğal seçilim; turnuva seçilimi, aritmetik çaprazlama ve mutasyon.",
        "runner": _run_ga,
        "params": [
            ("pop_size",       "Popülasyon Boyutu (pop_size)", 10,  100, 50,  2),
            ("max_iter",       "Jenerasyon Sayısı",            20,  500, 100, 10),
            ("crossover_rate", "Çaprazlama Oranı",             0.1, 1.0, 0.9, 0.05),
            ("mutation_rate",  "Mutasyon Oranı",               0.0, 1.0, 0.1, 0.05),
        ],
    },
}


# =====================================================================
#  YARDIMCI FONKSİYONLAR
# =====================================================================
@st.cache_data(show_spinner=False)
def compute_surface(benchmark_choice):
    """Yüzeyi benchmark adına göre hesaplar ve önbelleğe alır."""
    cfg = BENCHMARK_CONFIG[benchmark_choice]
    obj_func = cfg["func"]
    bounds = cfg["bounds"]
    x = np.linspace(bounds[0][0], bounds[0][1], GRID_SIZE)
    y = np.linspace(bounds[1][0], bounds[1][1], GRID_SIZE)
    X, Y = np.meshgrid(x, y)
    Z = np.vectorize(lambda xi, yi: obj_func([xi, yi]))(X, Y)
    return X, Y, Z


def build_figure(X, Y, Z, floor_z, history, show_iter, best_pos):
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
        x=[best_pos[0]], y=[best_pos[1]],
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
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def render_params(algo_key, algo_cfg):
    """Algoritmanın parametre slider'larını çizer ve değerleri döndürür."""
    params = {}
    specs = algo_cfg["params"]
    cols = st.columns(len(specs))
    for col, (key, label, lo, hi, default, step) in zip(cols, specs):
        with col:
            params[key] = st.slider(
                label, lo, hi, default, step,
                key=f"{algo_key}_{key}",
            )
    return params


def make_signature(benchmark_choice, params):
    """Optimizasyonun yeniden çalışması gerekip gerekmediğini belirleyen imza."""
    return (benchmark_choice, tuple(sorted(params.items())))


# =====================================================================
#  SEKME (TAB) İÇERİĞİ
# =====================================================================
def render_algorithm_tab(algo_key):
    algo_cfg = ALGO_CONFIG[algo_key]
    state_key = f"{algo_key}_results"

    st.markdown(f"### {algo_cfg['label']}")
    st.caption(algo_cfg["desc"])

    # --- Benchmark seçimi (diğer algoritmalardaki gibi dropdown) ---
    top_left, top_right = st.columns([2, 3])
    with top_left:
        benchmark_choice = st.selectbox(
            "Benchmark Fonksiyonu Seçin",
            list(BENCHMARK_CONFIG.keys()),
            key=f"{algo_key}_benchmark",
        )
    cfg = BENCHMARK_CONFIG[benchmark_choice]

    # --- Algoritma parametreleri ---
    with st.expander("⚙️ Algoritma Parametreleri", expanded=True):
        params = render_params(algo_key, algo_cfg)
        run_button = st.button(
            "🎲 Yeniden Çalıştır (yeni rastgele başlangıç)",
            key=f"{algo_key}_run",
            use_container_width=True,
        )

    # --- Optimizasyon çalıştırma (imza değişti veya buton basıldıysa) ---
    signature = make_signature(benchmark_choice, params)
    prev = st.session_state.get(state_key)
    needs_run = (prev is None) or (prev.get("signature") != signature) or run_button

    if needs_run:
        with st.spinner(f"{algo_cfg['label']} • {benchmark_choice} için optimizasyon hesaplanıyor..."):
            best_fitness, best_pos, history = algo_cfg["runner"](cfg["func"], cfg["bounds"], params)
        st.session_state[state_key] = {
            "best_fitness": best_fitness,
            "best_pos": best_pos,
            "history": history,
            "benchmark": benchmark_choice,
            "signature": signature,
        }

    res = st.session_state[state_key]

    # --- Görselleştirme ---
    st.subheader("🔍 3D Arama Uzayı ve Tarama Geçmişi")
    col1, col2 = st.columns([1, 3])

    total_steps = len(res["history"])
    with col1:
        st.success("🏆 **Algoritmanın Bulduğu Sonuç**")
        st.metric("En İyi f(x, y)", f"{res['best_fitness']:.5f}")
        st.metric("Optimum x (x1)", f"{res['best_pos'][0]:.5f}")
        st.metric("Optimum y (x2)", f"{res['best_pos'][1]:.5f}")
        st.info(cfg["ideal_text"])
        st.markdown("---")
        st.markdown("👀 **Tarama Adımlarını İncele**")
        step = max(1, total_steps // 100)
        show_iter = st.slider(
            "Gösterilecek Adım Sayısı",
            1, total_steps, total_steps, step,
            key=f"{algo_key}_show",
            help="Algoritmanın değerlendirdiği toplam aday nokta sayısı üzerinden ilerler.",
        )
        st.caption(f"Toplam değerlendirilen aday nokta: **{total_steps}**")

    with col2:
        X, Y, Z = compute_surface(benchmark_choice)
        floor_z = np.min(Z) + cfg["z_floor_offset"]
        fig = build_figure(X, Y, Z, floor_z, res["history"], show_iter, res["best_pos"])
        st.plotly_chart(fig, use_container_width=True, key=f"{algo_key}_plot")


# =====================================================================
#  SAYFA YAPILANDIRMASI
# =====================================================================
st.set_page_config(page_title="Metaheuristic Optimizasyon Benchmark", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background-color: #1E2129; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🧠 Metasezgisel Optimizasyon Algoritmaları • Benchmark Testleri")
st.markdown(
    "Aşağıdaki **sekmeler** arasında geçiş yaparak farklı optimizasyon "
    "algoritmalarının (HS, PSO, GA) arama uzayını nasıl taradığını "
    "karşılaştırabilirsiniz. Her sekmede benchmark fonksiyonunu ve algoritma "
    "parametrelerini bağımsız olarak ayarlayabilirsiniz."
)

# --- Sekmeler (web tarayıcı sekmeleri gibi algoritma geçişi) ---
tab_keys = list(ALGO_CONFIG.keys())
tab_labels = [ALGO_CONFIG[k]["label"] for k in tab_keys]
tabs = st.tabs(tab_labels)

for tab, algo_key in zip(tabs, tab_keys):
    with tab:
        render_algorithm_tab(algo_key)
