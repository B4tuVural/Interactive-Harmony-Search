# 🎶 Harmony Search Algoritması — İnteraktif Benchmark Görselleştirici

Harmony Search (HS) meta-sezgisel optimizasyon algoritmasını iki klasik benchmark fonksiyonu üzerinde çalıştıran ve sonuçları interaktif 3D grafik ile gösteren bir Streamlit uygulaması.

## Özellikler

- **Canlı parametre kontrolü** — HMS, kabul oranı, perde ayarlama ve bant genişliğini yan menüden anında değiştirebilirsiniz
- **İki benchmark fonksiyonu** — Logaritmik Rosenbrock ve Michalewicz
- **3D görselleştirme** — Algoritmanın tarayan tüm noktalar ve bulunan optimum, yüzey üzerinde işaretlenir
- **İterasyon kaydırıcısı** — Tarama geçmişini adım adım inceleyebilirsiniz

## Benchmark Fonksiyonları

| Fonksiyon | Arama Uzayı | Teorik Minimum |
|---|---|---|
| Logaritmik Rosenbrock | $x, y \in [-5, 5]$ | $f = 0$ konumu $(1, 1)$ |
| Michalewicz | $x, y \in [0, \pi]$ | $f \approx -1.801$ konumu $(2.203, 1.570)$ |

## Kurulum

```bash
git clone https://github.com/KULLANICI_ADI/harmony-search.git
cd harmony-search
pip install -r requirements.txt
```

## Çalıştırma

```bash
streamlit run app.py
```

Tarayıcınızda otomatik olarak `http://localhost:8501` adresi açılır.

## Algoritma Parametreleri

| Parametre | Açıklama | Varsayılan |
|---|---|---|
| **HMS** | Harmony Memory Size — hafızada tutulan aday sayısı | 20 |
| **r_accept** | Hafızadan seçim olasılığı | 0.85 |
| **r_pa** | Perde ayarlama olasılığı | 0.50 |
| **bw** | Perde ayarlama bant genişliği | 0.15 |
| **max_iter** | Maksimum iterasyon sayısı | 5000 |

## Proje Yapısı

```
harmony-search/
├── app.py            # Streamlit arayüzü ve görselleştirme
├── utils.py          # HS algoritması ve benchmark fonksiyonları
└── requirements.txt
```

## Algoritma Hakkında

Harmony Search, müzisyenlerin doğaçlama sırasındaki arayışından esinlenen bir meta-sezgisel optimizasyon yöntemidir (Geem et al., 2001). Her iterasyonda:

1. Mevcut **Harmony Memory**'den bir aday seçilir (olasılık: `r_accept`)
2. Seçilen değer, `r_pa` olasılığıyla `bw` çarpanı kadar rastgele kaydırılır (Pitch Adjusting)
3. Yeni aday hafızadaki en kötü adaydan iyiyse yer değiştirir

## Gereksinimler

- Python 3.9+
- numpy, streamlit, plotly (`requirements.txt` dosyasına bakın)
