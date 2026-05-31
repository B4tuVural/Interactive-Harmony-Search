import numpy as np


# =====================================================================
#  HARMONY SEARCH (HS)
# =====================================================================
def HarmonySearch(obj_function, bounds, HMS, r_accept, r_pa, bw, max_iter):
    '''
    [obj_function, lower_bound, upper_bound, num_vars]
    hedef fonksiyon: obj_func, lower_bound: uzayın alt sınırı
    upper_bound: uzayın üst sınırı, num_vars: uzayın eksen sayısı

    HMS :  Harmony Hafızası Boyutu (Harmony Memory Size)
    r_accept : Kabul oranı (Accept Rate)
    r_pa :  Perde Ayarlama oranı (Pitch Adjusting Rate)
    bw :  bant genişliği çarpanı (Bandwidth ratio)
    max_iter : Maks. iterasyon sayısı değişkeni (Max iteration variable)
    '''
    print("HS Optimizasyonu Başladı")
    print("Harmony Memory başlatılıyor...")

    num_vars = len(bounds)
    lower_bound = np.array([b[0] for b in bounds])
    upper_bound = np.array([b[1] for b in bounds])

    # Harmony Memory (HM) Başlatma
    HM = lower_bound + (upper_bound - lower_bound) * np.random.rand(HMS, num_vars)
    HM_fitness = np.apply_along_axis(obj_function, 1, HM)

    # Başlangıç sıralaması
    sort_idx = np.argsort(HM_fitness)
    HM = HM[sort_idx]
    HM_fitness = HM_fitness[sort_idx]

    # Görselleştirme için geçmişi (history) tutma
    history = np.zeros((max_iter, num_vars))

    for t in range(max_iter):
        new_harmony = np.zeros(num_vars)

        for i in range(num_vars):
            if np.random.rand() < r_accept:
                rand_idx = np.random.randint(HMS)
                new_harmony[i] = HM[rand_idx, i]

                # Pitch Adjusting (Perde Ayarlama)
                if np.random.rand() < r_pa:
                    new_harmony[i] += (np.random.rand() * 2 - 1) * bw
            else:
                # Rastgele yeni değer
                new_harmony[i] = lower_bound[i] + (upper_bound[i] - lower_bound[i]) * np.random.rand()

        # Sınır kontrolleri (Boundary check)
        new_harmony = np.clip(new_harmony, lower_bound, upper_bound)
        new_fitness = obj_function(new_harmony)

        # Geçmişe kaydet
        history[t, :] = new_harmony

        # Eğer yeni aday, hafızadaki en kötü adaydan daha iyiyse yer değiştir
        if new_fitness < HM_fitness[-1]:
            HM[-1, :] = new_harmony
            HM_fitness[-1] = new_fitness

            # Yeniden sırala
            sort_idx = np.argsort(HM_fitness)
            HM = HM[sort_idx]
            HM_fitness = HM_fitness[sort_idx]

    best_harmony = HM[0, :]
    best_fitness = HM_fitness[0]

    return best_fitness, best_harmony, history


# =====================================================================
#  PARTICLE SWARM OPTIMIZATION (PSO)   ->  PSO.m karşılığı
# =====================================================================
def PSO(obj_function, bounds, pop_size, max_iter, c1, c2, w):
    '''
    pop_size : Sürü (parçacık) boyutu
    max_iter : Maks. iterasyon sayısı
    c1 : Bilişsel katsayı (kişisel en iyiye çekim)
    c2 : Sosyal katsayı (global en iyiye çekim)
    w  : Atalet ağırlığı (inertia weight)
    '''
    print("=== PSO Optimizasyonu Başladı ===")

    num_vars = len(bounds)
    lower_bound = np.array([b[0] for b in bounds])
    upper_bound = np.array([b[1] for b in bounds])

    # Başlangıç pozisyonları ve hızları
    pos = lower_bound + (upper_bound - lower_bound) * np.random.rand(pop_size, num_vars)
    vel = np.zeros((pop_size, num_vars))

    pbest_pos = pos.copy()
    pbest_fitness = np.array([obj_function(pos[i, :]) for i in range(pop_size)], dtype=float)

    idx = int(np.argmin(pbest_fitness))
    best_fitness = pbest_fitness[idx]
    best_pos = pbest_pos[idx, :].copy()

    history = np.zeros((max_iter * pop_size, num_vars))
    hist_idx = 0

    for t in range(max_iter):
        for i in range(pop_size):
            # Hız Güncellemesi
            r1 = np.random.rand(num_vars)
            r2 = np.random.rand(num_vars)
            vel[i, :] = (w * vel[i, :]
                         + c1 * r1 * (pbest_pos[i, :] - pos[i, :])
                         + c2 * r2 * (best_pos - pos[i, :]))

            # Pozisyon Güncellemesi
            pos[i, :] = pos[i, :] + vel[i, :]

            # Sınır Kontrolü
            pos[i, :] = np.clip(pos[i, :], lower_bound, upper_bound)

            # Değerlendirme
            fit = obj_function(pos[i, :])
            history[hist_idx, :] = pos[i, :]
            hist_idx += 1

            # Kişisel En İyi (pbest) Güncellemesi
            if fit < pbest_fitness[i]:
                pbest_fitness[i] = fit
                pbest_pos[i, :] = pos[i, :].copy()

            # Global En İyi (gbest) Güncellemesi
            if fit < best_fitness:
                best_fitness = fit
                best_pos = pos[i, :].copy()

    print("Optimizasyon Tamamlandı!")
    print(f"Bulunan En İyi Uygunluk Değeri (f(x*)): {best_fitness:f}")
    return best_fitness, best_pos, history


# =====================================================================
#  GENETIC ALGORITHM (GA)   ->  GA.m karşılığı
# =====================================================================
def _tournament_selection(fitness, k=3):
    # Turnuva Seçilimi (Tournament Selection)
    rand_idx = np.random.randint(0, len(fitness), k)
    best_i = int(np.argmin(fitness[rand_idx]))
    return rand_idx[best_i]


def GA(obj_function, bounds, pop_size, max_iter, crossover_rate, mutation_rate):
    '''
    pop_size : Popülasyon boyutu
    max_iter : Jenerasyon (iterasyon) sayısı
    crossover_rate : Aritmetik çaprazlama oranı
    mutation_rate  : Mutasyon oranı
    '''
    print("=== GA Optimizasyonu Başladı ===")

    num_vars = len(bounds)
    lower_bound = np.array([b[0] for b in bounds])
    upper_bound = np.array([b[1] for b in bounds])

    pop = lower_bound + (upper_bound - lower_bound) * np.random.rand(pop_size, num_vars)
    fitness = np.array([obj_function(pop[i, :]) for i in range(pop_size)], dtype=float)

    history = np.zeros((max_iter * pop_size, num_vars))
    hist_idx = 0

    best_idx = int(np.argmin(fitness))
    best_fitness = fitness[best_idx]
    best_pos = pop[best_idx, :].copy()

    for t in range(max_iter):
        new_pop = np.zeros((pop_size, num_vars))

        for i in range(0, pop_size, 2):
            # Turnuva Seçilimi
            p1 = pop[_tournament_selection(fitness), :]
            p2 = pop[_tournament_selection(fitness), :]

            # Aritmetik Çaprazlama (Crossover)
            if np.random.rand() < crossover_rate:
                alpha = np.random.rand()
                c1 = alpha * p1 + (1 - alpha) * p2
                c2 = alpha * p2 + (1 - alpha) * p1
            else:
                c1 = p1.copy()
                c2 = p2.copy()

            # Mutasyon
            if np.random.rand() < mutation_rate:
                c1 = c1 + np.random.randn(num_vars) * 0.1 * (upper_bound - lower_bound)
            if np.random.rand() < mutation_rate:
                c2 = c2 + np.random.randn(num_vars) * 0.1 * (upper_bound - lower_bound)

            # Sınır Kontrolü
            new_pop[i, :] = np.clip(c1, lower_bound, upper_bound)
            if i + 1 < pop_size:
                new_pop[i + 1, :] = np.clip(c2, lower_bound, upper_bound)

        pop = new_pop

        for i in range(pop_size):
            fit = obj_function(pop[i, :])
            fitness[i] = fit
            history[hist_idx, :] = pop[i, :]
            hist_idx += 1

            if fit < best_fitness:
                best_fitness = fit
                best_pos = pop[i, :].copy()

    print("Optimizasyon Tamamlandı!")
    print(f"Bulunan En İyi Uygunluk Değeri (f(x*)): {best_fitness:f}")
    return best_fitness, best_pos, history


# =====================================================================
#  BENCHMARK FONKSİYONLARI
# =====================================================================
def rosenbrock(x):
    # Logaritmik Rosenbrock Banana Fonksiyonu
    return np.log(1 + (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2)


def michalewicz(x, m=10):
    # Michalewicz's Bivariate Function
    term1 = -np.sin(x[0]) * (np.sin(x[0]**2 / np.pi))**(2 * m)
    term2 = -np.sin(x[1]) * (np.sin(2 * x[1]**2 / np.pi))**(2 * m)
    return term1 + term2
