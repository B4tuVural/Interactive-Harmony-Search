import numpy as np

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


def rosenbrock(x):
    # Logaritmik Rosenbrock Banana Fonksiyonu
    return np.log(1 + (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2)

def michalewicz(x, m=10):
    # Michalewicz's Bivariate Function
    term1 = -np.sin(x[0]) * (np.sin(x[0]**2 / np.pi))**(2 * m)
    term2 = -np.sin(x[1]) * (np.sin(2 * x[1]**2 / np.pi))**(2 * m)
    return term1 + term2