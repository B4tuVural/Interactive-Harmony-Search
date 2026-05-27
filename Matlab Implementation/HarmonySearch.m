function [best_fitness, best_harmony, history] = HarmonySearch(obj_func, lower_bound, upper_bound, num_vars, HMS, r_accept, r_pa, bw, max_iter)

% [obj_function, lower_bound, upper_bound, num_vars]
% hedef fonksiyon: obj_func, lower_bound: uzayın alt sınırı
% upper_bound: uzayın üst sınırı, num_vars: uzayın eksen sayısı

% HMS :  Harmony Hafızası Boyutu (Harmony Memory Size)
% r_accept : Kabul oranı (Accept Rate)
% r_pa :  Perde Ayarlama oranı (Pitch Adjusting Rate)
% bw :  bant genişliği çarpanı (Bandwidth ratio)
% max_iter : Maks. iterasyon sayısı değişkeni (Max iteration variable)

fprintf('=== HS Optimizasyonu Başladı ===\n');
fprintf('Harmony Memory başlatılıyor...\n\n');

% (low, upper) aralığında HMS x num_vars boyutlarında rastgele
% Harmony Memory Oluşturuluyor.
HM = lower_bound + (upper_bound - lower_bound) .* rand(HMS, num_vars);

%Harmony Memory cost (bedel) vektörü hesaplanması
HM_fitness = zeros(HMS, 1);
for k = 1:HMS
    HM_fitness(k) = obj_func(HM(k, :)); % HMS x 1  === (k,:) 
end

% [sorted_arr, sort_index] = sort() 
[HM_fitness, fitness_indexes] = sort(HM_fitness); 
HM = HM(fitness_indexes, :); % satırları indexe göre sıralar.

% GÖRSELLEŞTİRME İÇİN EKLENDİ: Taranan noktaları tutacak matris
history = zeros(max_iter, num_vars);

t = 1;
while (t <= max_iter)
    new_harmony = zeros(1, num_vars);

    for i = 1:num_vars
        if rand < r_accept
            rand_idx = randi(HMS);
            new_harmony(i) = HM(rand_idx, i);

            if rand < r_pa
                new_harmony(i) = new_harmony(i) + (2*rand - 1) * bw;
            end
       
        else
            new_harmony(i) = lower_bound + (upper_bound - lower_bound) * rand;
        end % end if
    end % end for
    
    new_fitness = obj_func(new_harmony);
    
    % GÖRSELLEŞTİRME İÇİN EKLENDİ: Üretilen yeni adayı geçmişe kaydet
    history(t, :) = new_harmony;

    if new_fitness < HM_fitness(end)
        % Kötü olanı çıkar, yenisini ekle
        HM(end, :) = new_harmony;
        HM_fitness(end) = new_fitness;

       [HM_fitness, fitness_indexes] = sort(HM_fitness); 
       HM = HM(fitness_indexes, :); % satırları indexe göre sıralar.
    end
    
    t = t+1; % while iterasyonu artırma
end %end while 

% Çıkış değişkenlerinin ayarlanması
best_harmony = HM(1, :);
best_fitness = HM_fitness(1);

% Sonucu ekrana yazdır
fprintf('Optimizasyon Tamamlandı!\n');
fprintf('Bulunan En İyi Uygunluk Değeri (f(x*)): %f\n', best_fitness);

end %function end