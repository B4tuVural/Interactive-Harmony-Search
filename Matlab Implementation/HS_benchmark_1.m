clc, clear, close all;

% Parametreler
lower_bound = -5;
upper_bound = 5;
num_vars = 2;

HMS = 30;
r_accept = 0.85;
r_pa = 0.5;
bw = 0.15;
max_iter = 5000;

% Logaritmik Rosenbrock Banana Fonksiyonu
obj_func = @(x) log(1 + (1 - x(1)).^2 + 100 * (x(2) - x(1).^2).^2); 

% Algoritmayı çalıştır ve geçmişi (history) al
[best_fitness, best_harmony, history] = HarmonySearch(obj_func, lower_bound, upper_bound, num_vars, HMS, r_accept, r_pa, bw, max_iter);

% Sonuçları Yazdırma
fprintf('\n--- Optimizasyon Sonucu ---\n');
fprintf('Bulunan En İyi x değeri: %f\n', best_harmony(1));
fprintf('Bulunan En İyi y değeri: %f\n', best_harmony(2));
fprintf('Minimum Fonksiyon Değeri f(x,y): %f\n', best_fitness);

%% --- 3D GÖRSELLEŞTİRME (Yüklediğin Görseldeki Estetik ve Animasyon) ---

figure('Name', 'HS Adım Adım Arama Uzayı Taraması', 'Color', 'w', 'Position', [150, 100, 850, 650]);

% 1. Yüzey için ızgara oluştur (Görseldeki gibi biraz geniş aralıklı ağ)
[X, Y] = meshgrid(linspace(-5, 5, 60)); 
Z = log(1 + (1 - X).^2 + 100 .* (Y - X.^2).^2);

% 2. Yüzeyi çiz (Yüzey içi beyaz, çizgiler gri, dipteki konturlar siyah tonlarında)
s = surfc(X, Y, Z);
s(1).FaceColor = 'w'; 
s(1).EdgeColor = [0.4 0.4 0.4]; 
s(2).LineWidth = 1; 
colormap(gray); 
hold on;

% 3. Eksen ve Görünüm Ayarları (Tamamen görseldeki kutusuz, sadece kalın rakamlı yapı)
ax = gca;
z_floor = ax.ZLim(1); % Dip noktayı bul (konturların olduğu taban)

set(gca, 'FontSize', 16, 'FontWeight', 'bold', 'Box', 'off', 'LineWidth', 1);
ax.XTick = [-5, 0, 5];
ax.YTick = [-5, 0, 5];
ax.ZTick = [0, 10, 20];
xlabel(''); ylabel(''); zlabel(''); % İsim etiketlerini kaldırarak görseldeki sadeliği yakaladık

view([-40, 25]); % Kamera açısı
grid on;

% 4. NOKTALARI ADIM ADIM TARAMA (Animasyon Kısmı)
fprintf('\nGrafik üzerinde adım adım tarama animasyonu başlatılıyor...\n');

% Görseldeki toz bulutu tarzı için 'k.' (küçük siyah noktalar) kullanıyoruz
plot_step = 100; % Hız ayarı: Her frame'de çizilecek nokta sayısı
for i = 1:plot_step:max_iter
    idx_end = min(i+plot_step-1, max_iter);
    
    % Noktaları tabanda sırayla tarayarak çizdir
    plot3(history(i:idx_end, 1), history(i:idx_end, 2), repmat(z_floor, idx_end-i+1, 1), 'g.', 'MarkerSize', 6);
    
    % MATLAB ekranını anlık güncelleyerek animasyon efekti ver
    drawnow;
end

% 5. En son bulunan optimum noktayı bariz şekilde işaretle
plot3(best_harmony(1), best_harmony(2), z_floor, 'p', 'MarkerSize', 15, 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'r');

hold off;
fprintf('Görselleştirme tamamlandı!\n');