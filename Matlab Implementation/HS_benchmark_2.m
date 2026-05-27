clc, clear, close all;

% Parametreler (Michalewicz Fonksiyonu için)
lower_bound = 0;
upper_bound = pi;
num_vars = 2;

HMS = 30;
r_accept = 0.85;
r_pa = 0.5;
bw = 0.15;
max_iter = 5000;

% Michalewicz's Bivariate Function
obj_func = @(x) -sin(x(1)) * (sin(x(1)^2 / pi))^20 - sin(x(2)) * (sin(2 * x(2)^2 / pi))^20;

% Algoritmayı çalıştır ve geçmişi (history) al
[best_fitness, best_harmony, history] = HarmonySearch(obj_func, lower_bound, upper_bound, num_vars, HMS, r_accept, r_pa, bw, max_iter);

% Sonuçları Yazdırma
fprintf('\n--- Optimizasyon Sonucu ---\n');
fprintf('Bulunan En İyi x değeri: %f\n', best_harmony(1));
fprintf('Bulunan En İyi y değeri: %f\n', best_harmony(2));
fprintf('Minimum Fonksiyon Değeri f(x,y): %f\n', best_fitness);

%% --- 3D GÖRSELLEŞTİRME (Yeşil Tarama Animasyonlu ve Monokromatik) ---

figure('Name', 'HS Adım Adım Arama Uzayı Taraması - Michalewicz', 'Color', 'w', 'Position', [150, 100, 850, 650]);

% 1. Yüzey için ızgara oluştur (Michalewicz keskin olduğu için ızgara sayısını 80 yaptık)
[X, Y] = meshgrid(linspace(0, pi, 80)); 
% Matris işlemleri için obj_func denkleminin eleman bazlı (.* ve .^) hali:
Z = -sin(X) .* (sin(X.^2 / pi)).^20 - sin(Y) .* (sin(2 .* Y.^2 / pi)).^20;

% 2. Yüzeyi çiz (Yüzey içi beyaz, çizgiler gri, dipteki konturlar siyah tonlarında)
s = surfc(X, Y, Z);
s(1).FaceColor = 'w'; 
s(1).EdgeColor = [0.4 0.4 0.4]; 
s(2).LineWidth = 1; 
colormap(gray); 
hold on;

% 3. Eksen ve Görünüm Ayarları
ax = gca;
z_floor = ax.ZLim(1); % Konturların ve animasyonun yerleştirileceği zemin

% Eksenleri görseldeki gibi kalın, sade yapıyoruz ve pi değerlerine göre ayarlıyoruz
set(gca, 'FontSize', 16, 'FontWeight', 'bold', 'Box', 'off', 'LineWidth', 1);
ax.XTick = [0, pi/2, pi];
ax.XTickLabel = {'0', '\pi/2', '\pi'};
ax.YTick = [0, pi/2, pi];
ax.YTickLabel = {'0', '\pi/2', '\pi'};
xlabel(''); ylabel(''); zlabel(''); 

view([-35, 45]); % Bu fonksiyonun derin vadilerini göstermek için biraz daha yüksek bir açı
grid on;

% 4. NOKTALARI ADIM ADIM TARAMA (Animasyon Kısmı - Yeşil Noktalar)
fprintf('\nGrafik üzerinde adım adım tarama animasyonu başlatılıyor...\n');

plot_step = 125; % Hız ayarı: Her frame'de çizilecek nokta sayısı
for i = 1:plot_step:max_iter
    idx_end = min(i+plot_step-1, max_iter);
    
    % Noktaları tabanda sırayla tarayarak çizdir ('g.' yeşil noktalar)
    plot3(history(i:idx_end, 1), history(i:idx_end, 2), repmat(z_floor, idx_end-i+1, 1), 'g.', 'MarkerSize', 4);
    
    % MATLAB ekranını anlık güncelleyerek animasyon efekti ver
    drawnow;
end

% 5. En son bulunan optimum noktayı bariz şekilde işaretle
plot3(best_harmony(1), best_harmony(2), z_floor, 'p', 'MarkerSize', 15, 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'r');

hold off;
fprintf('Görselleştirme tamamlandı!\n');