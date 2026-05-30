clc, clear, close all;

% Parametreler
lower_bound = 0;
upper_bound = pi;
num_vars = 2;

pop_size = 40; 
max_iter = 100;
crossover_rate = 0.8; 
mutation_rate = 0.1; 

% Michalewicz Fonksiyonu
obj_func = @(x) -sin(x(1)) * (sin(x(1)^2 / pi))^20 - sin(x(2)) * (sin(2 * x(2)^2 / pi))^20;

% Algoritmayı çalıştır
[best_fitness, best_pos, history] = GA(obj_func, lower_bound, upper_bound, num_vars, pop_size, max_iter, crossover_rate, mutation_rate);

% --- Görselleştirme ---
figure('Name', 'GA Arama Uzayı - Michalewicz', 'Color', 'w', 'Position', [150, 100, 850, 650]);
[X, Y] = meshgrid(linspace(0, pi, 80)); 
Z = -sin(X) .* (sin(X.^2 / pi)).^20 - sin(Y) .* (sin(2 .* Y.^2 / pi)).^20;

s = surfc(X, Y, Z); 
s(1).FaceColor = 'w'; 
s(1).EdgeColor = [0.4 0.4 0.4]; 
s(2).LineWidth = 1; 
colormap(gray); 
hold on; grid on;

ax = gca; 
z_floor = ax.ZLim(1); 
set(gca, 'FontSize', 16, 'FontWeight', 'bold', 'Box', 'off', 'LineWidth', 1);
ax.XTick = [0, pi/2, pi]; ax.XTickLabel = {'0', '\pi/2', '\pi'};
ax.YTick = [0, pi/2, pi]; ax.YTickLabel = {'0', '\pi/2', '\pi'};
xlabel(''); ylabel(''); zlabel(''); 
view([-35, 45]);

fprintf('\nGrafik üzerinde adım adım tarama animasyonu başlatılıyor...\n');
plot_step = pop_size; % Jenerasyon jenerasyon (popülasyon boyutu kadar) tarama animasyonu
total_hist = size(history, 1);
for i = 1:plot_step:total_hist
    idx_end = min(i+plot_step-1, total_hist);
    plot3(history(i:idx_end, 1), history(i:idx_end, 2), repmat(z_floor, idx_end-i+1, 1), 'g.', 'MarkerSize', 6);
    drawnow;
end

plot3(best_pos(1), best_pos(2), z_floor, 'p', 'MarkerSize', 15, 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'r'); 
hold off;
fprintf('Görselleştirme tamamlandı!\n');