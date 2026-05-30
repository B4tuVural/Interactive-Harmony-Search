clc, clear, close all;

lower_bound = -5; upper_bound = 5; num_vars = 2;
pop_size = 30; max_iter = 150; 
c1 = 1.5; c2 = 1.5; w = 0.7; % PSO Parametreleri

obj_func = @(x) log(1 + (1 - x(1)).^2 + 100 * (x(2) - x(1).^2).^2); 

[best_fitness, best_pos, history] = PSO(obj_func, lower_bound, upper_bound, num_vars, pop_size, max_iter, c1, c2, w);

% --- Görselleştirme ---
figure('Name', 'PSO Arama Uzayı - Rosenbrock', 'Color', 'w', 'Position', [150, 100, 850, 650]);
[X, Y] = meshgrid(linspace(-5, 5, 60)); 
Z = log(1 + (1 - X).^2 + 100 .* (Y - X.^2).^2);

s = surfc(X, Y, Z); s(1).FaceColor = 'w'; s(1).EdgeColor = [0.4 0.4 0.4]; s(2).LineWidth = 1; 
colormap(gray); hold on; grid on;

ax = gca; z_floor = ax.ZLim(1); 
set(gca, 'FontSize', 16, 'FontWeight', 'bold', 'Box', 'off', 'LineWidth', 1);
ax.XTick = [-5, 0, 5]; ax.YTick = [-5, 0, 5]; ax.ZTick = [0, 10, 20]; view([-40, 25]);

plot_step = pop_size; % Her iterasyonda tüm sürüyü çiz
total_hist = size(history, 1);
for i = 1:plot_step:total_hist
    idx_end = min(i+plot_step-1, total_hist);
    plot3(history(i:idx_end, 1), history(i:idx_end, 2), repmat(z_floor, idx_end-i+1, 1), 'g.', 'MarkerSize', 6);
    drawnow;
end

plot3(best_pos(1), best_pos(2), z_floor, 'p', 'MarkerSize', 15, 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'r'); hold off;