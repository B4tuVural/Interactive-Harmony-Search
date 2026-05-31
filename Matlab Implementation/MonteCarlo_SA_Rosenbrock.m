clc, clear, close all;
% =====================================================================
%  MONTE CARLO SIMULASYONU
%  Algoritma : SA (Simulated Annealing)
%  Benchmark : Logaritmik Rosenbrock
%
%  Bu betik SA.m fonksiyonunu N_runs kez bagimsiz olarak calistirir.
%  "Ulasmak istedigimiz cozume" (teorik global minimum) gore:
%       en yakin  -> EN IYI
%       en uzak   -> EN KOTU
%  Ayrica tum kosularin ORTALAMA ve STANDART SAPMA degerleri raporlanir.
%  GEREKLI: SA.m ayni klasorde bulunmalidir.
% =====================================================================

N_runs = 100;          % Monte Carlo kosu (simulasyon) sayisi

% --- Arama uzayi ve hedef ---
lower_bound = -5;
upper_bound = 5;
num_vars = 2;

obj_func = @(x) log(1 + (1 - x(1)).^2 + 100 * (x(2) - x(1).^2).^2);
target   = 0;        % Teorik global minimum f_min
opt_pos  = [1, 1];   % Teorik optimum konum (x*, y*)

% --- SA parametreleri ---
T0 = 100;
alpha = 0.99;
max_iter = 5000;

% --- Monte Carlo dongusu ---
fprintf('=== Monte Carlo: SA / Logaritmik Rosenbrock ===\n');
fprintf('%d bagimsiz kosu calistiriliyor...\n', N_runs);

fits = zeros(N_runs, 1);
positions = zeros(N_runs, num_vars);

for k = 1:N_runs
    % evalc, algoritmanin icindeki fprintf ciktilarini bastirir
    [~, bf, bp] = evalc('SA(obj_func, lower_bound, upper_bound, num_vars, T0, alpha, max_iter)');
    fits(k) = bf;
    positions(k, :) = bp;
end

% --- Istatistikler (hedefe yakinliga gore) ---
errors = abs(fits - target);          % hedefe uzaklik (f cinsinden)
[~, best_idx]  = min(errors);         % en yakin -> en iyi
[~, worst_idx] = max(errors);         % en uzak  -> en kotu

best_fit   = fits(best_idx);
worst_fit  = fits(worst_idx);
mean_fit   = mean(fits);
std_fit    = std(fits);
best_pos_mc  = positions(best_idx, :);
worst_pos_mc = positions(worst_idx, :);

% --- Sonuc raporu ---
fprintf('\n--------------------------------------------------\n');
fprintf(' SONUCLAR (SA / Logaritmik Rosenbrock, N = %d)\n', N_runs);
fprintf('--------------------------------------------------\n');
fprintf(' Teorik hedef f_min        : %.5f  @ (%.4f, %.4f)\n', target, opt_pos(1), opt_pos(2));
fprintf(' EN IYI  f* (en yakin)     : %.6g  @ (%.4f, %.4f)\n', best_fit, best_pos_mc(1), best_pos_mc(2));
fprintf(' EN KOTU f* (en uzak)      : %.6g  @ (%.4f, %.4f)\n', worst_fit, worst_pos_mc(1), worst_pos_mc(2));
fprintf(' ORTALAMA f*               : %.6g\n', mean_fit);
fprintf(' STANDART SAPMA            : %.6g\n', std_fit);
fprintf('--------------------------------------------------\n');

% --- Dagilim grafigi (histogram) ---
figure('Name', 'Monte Carlo - SA / Logaritmik Rosenbrock', 'Color', 'w', ...
       'Position', [200, 150, 800, 550]);
histogram(fits, 'FaceColor', [0.2 0.5 0.8], 'EdgeColor', 'w');
hold on; grid on;
yl = ylim;
% Hedef (teorik minimum) cizgisi
plot([target target], yl, 'g--', 'LineWidth', 2);
% Ortalama cizgisi
plot([mean_fit mean_fit], yl, 'r-', 'LineWidth', 2);
hold off;
set(gca, 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Bulunan f* (uygunluk degeri)');
ylabel('Frekans (kosu sayisi)');
title(sprintf('SA / Logaritmik Rosenbrock  |  ort=%.4g, std=%.4g  (N=%d)', ...
      mean_fit, std_fit, N_runs));
legend({'f* dagilimi', 'Hedef (f_{min})', 'Ortalama'}, 'Location', 'best');

fprintf('Grafik olusturuldu.\n');
