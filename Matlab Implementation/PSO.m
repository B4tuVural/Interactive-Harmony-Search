function [best_fitness, best_pos, history] = PSO(obj_func, lower_bound, upper_bound, num_vars, pop_size, max_iter, c1, c2, w)

fprintf('=== PSO Optimizasyonu Başladı ===\n');

% Başlangıç pozisyonları ve hızları
pos = lower_bound + (upper_bound - lower_bound) .* rand(pop_size, num_vars);
vel = zeros(pop_size, num_vars);

pbest_pos = pos;
pbest_fitness = zeros(pop_size, 1);

for i = 1:pop_size
    pbest_fitness(i) = obj_func(pos(i,:));
end

[best_fitness, idx] = min(pbest_fitness);
best_pos = pbest_pos(idx, :);

history = zeros(max_iter * pop_size, num_vars);
hist_idx = 1;

for t = 1:max_iter
    for i = 1:pop_size
        % Hız Güncellemesi
        r1 = rand(1, num_vars); 
        r2 = rand(1, num_vars);
        vel(i,:) = w * vel(i,:) + c1 * r1 .* (pbest_pos(i,:) - pos(i,:)) + c2 * r2 .* (best_pos - pos(i,:));

        % Pozisyon Güncellemesi
        pos(i,:) = pos(i,:) + vel(i,:);

        % Sınır Kontrolü
        pos(i,:) = max(pos(i,:), lower_bound);
        pos(i,:) = min(pos(i,:), upper_bound);

        % Değerlendirme
        fit = obj_func(pos(i,:));
        history(hist_idx, :) = pos(i,:);
        hist_idx = hist_idx + 1;

        % Kişisel En İyi (pbest) Güncellemesi
        if fit < pbest_fitness(i)
            pbest_fitness(i) = fit;
            pbest_pos(i,:) = pos(i,:);
        end

        % Global En İyi (gbest) Güncellemesi
        if fit < best_fitness
            best_fitness = fit;
            best_pos = pos(i,:);
        end
    end
end

fprintf('Optimizasyon Tamamlandı!\n');
fprintf('Bulunan En İyi Uygunluk Değeri (f(x*)): %f\n', best_fitness);
end