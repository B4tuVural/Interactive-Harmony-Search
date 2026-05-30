function [best_fitness, best_pos, history] = GA(obj_func, lower_bound, upper_bound, num_vars, pop_size, max_iter, crossover_rate, mutation_rate)

fprintf('=== GA Optimizasyonu Başladı ===\n');

pop = lower_bound + (upper_bound - lower_bound) .* rand(pop_size, num_vars);
fitness = zeros(pop_size, 1);

history = zeros(max_iter * pop_size, num_vars);
hist_idx = 1;

for i = 1:pop_size
    fitness(i) = obj_func(pop(i,:));
end

[best_fitness, best_idx] = min(fitness);
best_pos = pop(best_idx, :);

for t = 1:max_iter
    new_pop = zeros(pop_size, num_vars);

    for i = 1:2:pop_size
        % Turnuva Seçilimi (Tournament Selection)
        p1 = pop(tournament_selection(fitness), :);
        p2 = pop(tournament_selection(fitness), :);

        % Aritmetik Çaprazlama (Crossover)
        if rand() < crossover_rate
            alpha = rand();
            c1 = alpha*p1 + (1-alpha)*p2;
            c2 = alpha*p2 + (1-alpha)*p1;
        else
            c1 = p1; c2 = p2;
        end

        % Mutasyon
        if rand() < mutation_rate
            c1 = c1 + randn(1, num_vars) * 0.1 * (upper_bound - lower_bound);
        end
        if rand() < mutation_rate
            c2 = c2 + randn(1, num_vars) * 0.1 * (upper_bound - lower_bound);
        end

        % Sınır Kontrolü
        new_pop(i, :) = max(min(c1, upper_bound), lower_bound);
        if i+1 <= pop_size
            new_pop(i+1, :) = max(min(c2, upper_bound), lower_bound);
        end
    end

    pop = new_pop;

    for i = 1:pop_size
        fit = obj_func(pop(i,:));
        fitness(i) = fit;
        history(hist_idx, :) = pop(i,:);
        hist_idx = hist_idx + 1;

        if fit < best_fitness
            best_fitness = fit;
            best_pos = pop(i,:);
        end
    end
end

fprintf('Optimizasyon Tamamlandı!\n');
fprintf('Bulunan En İyi Uygunluk Değeri (f(x*)): %f\n', best_fitness);
end

% Yardımcı Fonksiyon: Turnuva Seçilimi
function idx = tournament_selection(fitness)
k = 3; % Turnuva boyutu
rand_idx = randi(length(fitness), k, 1);
[~, best_i] = min(fitness(rand_idx));
idx = rand_idx(best_i);
end