%% 控制系统辨识与智能优化PID控制实验
clear all; close all; clc;

%% 1. 加载数据与系统辨识
% 加载数据
data = readtable('temperature.csv');
time = data.time;
temp = data.temperature;
voltage = data.volte;

% 绘制原始数据曲线
figure;
plot(time, temp, 'r-', 'LineWidth', 1.5); hold on;
plot(time, voltage*10, 'b--', 'LineWidth', 1); % 电压放大10倍以便观察
legend('温度', '加热电压×10');
xlabel('时间（s）'); ylabel('信号值');
title('加热炉阶跃响应曲线');
grid on;

% 检查电压是否为恒定值
if all(voltage == voltage(1))
    warning('电压数据为恒定值，可能影响系统辨识准确性');
    % 使用假设的阶跃输入幅度
    step_input = 3.5; 
else
    % 计算实际阶跃输入幅度
    step_input = max(voltage) - min(voltage);
end

% 寻找稳态值和初始值
initial_temp = temp(1);
steady_state_temp = temp(end);
fprintf('初始温度: %.4f ℃\n', initial_temp);
fprintf('稳态温度: %.4f ℃\n', steady_state_temp);

% 两点法辨识一阶系统模型
% 计算系统增益K = (稳态输出 - 初始输出)/输入变化
K = (steady_state_temp - initial_temp) / step_input;
fprintf('系统增益K: %.4f\n', K);

% 寻找达到63.2%稳态值的时间
target_temp = initial_temp + 0.632*(steady_state_temp - initial_temp);
[~, idx] = min(abs(temp - target_temp));
t_63 = time(idx);
fprintf('达到63.2%%稳态值的时间: %.2f s\n', t_63);

% 计算时间常数T = t_63 - t_delay
% 寻找延迟时间t_delay（温度开始明显上升的时间）
temp_change = diff(temp);
[~, delay_idx] = find(abs(temp_change) > 0.05, 1);
t_delay = time(delay_idx);
fprintf('延迟时间: %.2f s\n', t_delay);

T = t_63 - t_delay;
fprintf('时间常数T: %.2f s\n', T);

% 构建辨识出的传递函数
sys_identified = tf(K, [T, 1]);
fprintf('辨识出的传递函数: %.4f/(%.2f*s + 1)\n', K, T);

%% 2. 基于智能算法的PID参数优化
% 设置目标温度
setpoint = 35;

% 2.1 遗传算法优化PID参数
fprintf('\n==== 开始遗传算法优化PID参数 ====\n');
[Kp_GA, Ki_GA, Kd_GA] = GA_PID_Optimization(sys_identified, setpoint, time, temp);
fprintf('遗传算法优化结果: Kp=%.4f, Ki=%.4f, Kd=%.4f\n', Kp_GA, Ki_GA, Kd_GA);

% 2.2 蚁群算法优化PID参数
fprintf('\n==== 开始蚁群算法优化PID参数 ====\n');
[Kp_ACO, Ki_ACO, Kd_ACO] = ACO_PID_Optimization(sys_identified, setpoint, time, temp);
fprintf('蚁群算法优化结果: Kp=%.4f, Ki=%.4f, Kd=%.4f\n', Kp_ACO, Ki_ACO, Kd_ACO);

% 2.3 模拟退火算法优化PID参数
fprintf('\n==== 开始模拟退火算法优化PID参数 ====\n');
[Kp_SA, Ki_SA, Kd_SA] = SA_PID_Optimization(sys_identified, setpoint, time, temp);
fprintf('模拟退火算法优化结果: Kp=%.4f, Ki=%.4f, Kd=%.4f\n', Kp_SA, Ki_SA, Kd_SA);

% 2.4 粒子群优化算法(PSO)优化PID参数
fprintf('\n==== 开始粒子群优化算法(PSO)优化PID参数 ====\n');
[Kp_PSO, Ki_PSO, Kd_PSO] = PSO_PID_Optimization(sys_identified, setpoint, time, temp);
fprintf('PSO算法优化结果: Kp=%.4f, Ki=%.4f, Kd=%.4f\n', Kp_PSO, Ki_PSO, Kd_PSO);

%% 3. 构建闭环系统并仿真
% 构建PID控制器
C_GA = tf([Kd_GA, Kp_GA, Ki_GA], [1, 0]);
C_ACO = tf([Kd_ACO, Kp_ACO, Ki_ACO], [1, 0]);
C_SA = tf([Kd_SA, Kp_SA, Ki_SA], [1, 0]);
C_PSO = tf([Kd_PSO, Kp_PSO, Ki_PSO], [1, 0]);

% 构建闭环系统
sys_closed_GA = feedback(C_GA * sys_identified, 1);
sys_closed_ACO = feedback(C_ACO * sys_identified, 1);
sys_closed_SA = feedback(C_SA * sys_identified, 1);
sys_closed_PSO = feedback(C_PSO * sys_identified, 1);

% 仿真时间
sim_time = 0:0.5:max(time);

% 仿真系统响应
[y_GA, t] = step(setpoint*sys_closed_GA, sim_time);
[y_ACO, t] = step(setpoint*sys_closed_ACO, sim_time);
[y_SA, t] = step(setpoint*sys_closed_SA, sim_time);
[y_PSO, t] = step(setpoint*sys_closed_PSO, sim_time);

%% 4. 计算性能指标并比较
% 4.1 计算GA优化PID的性能指标
[overshoot_GA, rise_time_GA, settling_time_GA, steady_error_GA] = calculate_performance(y_GA, t, setpoint);
% 4.2 计算ACO优化PID的性能指标
[overshoot_ACO, rise_time_ACO, settling_time_ACO, steady_error_ACO] = calculate_performance(y_ACO, t, setpoint);
% 4.3 计算SA优化PID的性能指标
[overshoot_SA, rise_time_SA, settling_time_SA, steady_error_SA] = calculate_performance(y_SA, t, setpoint);
% 4.4 计算PSO优化PID的性能指标
[overshoot_PSO, rise_time_PSO, settling_time_PSO, steady_error_PSO] = calculate_performance(y_PSO, t, setpoint);

% 输出性能比较
fprintf('\n==== 性能指标比较 ====\n');
fprintf('算法\t\t超调量(%%)\t上升时间(s)\t调节时间(s)\t稳态误差(℃)\n');
fprintf('遗传算法\t%.2f\t\t%.2f\t\t%.2f\t\t%.4f\n', ...
    overshoot_GA, rise_time_GA, settling_time_GA, steady_error_GA);
fprintf('蚁群算法\t%.2f\t\t%.2f\t\t%.2f\t\t%.4f\n', ...
    overshoot_ACO, rise_time_ACO, settling_time_ACO, steady_error_ACO);
fprintf('模拟退火\t%.2f\t\t%.2f\t\t%.2f\t\t%.4f\n', ...
    overshoot_SA, rise_time_SA, settling_time_SA, steady_error_SA);
fprintf('PSO算法\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.4f\n', ...
    overshoot_PSO, rise_time_PSO, settling_time_PSO, steady_error_PSO);

%% 5. 绘制闭环响应比较图
figure;
plot(t, y_GA, 'r-', 'LineWidth', 2); hold on;
plot(t, y_ACO, 'g--', 'LineWidth', 2);
plot(t, y_SA, 'b-.', 'LineWidth', 2);
plot(t, y_PSO, 'm:', 'LineWidth', 2);
plot(t, setpoint*ones(size(t)), 'k:', 'LineWidth', 1.5);
legend('GA优化PID', 'ACO优化PID', 'SA优化PID', 'PSO优化PID', '设定值');
xlabel('时间 (s)'); ylabel('温度 (℃)');
title('不同算法优化PID控制器的闭环响应比较');
grid on;

%% 粒子群优化(PSO)算法优化PID控制器参数
function [Kp_opt, Ki_opt, Kd_opt] = PSO_PID_Optimization(sys_identified, setpoint, time_data, temp_data)
    % 设置PSO算法参数
    swarm_size = 50;             % 粒子群大小
    max_iterations = 100;        % 最大迭代次数
    c1 = 2;                      % 个体学习因子
    c2 = 2;                      % 社会学习因子
    w_max = 0.9;                 % 惯性权重最大值
    w_min = 0.4;                 % 惯性权重最小值
    
    % 参数搜索范围 [Kp_min, Ki_min, Kd_min; Kp_max, Ki_max, Kd_max]
    param_bounds = [0, 0, 0; 100, 10, 10];
    
    % 初始化粒子位置和速度
    positions = zeros(swarm_size, 3);
    velocities = zeros(swarm_size, 3);
    
    for i = 1:swarm_size
        % 随机初始化位置
        positions(i,:) = param_bounds(1,:) + rand(1,3) .* (param_bounds(2,:) - param_bounds(1,:));
        
        % 随机初始化速度
        velocities(i,:) = -1 + 2*rand(1,3);  % 速度范围 [-1, 1]
    end
    
    % 初始化个体最优位置和全局最优位置
    personal_best_positions = positions;
    personal_best_fitness = zeros(swarm_size, 1) + inf;
    
    [~, global_best_idx] = min(personal_best_fitness);
    global_best_position = personal_best_positions(global_best_idx,:);
    global_best_fitness = personal_best_fitness(global_best_idx);
    
    % 记录每代最优适应度
    best_fitness_history = zeros(max_iterations, 1);
    
    % 主循环
    for iteration = 1:max_iterations
        % 计算惯性权重（线性递减）
        w = w_max - (w_max - w_min) * iteration / max_iterations;
        
        % 评估每个粒子的适应度
        for i = 1:swarm_size
            Kp = positions(i,1);
            Ki = positions(i,2);
            Kd = positions(i,3);
            
            fitness = pid_fitness([Kp, Ki, Kd], sys_identified, setpoint, time_data, temp_data);
            
            % 更新个体最优
            if fitness < personal_best_fitness(i)
                personal_best_fitness(i) = fitness;
                personal_best_positions(i,:) = positions(i,:);
            end
            
            % 更新全局最优
            if fitness < global_best_fitness
                global_best_fitness = fitness;
                global_best_position = positions(i,:);
            end
        end
        
        % 记录当前迭代的最优适应度
        best_fitness_history(iteration) = global_best_fitness;
        
        % 输出当前迭代结果
        fprintf('Iteration %d: Best Fitness = %.4f, Kp = %.4f, Ki = %.4f, Kd = %.4f\n', ...
            iteration, global_best_fitness, global_best_position(1), ...
            global_best_position(2), global_best_position(3));
        
        % 更新粒子速度和位置
        for i = 1:swarm_size
            % 计算加速度分量
            r1 = rand();
            r2 = rand();
            acceleration1 = c1 * r1 * (personal_best_positions(i,:) - positions(i,:));
            acceleration2 = c2 * r2 * (global_best_position - positions(i,:));
            
            % 更新速度
            velocities(i,:) = w * velocities(i,:) + acceleration1 + acceleration2;
            
            % 更新位置
            positions(i,:) = positions(i,:) + velocities(i,:);
            
            % 确保位置在参数范围内
            positions(i,:) = max(positions(i,:), param_bounds(1,:));
            positions(i,:) = min(positions(i,:), param_bounds(2,:));
        end
    end
    
    % 返回最优参数
    Kp_opt = global_best_position(1);
    Ki_opt = global_best_position(2);
    Kd_opt = global_best_position(3);
    
    % 绘制适应度变化曲线
    figure;
    plot(1:max_iterations, best_fitness_history, 'LineWidth', 2);
    xlabel('迭代次数');
    ylabel('适应度值 (PID性能指标)');
    title('粒子群优化算法(PSO)优化过程');
    grid on;
end

%% 遗传算法（GA）优化PID控制器参数
function [Kp_opt, Ki_opt, Kd_opt] = GA_PID_Optimization(sys_identified, setpoint, time_data, temp_data)
    % 设置遗传算法参数
    population_size = 50;      % 种群大小
    max_generations = 100;     % 最大迭代次数
    crossover_rate = 0.8;      % 交叉概率
    mutation_rate = 0.1;       % 变异概率
    elitism_count = 2;         % 精英个体数量
    
    % 参数搜索范围 [Kp_min, Ki_min, Kd_min; Kp_max, Ki_max, Kd_max]
    param_bounds = [0, 0, 0; 100, 10, 10];
    
    % 初始化种群
    population = zeros(population_size, 3);
    for i = 1:population_size
        population(i,:) = param_bounds(1,:) + rand(1,3) .* (param_bounds(2,:) - param_bounds(1,:));
    end
    
    % 记录每代最优适应度
    best_fitness_history = zeros(max_generations, 1);
    
    % 主循环
    for generation = 1:max_generations
        % 计算适应度
        fitness = zeros(population_size, 1);
        for i = 1:population_size
            Kp = population(i,1);
            Ki = population(i,2);
            Kd = population(i,3);
            fitness(i) = -pid_fitness([Kp, Ki, Kd], sys_identified, setpoint, time_data, temp_data);
        end
        
        % 记录最优适应度和参数
        [max_fitness, max_idx] = max(fitness);
        best_fitness_history(generation) = max_fitness;
        best_params = population(max_idx,:);
        
        % 输出当前迭代结果
        fprintf('Generation %d: Best Fitness = %.4f, Kp = %.4f, Ki = %.4f, Kd = %.4f\n', ...
            generation, max_fitness, best_params(1), best_params(2), best_params(3));
        
        % 选择操作 - 锦标赛选择
        new_population = zeros(population_size, 3);
        
        % 精英保留
        [~, elite_indices] = sort(fitness, 'descend');
        new_population(1:elitism_count,:) = population(elite_indices(1:elitism_count),:);
        
        % 锦标赛选择
        tournament_size = 3;
        for i = (elitism_count+1):population_size
            % 随机选择参赛个体
            tournament_indices = randperm(population_size, tournament_size);
            tournament_fitness = fitness(tournament_indices);
            [~, winner_idx] = max(tournament_fitness);
            winner = population(tournament_indices(winner_idx),:);
            new_population(i,:) = winner;
        end
        
        % 交叉操作
        for i = (elitism_count+1):2:(population_size-1)
            if rand < crossover_rate
                % 算术交叉
                alpha = rand;
                parent1 = new_population(i,:);
                parent2 = new_population(i+1,:);
                new_population(i,:) = alpha * parent1 + (1-alpha) * parent2;
                new_population(i+1,:) = (1-alpha) * parent1 + alpha * parent2;
                
                % 确保参数在范围内
                new_population(i,:) = max(new_population(i,:), param_bounds(1,:));
                new_population(i,:) = min(new_population(i,:), param_bounds(2,:));
                new_population(i+1,:) = max(new_population(i+1,:), param_bounds(1,:));
                new_population(i+1,:) = min(new_population(i+1,:), param_bounds(2,:));
            end
        end
        
        % 变异操作
        for i = (elitism_count+1):population_size
            if rand < mutation_rate
                % 高斯变异
                mutation_factor = 0.1;
                mutation = mutation_factor * randn(1,3) .* (param_bounds(2,:) - param_bounds(1,:));
                new_population(i,:) = new_population(i,:) + mutation;
                
                % 确保参数在范围内
                new_population(i,:) = max(new_population(i,:), param_bounds(1,:));
                new_population(i,:) = min(new_population(i,:), param_bounds(2,:));
            end
        end
        
        % 更新种群
        population = new_population;
    end
    
    % 返回最优参数
    Kp_opt = best_params(1);
    Ki_opt = best_params(2);
    Kd_opt = best_params(3);
    
    % 绘制适应度变化曲线
    figure;
    plot(1:max_generations, -best_fitness_history, 'LineWidth', 2);
    xlabel('迭代次数');
    ylabel('适应度值 (PID性能指标)');
    title('遗传算法优化过程');
    grid on;
end

%% 蚁群算法（ACO）优化PID控制器参数
function [Kp_opt, Ki_opt, Kd_opt] = ACO_PID_Optimization(sys_identified, setpoint, time_data, temp_data)
    % 设置蚁群算法参数
    ant_count = 30;             % 蚂蚁数量
    max_iterations = 50;        % 最大迭代次数
    alpha = 1;                  % 信息素重要程度因子
    beta = 2;                   % 启发式信息重要程度因子
    rho = 0.5;                  % 信息素挥发系数
    Q = 100;                    % 信息素增加强度系数
    
    % 参数搜索范围 [Kp_min, Ki_min, Kd_min; Kp_max, Ki_max, Kd_max]
    param_bounds = [0, 0, 0; 100, 10, 10];
    
    % 将参数空间离散化
    param_divisions = 20;       % 每个参数的离散化级别
    param_intervals = zeros(3, param_divisions);
    for i = 1:3
        param_intervals(i,:) = linspace(param_bounds(1,i), param_bounds(2,i), param_divisions);
    end
    
    % 初始化信息素矩阵
    pheromone = ones(param_divisions, param_divisions, param_divisions);
    
    % 记录每代最优适应度和参数
    best_fitness_history = zeros(max_iterations, 1);
    best_params_history = zeros(max_iterations, 3);
    
    % 主循环
    for iteration = 1:max_iterations
        % 初始化所有蚂蚁的解
        ant_solutions = zeros(ant_count, 3);
        ant_fitness = zeros(ant_count, 1);
        
        % 每只蚂蚁构建解
        for ant = 1:ant_count
            % 选择Kp, Ki, Kd的离散化索引
            kp_idx = 1; ki_idx = 1; kd_idx = 1;
            
            % 计算选择概率
            probability = pheromone .^ alpha;
            
            % 选择Kp索引
            kp_prob = sum(sum(probability, 2), 3);
            kp_prob = kp_prob / sum(kp_prob);
            kp_idx = randsample(param_divisions, 1, true, kp_prob);
            
            % 选择Ki索引
            ki_prob = sum(probability(kp_idx,:,:), 3);
            ki_prob = ki_prob / sum(ki_prob);
            ki_idx = randsample(param_divisions, 1, true, ki_prob);
            
            % 选择Kd索引
            kd_prob = probability(kp_idx, ki_idx, :);
            kd_prob = kd_prob / sum(kd_prob);
            kd_idx = randsample(param_divisions, 1, true, kd_prob);
            
            % 转换为实际参数值
            ant_solutions(ant,1) = param_intervals(1, kp_idx);
            ant_solutions(ant,2) = param_intervals(2, ki_idx);
            ant_solutions(ant,3) = param_intervals(3, kd_idx);
            
            % 计算适应度
            Kp = ant_solutions(ant,1);
            Ki = ant_solutions(ant,2);
            Kd = ant_solutions(ant,3);
            ant_fitness(ant) = pid_fitness([Kp, Ki, Kd], sys_identified, setpoint, time_data, temp_data);
        end
        
        % 记录最优解
        [min_fitness, min_idx] = min(ant_fitness);
        best_fitness_history(iteration) = min_fitness;
        best_params_history(iteration,:) = ant_solutions(min_idx,:);
        
        % 输出当前迭代结果
        fprintf('Iteration %d: Best Fitness = %.4f, Kp = %.4f, Ki = %.4f, Kd = %.4f\n', ...
            iteration, min_fitness, best_params_history(iteration,1), ...
            best_params_history(iteration,2), best_params_history(iteration,3));
        
        % 更新信息素
        pheromone = (1-rho) * pheromone;  % 信息素挥发
        
        % 增加信息素
        for ant = 1:ant_count
            kp_idx = find(param_intervals(1,:) == ant_solutions(ant,1));
            ki_idx = find(param_intervals(2,:) == ant_solutions(ant,2));
            kd_idx = find(param_intervals(3,:) == ant_solutions(ant,3));
            
            % 适应度越小，信息素增加越多
            pheromone(kp_idx, ki_idx, kd_idx) = pheromone(kp_idx, ki_idx, kd_idx) + Q / ant_fitness(ant);
        end
    end
    
    % 返回最优参数
    [~, best_idx] = min(best_fitness_history);
    Kp_opt = best_params_history(best_idx,1);
    Ki_opt = best_params_history(best_idx,2);
    Kd_opt = best_params_history(best_idx,3);
    
    % 绘制适应度变化曲线
    figure;
    plot(1:max_iterations, best_fitness_history, 'LineWidth', 2);
    xlabel('迭代次数');
    ylabel('适应度值 (PID性能指标)');
    title('蚁群算法优化过程');
    grid on;
end

%% 模拟退火算法（SA）优化PID控制器参数
function [Kp_opt, Ki_opt, Kd_opt] = SA_PID_Optimization(sys_identified, setpoint, time_data, temp_data)
    % 设置模拟退火算法参数
    initial_temp = 100;         % 初始温度
    final_temp = 0.1;           % 终止温度
    cooling_rate = 0.95;        % 降温速率
    iterations_per_temp = 50;   % 每个温度下的迭代次数
    
    % 参数搜索范围 [Kp_min, Ki_min, Kd_min; Kp_max, Ki_max, Kd_max]
    param_bounds = [0, 0, 0; 100, 10, 10];
    
    % 初始化当前解
    current_solution = param_bounds(1,:) + rand(1,3) .* (param_bounds(2,:) - param_bounds(1,:));
    current_fitness = pid_fitness(current_solution, sys_identified, setpoint, time_data, temp_data);
    
    % 初始化最优解
    best_solution = current_solution;
    best_fitness = current_fitness;
    
    % 记录优化过程
    temp_history = [];
    best_fitness_history = [];
    
    % 当前温度
    current_temp = initial_temp;
    
    % 主循环
    iteration = 0;
    while current_temp > final_temp
        iteration = iteration + 1;
        
        % 在每个温度下进行多次迭代
        for i = 1:iterations_per_temp
            % 生成新解（在当前解附近进行扰动）
            perturbation = 0.1 * current_temp * randn(1,3) .* (param_bounds(2,:) - param_bounds(1,:));
            new_solution = current_solution + perturbation;
            
            % 确保新解在参数范围内
            new_solution = max(new_solution, param_bounds(1,:));
            new_solution = min(new_solution, param_bounds(2,:));
            
            % 计算新解的适应度
            new_fitness = pid_fitness(new_solution, sys_identified, setpoint, time_data, temp_data);
            
            % 计算适应度差异
            delta_fitness = new_fitness - current_fitness;
            
            % 判断是否接受新解
            if delta_fitness < 0 || rand < exp(-delta_fitness / current_temp)
                current_solution = new_solution;
                current_fitness = new_fitness;
            end
            
            % 更新最优解
            if current_fitness < best_fitness
                best_solution = current_solution;
                best_fitness = current_fitness;
            end
        end
        
        % 记录当前温度和最优适应度
        temp_history = [temp_history, current_temp];
        best_fitness_history = [best_fitness_history, best_fitness];
        
        % 输出当前迭代结果
        fprintf('Iteration %d: Temperature = %.4f, Best Fitness = %.4f, Kp = %.4f, Ki = %.4f, Kd = %.4f\n', ...
            iteration, current_temp, best_fitness, best_solution(1), best_solution(2), best_solution(3));
        
        % 降温
        current_temp = current_temp * cooling_rate;
    end
    
    % 返回最优参数
    Kp_opt = best_solution(1);
    Ki_opt = best_solution(2);
    Kd_opt = best_solution(3);
    
    % 绘制适应度变化曲线
    figure;
    semilogx(temp_history, best_fitness_history, 'LineWidth', 2);
    xlabel('温度 (对数刻度)');
    ylabel('适应度值 (PID性能指标)');
    title('模拟退火算法优化过程');
    grid on;
end

%% PID控制器适应度函数
function J = pid_fitness(params, sys, setpoint, time_data, temp_data)
    Kp = params(1);
    Ki = params(2);
    Kd = params(3);
    
    % 构建PID控制器
    C = tf([Kd, Kp, Ki], [1, 0]);

    sys_cl = feedback(C * sys, 1);  % 闭环
    t = 0:0.5:max(time_data);
    [y, t] = step(setpoint * sys_cl, t);
    
    % 检查系统是否稳定（防止发散响应导致计算错误）
    if any(y > setpoint*2) || any(isnan(y)) || any(isinf(y))
        J = 1000; % 不稳定系统给予高适应度值
        return;
    end
    
    % 计算性能指标
    % 超调量
    [max_y, idx] = max(y);
    overshoot = max(0, (max_y - setpoint)/setpoint*100); % 确保超调量非负
    
    % 上升时间（修正：从10%到90%的时间）
    y_10pct = setpoint * 0.1;
    y_90pct = setpoint * 0.9;
    idx_10 = find(y >= y_10pct, 1, 'first');
    idx_90 = find(y >= y_90pct, 1, 'first');
    
    if ~isempty(idx_10) && ~isempty(idx_90)
        rise_time = t(idx_90) - t(idx_10);
    else
        rise_time = max(t); % 如果未达到10%或90%，设为最大时间
    end
    
    % 调节时间（2%准则）
    steady_idx = find(abs(y - setpoint) < setpoint*0.02, 1, 'last');
    if isempty(steady_idx)
        settling_time = max(t); % 如果未达到稳态，设为最大时间
    else
        settling_time = t(steady_idx);
    end
    
    % 稳态误差
    steady_state_error = abs(y(end) - setpoint);
    
    % 综合适应度函数（越小越好）
    J = overshoot + rise_time/10 + settling_time/10 + steady_state_error*10;
    
    % 添加惩罚项防止不合理参数
    if Kp < 0.1 || Ki < 0.01 || Kd < 0
        J = J + 100; % 惩罚过小的参数
    end
end

%% 计算系统性能指标
function [overshoot, rise_time, settling_time, steady_error] = calculate_performance(y, t, setpoint)
    % 计算超调量
    [max_y, ~] = max(y);
    overshoot = (max_y - setpoint)/setpoint*100;
    
    % 计算上升时间（10%-90%）
    y_10pct = setpoint * 0.1;
    y_90pct = setpoint * 0.9;
    idx_10 = find(y >= y_10pct, 1, 'first');
    idx_90 = find(y >= y_90pct, 1, 'first');
    
    if ~isempty(idx_10) && ~isempty(idx_90)
        rise_time = t(idx_90) - t(idx_10);
    else
        rise_time = max(t); % 如果未达到10%或90%，设为最大时间
    end
    
    % 计算调节时间（2%准则）
    steady_idx = find(abs(y - setpoint) < setpoint*0.02, 1, 'last');
    if isempty(steady_idx)
        settling_time = max(t); % 如果未达到稳态，设为最大时间
    else
        settling_time = t(steady_idx);
    end
    
    % 计算稳态误差
    steady_error = abs(y(end) - setpoint);
end