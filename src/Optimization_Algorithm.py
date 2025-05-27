import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple
import os
import sys
DEVICE = torch.device('cpu')
def get_resource_path(relative_path):
    """获取打包后文件的路径"""
    if hasattr(sys, '_MEIPASS'):
        # 如果是打包后的环境
        base_path = sys._MEIPASS
    else:
        # 如果是开发环境
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

Project_folder = get_resource_path("model\\") 
model_pth = "shuzhi_0.pth"      # 模型文件名
net = torch.load(Project_folder  + model_pth, map_location=DEVICE,weights_only=False)
random_indices = np.array([222,45,987,123,456,789,234,567,90,678,345,912,111,333,444,555,666,777,888,999])
# 2455
class GAOptimizer:
    def __init__(self, excel_path: str, population_size: int = 30, max_iter: int = 10):
        """初始化遗传算法优化器"""
        # 读取Excel数据
        self.df = pd.read_excel(excel_path)  
        self.param_columns = self.df.columns[1:6]  
        self.value_columns = self.df.columns[6:13]
        
        # 遗传算法参数
        self.population_size = population_size
        self.max_iter = max_iter
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        
        # 参数范围
        self.param_bounds = [(self.df[col].min(), self.df[col].max()) for col in self.param_columns]
        
        # 存储最佳分数历史
        self.best_scores_history = []

    def _initialize_population(self) -> np.ndarray:
        """初始化种群"""
        population = np.zeros((self.population_size, 5))
        
        # 从现有参数中随机采样初始化
        existing_params = [tuple(np.round(row[self.param_columns].values.astype(float), 3)) 
                         for _, row in self.df.iterrows()]
        indices = np.random.choice(len(existing_params), self.population_size)
        # indices = random_indices[:self.population_size]  
        for i, idx in enumerate(indices):
            population[i] = existing_params[idx]
            
        return population

    def _evaluate(self, position: np.ndarray) -> float:
        """评估适应度"""
        position = np.array(position, dtype=np.float32)
        position = torch.tensor(position)
        position = torch.unsqueeze(position, dim=0)
        position = torch.unsqueeze(position, dim=0)
        pred_0 = net(position).detach().numpy()
        return pred_0[0, 4]

    def _apply_constraints(self, position):
        """应用约束条件"""
        position[0] = np.clip(position[0], 3, 8)
        position[1] = np.clip(position[1], 2 * position[0], 3.25 * position[0])
        position[2] = np.clip(position[2], 10, 80 - position[1])
        position[3] = np.clip(position[3], 10, 90 - position[2] - position[1])
        position[4] = 100 - position[3] - position[2] - position[1]
        return position

    def _selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """选择操作 - 使用轮盘赌选择"""
        # 转换为最小化问题
        inverse_fitness = 1.0 / (fitness + 1e-6)
        prob = inverse_fitness / inverse_fitness.sum()
        selected_indices = np.random.choice(len(population), size=len(population), p=prob)
        return population[selected_indices]

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """交叉操作 - 均匀交叉"""
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
            
        mask = np.random.rand(5) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2

    def _mutation(self, individual: np.ndarray) -> np.ndarray:
        """变异操作"""
        for i in range(5):
            if np.random.rand() < self.mutation_rate:
                # 在当前值附近进行小范围变异
                individual[i] += np.random.normal(0, 0.1 * (self.param_bounds[i][1] - self.param_bounds[i][0]))
        return self._apply_constraints(individual)

    def optimize(self) -> Tuple[np.ndarray, float]:
        """执行优化过程"""
        population = self._initialize_population()
        fitness = np.array([self._evaluate(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx].copy()
        best_score = fitness[best_idx]
        
        self.best_scores_history.append(best_score)
        
        for _ in range(self.max_iter):
            # 选择
            selected = self._selection(population, fitness)
            
            # 交叉
            new_population = []
            for i in range(0, len(selected), 2):
                if i+1 >= len(selected):
                    new_population.append(selected[i])
                    continue
                    
                child1, child2 = self._crossover(selected[i], selected[i+1])
                new_population.extend([child1, child2])
            
            # 变异
            population = np.array([self._mutation(ind) for ind in new_population])
            
            # 评估
            fitness = np.array([self._evaluate(ind) for ind in population])
            
            # 更新最佳个体
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_score:
                best_individual = population[current_best_idx].copy()
                best_score = fitness[current_best_idx]
            
            self.best_scores_history.append(best_score)
        
        plt.plot(self.best_scores_history, marker='o', label=f'GA_{self.population_size}')
        plt.xlabel('Generation')
        plt.ylabel('Best Score')
        plt.title('Genetic Algorithm Optimization Progress')
        plt.legend()
        
        return best_individual, best_score
    

class SAOptimizer:
    def __init__(self, excel_path: str, max_iter: int = 100):
        """初始化模拟退火优化器"""
        # 读取Excel数据
        self.df = pd.read_excel(excel_path)  
        self.param_columns = self.df.columns[1:6]  
        self.value_columns = self.df.columns[6:13]
        
        # 模拟退火参数
        self.max_iter = max_iter
        self.initial_temp = 100.0
        self.final_temp = 0.1
        self.cooling_rate = 0.95
        
        # 参数范围
        self.param_bounds = [(self.df[col].min(), self.df[col].max()) for col in self.param_columns]
        
        # 存储最佳分数历史
        self.best_scores_history = []

    def _initialize_solution(self) -> np.ndarray:
        """初始化解"""
        existing_params = [tuple(np.round(row[self.param_columns].values.astype(float), 3)) 
                         for _, row in self.df.iterrows()]
        idx = np.random.choice(len(existing_params))
        # idx=222
        return np.array(existing_params[idx])

    def _evaluate(self, position: np.ndarray) -> float:
        """评估适应度"""
        position = np.array(position, dtype=np.float32)
        position = torch.tensor(position)
        position = torch.unsqueeze(position, dim=0)
        position = torch.unsqueeze(position, dim=0)
        pred_0 = net(position).detach().numpy()
        return pred_0[0, 4]

    def _apply_constraints(self, position):
        """应用约束条件"""
        position[0] = np.clip(position[0], 3, 8)
        position[1] = np.clip(position[1], 2 * position[0], 3.25 * position[0])
        position[2] = np.clip(position[2], 10, 80 - position[1])
        position[3] = np.clip(position[3], 10, 90 - position[2] - position[1])
        position[4] = 100 - position[3] - position[2] - position[1]
        return position

    def _get_neighbor(self, current_solution: np.ndarray, temp: float) -> np.ndarray:
        """获取邻域解"""
        # 温度越高，变异幅度越大
        scale = temp / self.initial_temp
        neighbor = current_solution.copy()
        
        for i in range(5):
            # 在当前解附近随机扰动
            bound_range = self.param_bounds[i][1] - self.param_bounds[i][0]
            neighbor[i] += np.random.normal(0, scale * bound_range * 0.1)
        
        return self._apply_constraints(neighbor)

    def optimize(self) -> Tuple[np.ndarray, float]:
        """执行优化过程"""
        current_solution = self._initialize_solution()
        current_score = self._evaluate(current_solution)
        best_solution = current_solution.copy()
        best_score = current_score
        
        self.best_scores_history.append(best_score)
        
        temp = self.initial_temp
        
        for i in range(self.max_iter):
            # 生成邻域解
            neighbor = self._get_neighbor(current_solution, temp)
            neighbor_score = self._evaluate(neighbor)
            
            # 计算能量差
            delta_e = neighbor_score - current_score
            
            # 决定是否接受新解
            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / temp):
                current_solution = neighbor
                current_score = neighbor_score
                
                # 更新最佳解
                if current_score < best_score:
                    best_solution = current_solution.copy()
                    best_score = current_score
            
            self.best_scores_history.append(best_score)
            
            # 降温
            temp *= self.cooling_rate
            if temp < self.final_temp:
                break
        
        plt.plot(self.best_scores_history, marker='o', label='SA')
        plt.xlabel('Iteration')
        plt.ylabel('Best Score')
        plt.title('Simulated Annealing Optimization Progress')
        plt.legend()
        
        return best_solution, best_score
    
class ACOOptimizer:
    def __init__(self, excel_path: str, num_ants: int = 20, max_iter: int = 50,
                 alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1, q: float = 1.0):
        """初始化蚁群优化器"""
        # 读取Excel数据
        self.df = pd.read_excel(excel_path)  
        self.param_columns = self.df.columns[1:6]  
        self.value_columns = self.df.columns[6:13]
        
        # ACO参数
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.alpha = alpha  # 信息素重要程度
        self.beta = beta    # 启发式信息重要程度
        self.rho = rho      # 信息素挥发系数
        self.q = q          # 信息素强度
        
        # 参数离散化 (将连续空间离散化为可选值)
        self.param_options = []
        for col in self.param_columns:
            min_val, max_val = self.df[col].min(), self.df[col].max()
            # 每个参数离散化为20个可选值
            self.param_options.append(np.linspace(min_val, max_val, 20))
        
        # 初始化信息素矩阵 (5个参数，每个参数20个可选值)
        self.pheromone = [np.ones(len(options)) * 0.1 for options in self.param_options]
        
        # 存储最佳分数历史
        self.best_scores_history = []

    def _evaluate(self, position: np.ndarray) -> float:
        """评估适应度"""
        position = np.array(position, dtype=np.float32)
        position = torch.tensor(position)
        position = torch.unsqueeze(position, dim=0)
        position = torch.unsqueeze(position, dim=0)
        pred_0 = net(position).detach().numpy()
        return pred_0[0, 4]

    def _apply_constraints(self, position):
        """应用约束条件"""
        position[0] = np.clip(position[0], 3, 8)
        position[1] = np.clip(position[1], 2 * position[0], 3.25 * position[0])
        position[2] = np.clip(position[2], 10, 80 - position[1])
        position[3] = np.clip(position[3], 10, 90 - position[2] - position[1])
        position[4] = 100 - position[3] - position[2] - position[1]
        return position

    def _construct_solution(self):
        """蚂蚁构建解"""
        solution = []
        for i in range(5):  # 5个参数
            # 计算选择概率
            pheromone = self.pheromone[i] ** self.alpha
            heuristic = 1.0 / (np.abs(self.param_options[i] - np.mean(self.param_options[i])) + 1e-6) ** self.beta
            
            # 确保 pheromone 和 heuristic 为非负数
            pheromone = np.maximum(pheromone, 0)
            heuristic = np.maximum(heuristic, 0)
            
            probabilities = pheromone * heuristic
            total = probabilities.sum()
            
            # 如果总和为 0，使用均匀分布
            if total == 0:
                probabilities = np.ones_like(probabilities) / len(probabilities)
            else:
                probabilities /= total
            
            # 根据概率选择参数值
            chosen_idx = np.random.choice(len(self.param_options[i]), p=probabilities)
            solution.append(self.param_options[i][chosen_idx])
        
        return np.array(solution)

    def optimize(self) -> Tuple[np.ndarray, float]:
        """执行优化过程"""
        best_solution = None
        best_score = float('inf')
        
        for iteration in range(self.max_iter):
            solutions = []
            scores = []
            
            # 每只蚂蚁构建解
            for _ in range(self.num_ants):
                solution = self._construct_solution()
                solution = self._apply_constraints(solution)
                score = self._evaluate(solution)
                
                solutions.append(solution)
                scores.append(score)
                
                # 更新最佳解
                if score < best_score:
                    best_solution = solution.copy()
                    best_score = score
            
            self.best_scores_history.append(best_score)
            
            # 更新信息素
            for i in range(5):  # 对每个参数
                # 信息素挥发
                self.pheromone[i] *= (1.0 - self.rho)
                
                # 信息素沉积 (只有最优解沉积信息素)
                for j in range(len(self.param_options[i])):
                    if np.isclose(self.param_options[i][j], best_solution[i]):
                        self.pheromone[i][j] += self.q / (best_score + 1e-6)
        
        plt.plot(self.best_scores_history, marker='o', label=f'ACO_{self.num_ants}')
        plt.xlabel('Iteration')
        plt.ylabel('Best Score')
        plt.title('Ant Colony Optimization Progress')
        plt.legend()
        
        return best_solution, best_score
    


if __name__ == "__main__":
    # 创建优化器实例
    ga_optimizer = GAOptimizer("库.xlsx", population_size=20, max_iter=10)
    sa_optimizer = SAOptimizer("库.xlsx", max_iter=10)
    aco_optimizer = ACOOptimizer("库.xlsx", num_ants=20, max_iter=10)

    
    # 执行优化
    ga_best_params, ga_best_score = ga_optimizer.optimize()
    sa_best_params, sa_best_score = sa_optimizer.optimize()
    aco_best_params, aco_best_score = aco_optimizer.optimize()

    
    # 显示结果
    plt.show()
    
    print("\n优化结果比较:")
    print(f"遗传算法最佳参数: {np.round(ga_best_params, 3)}, 最佳分数: {ga_best_score:.4f}")
    print(f"模拟退火最佳参数: {np.round(sa_best_params, 3)}, 最佳分数: {sa_best_score:.4f}")
    print(f"蚁群算法最佳参数: {np.round(aco_best_params, 3)}, 最佳分数: {aco_best_score:.4f}")
