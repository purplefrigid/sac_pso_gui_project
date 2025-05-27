import numpy as np  
import pandas as pd  
from typing import List, Tuple  
import torch
import matplotlib.pyplot as plt 
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
random_indices = np.array([2455,222,45,987,123,456,789,234,567,90,678,345,912,111,333,444,555,666,777,888,999])
class PSOOptimizer:  
    def __init__(self, excel_path: str, num_particles: int = 30, max_iter: int = 10):  
        """  
        初始化粒子群优化器  
        :param excel_path: Excel文件路径  
        :param num_particles: 粒子数量  
        :param max_iter: 最大迭代次数  
        """  
        # 读取Excel数据  
        self.df = pd.read_excel(excel_path)  
        self.param_columns = self.df.columns[1:6]  
        self.value_columns = self.df.columns[6:13]  
        
        # 存储参数和适应度的映射  
        self.param_cache = {}  
        for _, row in self.df.iterrows():  
            params = tuple(np.round(row[self.param_columns].values.astype(float), 3)) 
            self.param_cache[params] = row[self.value_columns].mean()  
        
        # PSO参数  
        self.num_particles = num_particles  
        self.max_iter = max_iter  
        self.w = 0.9    # 惯性权重  
        self.c1 = 0.1   # 个体学习因子  
        self.c2 = 0.1   # 群体学习因子  
        
        # 参数范围（基于数据统计）  
        self.param_bounds = [(self.df[col].min(), self.df[col].max()) for col in self.param_columns]  

    def _initialize_particles(self) -> Tuple[np.ndarray, np.ndarray]:  
        """ 初始化粒子位置和速度 """  
        particles = np.zeros((self.num_particles, 5))  
        velocities = np.zeros_like(particles)  
        
        # 从现有参数中随机采样初始化  
        existing_params = list(self.param_cache.keys())  
        indices = np.random.choice(len(existing_params), self.num_particles) 
                # 将 random_indices 的每个值代入计算 self._evaluate
        # choose_list=list(range(0, len(random_indices) ))  
        # candidates = {i: np.array(existing_params[j]).astype(np.float32) for i, j in zip(choose_list, random_indices)}
        # scores = {i: self._evaluate(candidates[i]) for i in choose_list}

        # # 对 scores 按值从小到大排序，选择前 self.n_pistons 个
        # sorted_agents = sorted(scores, key=scores.get)[:self.num_particles]
        # indices = [random_indices[agent] for agent in sorted_agents]
        # indices = random_indices[:self.num_particles]  
        print(f"indices: {indices}")
        for i, idx in enumerate(indices):  
            particles[i] = existing_params[idx]  
            
        return particles, velocities  

    def _evaluate(self, position: np.ndarray) -> float:  
        """ 评估适应度 """                               # 将input输入训练完成的网络，得到预测值

        position= np.array(position,dtype=np.float32)
        position = torch.tensor(position)                             # 将ndarry转换为tensor
        position = torch.unsqueeze(position, dim=0)                   # input.shape从[2]→[1, 2]
        position = torch.unsqueeze(position, dim=0)                   # 将input.shape从[1, 2]→[1, 1, 2]，才可作为网络输入用于预测
        pred_0 = net(position).detach().numpy() 
        return  pred_0[0,4] 

    # def _apply_constraints(self, position: np.ndarray) -> np.ndarray:
    #     """ 应用约束条件 """
    #     a, b, c, d, e = position
    #     a = np.clip(a, 3, 8)
    #     b = np.clip(b, 2 * a, 3.25 * a)
    #     c = np.clip(c, 10, 80-b)
    #     d = np.clip(d, 10, 90 - c)
    #     e = 100 -c-d-b
    #     return np.array([a, b, c, d, e])
    def _apply_constraints(self, list):
        """ 应用约束条件 """
        list[0] = np.clip(list[0], 3, 8)
        list[1] = np.clip(list[1], 2 * list[0], 3.25 * list[0])
        list[2] = np.clip(list[2], 10, 80-list[1])
        list[3] = np.clip(list[3], 10, 90 -list[2]-list[1])
        list[4] = 100 -list[3]-list[2]-list[1]
        return list
    
    def optimize(self) -> Tuple[np.ndarray, float]:  
        """ 执行优化过程 """  
        particles, velocities = self._initialize_particles()  
        best_positions = particles.copy()  
        best_scores = np.array([self._evaluate(p) for p in particles])  
        
        global_best_idx = np.argmin(best_scores)  
        global_best_position = particles[global_best_idx]  
        global_best_score = best_scores[global_best_idx]
        global_best_scores = []  
        global_best_scores.append( self._evaluate(particles[0]) )
        all_best_scores = []
        for _ in range(self.max_iter):  
            for i in range(self.num_particles):  
                # 更新速度  
                r1, r2 = np.random.rand(5), np.random.rand(5)  
                velocities[i] = (self.w * velocities[i] +  
                                self.c1 * r1 * (best_positions[i] - particles[i]) +  
                                self.c2 * r2 * (global_best_position - particles[i]))  
                
                # 更新位置  
                new_position = particles[i] + velocities[i]  
                
                # 边界处理  
                # new_position = np.clip(new_position,   
                #                       [b[0] for b in self.param_bounds],  
                #                       [b[1] for b in self.param_bounds])  
                new_position = self._apply_constraints(new_position)               
                # 评估新位置  
                current_score = self._evaluate(new_position)  
                
                # 更新个体最优  
                if current_score < best_scores[i]:  
                    best_positions[i] = new_position  
                    best_scores[i] = current_score  
                    
                    # 更新全局最优  
                    if current_score < global_best_score:  
                        global_best_position = new_position  
                        global_best_score = current_score  
            all_best_scores.append(best_scores.copy())
            global_best_scores.append(global_best_score)
            # 动态调整惯性权重  
            self.w *= 0.95  

        plt.plot(global_best_scores,marker='o',label=f'pso_{self.num_particles}')
        plt.xlabel('Iteration')
        plt.ylabel('Global Best Score')
        plt.title('Optimization Progress')
        # plt.show()
        # all_best_scores = np.array(all_best_scores)
        # for i in range(self.num_particles):
        #     plt.plot(all_best_scores[:, i], label=f'Particle {i+1}')
        # plt.xlabel('Iteration')
        # plt.ylabel('Best Score')
        # plt.title('Best Scores of All Particles')
        # plt.legend()
        # plt.show()
        return global_best_position, global_best_score  

# 使用示例  
if __name__ == "__main__":  
    optimizer = PSOOptimizer("库.xlsx", num_particles=10, max_iter=10)  
    best_params, best_score = optimizer.optimize()  
    optimizer = PSOOptimizer("库.xlsx", num_particles=20, max_iter=10)  
    best_params, best_score = optimizer.optimize()  
    # plt.show()
    # pred = net(torch.unsqueeze(torch.unsqueeze(torch.tensor(np.array(best_params,dtype=np.float32)), dim=0), dim=0)).detach().numpy() 
    # print(f"最优参数组合: {np.round(best_params, 3)}")  
    # print(f"最优值: {np.round(pred,3)}")  
    # print(f"最佳适应度值: {best_score:.4f}")   