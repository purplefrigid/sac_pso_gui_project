import os

import gymnasium
import numpy as np
import pandas as pd 
from gymnasium import spaces
from gymnasium.utils import EzPickle
import torch
from pettingzoo import AECEnv
from pettingzoo.classic.tictactoe.board import Board
from pettingzoo.utils import agent_selector, wrappers
import matplotlib.pyplot as plt 
from copy import deepcopy  
import sys
from src.config import global_variables
import torch

def get_resource_path(relative_path):
    """获取打包后文件的路径"""
    if hasattr(sys, '_MEIPASS'):
        # 如果是打包后的环境
        base_path = sys._MEIPASS
    else:
        # 如果是开发环境
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

Project_folder = get_resource_path("model\\")                  # 项目文件夹路径
DEVICE = torch.device('cpu')

# if global_variables["choose"] == "竖直":
#     model_pth = "shuzhi_0.pth"      # 模型文件名
#     net = torch.load(Project_folder  + model_pth, map_location=DEVICE,weights_only=False)
#     # print(net.to(DEVICE))
#     model_pth = "90.pth"      # 模型文件名
#     net1 = torch.load(Project_folder  + model_pth, map_location=DEVICE,weights_only=False)
# elif global_variables["choose"] == "水平":
#     model_pth = "shuiping_0.pth"      # 模型文件名
#     net = torch.load(Project_folder  + model_pth, map_location=DEVICE,weights_only=False)
#     model_pth = "90.pth"      # 模型文件名
#     net1 = torch.load(Project_folder  + model_pth, map_location=DEVICE,weights_only=False)
# else:
#     assert False, "error"

net = torch.load(Project_folder  + global_variables["model"], map_location=DEVICE,weights_only=False)
net1 = torch.load(Project_folder  + global_variables["model1"], map_location=DEVICE,weights_only=False)

jie_gou =['结构一', '结构二', '结构三', '结构四', '结构五'] 
x = np.arange(len(jie_gou)) 
a=0
width = 0.1  # 每个柱的宽度  
freq =np.array((1.3,1.7,2.4,3.2,5.6,9.4,15),dtype=np.float32)
excel_path = "库.xlsx"
color=['r','g','b','c','m','y']
colors = [  
    'red',          # 红色  
    'black',         # 亮绿色  
    'gold',         # 金色  
    'darkviolet',   # 深紫罗兰色  
    'hotpink',         # 青色  
    'saddlebrown',  # 鞍棕色  
    'deepskyblue',  # 深天蓝色  
    'magenta',      # 洋红色   
    'teal',         # 水鸭色  
    'hotpink',      # 亮粉色  
    'chartreuse',   # 黄绿色  
    'darkred',      # 深红色  
    'purple',       # 紫色  
    'yellowgreen',  # 黄绿色  
    'crimson',      # 深红色  
    'turquoise'     # 松石绿  
]  

random_indices = np.array([222,45,987,123,456,789,234,567,90,678,345,912,111,333,444,555,666,777,888,999])

from matplotlib import rcParams  
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体  
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题 

def env(**kwargs):
    env = PSO_watch(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class PSO_watch(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human"],
        "name": "pso",
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(
        self, 
        n_pistons=20,
        render_mode: str | None = None, 
        train: str | None = True, 
    ):
        EzPickle.__init__(
            self,
            n_pistons=n_pistons,
            render_mode=render_mode,
            train=train,
        )
        # super().__init__()
        self.variable_ranges = [  
            (0.01, 1, 0.01),    # 第 1 个变量 a：范围 [3, 8]，步长 1  
            (0.01, 1, 0.01),   # 第 2 个变量 b：范围 [6, 26]，步长 2（近似 2 * a 到 3.25 * a 的范围）  
            (0.01, 1, 0.01),   # 第 3 个变量 b：范围 [6, 26]，步长 2（近似 2 * a 到 3.25 * a 的范围）  
        ]  
        self.low = np.array([var[0] for var in self.variable_ranges], dtype=np.float32)  
        self.high = np.array([var[1] for var in self.variable_ranges], dtype=np.float32)  
        self.step_sizes = np.array([var[2] for var in self.variable_ranges], dtype=np.float32)  

        self.n_pistons = n_pistons
        self.train = train
        self.agents = ["piston_" + str(r) for r in range(self.n_pistons)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.n_pistons))))
        self.action_spaces = dict(
                zip(
                    self.agents,
                    [
                        spaces.Box(
                            low=0, high=1, shape=(len(self.variable_ranges),2), dtype=np.float32)
                    ] 
                    * self.n_pistons,
                )
            )    
    
        self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    spaces.Box(
                    low=-np.inf, high=np.inf, shape=(5,3), dtype=np.float32  )    
                ]
                * self.n_pistons,
            )
        )
        self.rewards = {i: 0 for i in self.agents}
        self.c1 = {i: 0.3 for i in self.agents}
        self.c2 = {i: 0.3 for i in self.agents}
        self.w = {i: 0.7 for i in self.agents}

        self.velocities = {i: np.zeros(5, dtype=np.float32) for i in self.agents}
        self.best_positions = {i: np.zeros(5, dtype=np.float32) for i in self.agents}
        self.best_scores={i: float('inf') for i in self.agents}
        self.current_score={i: float('inf') for i in self.agents}
        self.global_best_score = float('inf')
        self.globe_best_positions = np.array([0,0,0,0,0], dtype=np.float32)

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.render_mode = render_mode
        self.data=[]
        

        self.df = pd.read_excel(excel_path)  
        self.param_columns = self.df.columns[1:6]  
        self.value_columns = self.df.columns[6:13]  
        
        # 存储参数和适应度的映射  
        self.param_cache = {}  
        for _, row in self.df.iterrows():  
            params = tuple(np.round(row[self.param_columns].values.astype(float), 3)) 
            self.param_cache[params] = row[self.value_columns].mean()  


        self.maxepoch = 10

        self.n = 10
        self.a = 2+len(self.agents)

        if not self.train:
            self.result_path=f'PSO supported by SAC_{self.n_pistons}_not train.txt'
        else:
            self.result_path=f'PSO supported by SAC_{self.n_pistons}.txt'

        self.jishu=0
    def observe(self, agent):
        array1 = self.state[agent].reshape(-1, 1)  
        array2 = self.best_positions[agent].reshape(-1, 1)  
        array3 = self.globe_best_positions.reshape(-1, 1)  
        observation = np.hstack([array1, array2, array3]) 
        assert not np.isscalar(observation) , "error"
        # print(f"agent:{agent},observation:\n{observation}")
        # print(f"agent_obs:{agent}")
        # return {"observation": observation}
        return observation
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def reset(self, seed=None, options=None):
        # reset environment
        global a 
        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        # self.terminations = {i: False for i in self.agents}
        # self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        # selects the first agent
        self._agent_selector.reinit(self.agents)
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.reset()
        self.current_step = 0 

        a = self.a

        existing_params = list(self.param_cache.keys())  
        # # indices = np.random.choice(len(existing_params), self.n_pistons) 
        indices = random_indices[:self.n_pistons] 
        # choose_list=list(range(0, len(random_indices)))  
        # existing_params = list(self.param_cache.keys())  

        # # 将 random_indices 的每个值代入计算 self._evaluate
        # candidates = {i: np.array(existing_params[j]).astype(np.float32) for i, j in zip(choose_list, random_indices)}
        # scores = {i: self._evaluate(candidates[i]) for i in choose_list}

        # # 对 scores 按值从小到大排序，选择前 self.n_pistons 个
        # sorted_agents = sorted(scores, key=scores.get)[:self.n_pistons]
        # indices = [random_indices[agent] for agent in sorted_agents]
        print(f"indices:{indices}\n") 
        # 使用新的 indices 更新 self.raw 和 self.state
        self.raw = {i: np.array(existing_params[j]).astype(np.float32) for i, j in zip(self.agents, indices)}
        self.state = deepcopy(self.raw)

        self.best_positions= deepcopy(self.raw)
        
        # self.data.append(self._evaluate(self.globe_best_positions))
        self.current_score={i: self._evaluate(self.state[i]) for i in self.agents}
        min_agent = min(self.current_score, key=self.current_score.get)
        self.globe_best_positions = self.raw[min_agent]

        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        
        particles = np.zeros((self.n_pistons, 5))  
        velocities = np.zeros_like(particles)  
        self.velocities = {j: velocities[i].astype(np.float32) for i,j in enumerate(self.agents)}
        self.state_history = {agent: [] for agent in self.agents}  
        self.best_score_history = {agent: [] for agent in self.agents}  
        # return obs,self.infos


    # def _seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)


    def step(self, action):
        # if (
        #     self.terminations[self.agent_selection]
        #     or self.truncations[self.agent_selection]
        # ):
        #     action= None
        #     return self._was_dead_step(action)
        agent = self.agent_selection
        if self.current_step == 0:
            self.data.append(self._evaluate(self.globe_best_positions))
            with open(self.result_path,"w") as f:
                f.write(f"Step: {self.current_step//len(self.agents)}, \nState: {self.state},\nCurrent_score:{self.current_score} \
                \nbest: {self.best_scores},\nbest_posion:{self.best_positions},\
                \nself.globe_best_positions:{self.globe_best_positions},\
                \nself.global_best_score:{self._evaluate(self.globe_best_positions)}\n"+'\n')
        self.current_step += 1 
        # self.current_step //= self.n_pistons 
        if action[0] >= 0.5:  # 减小  
            self.c1[agent] = round(max(self.low[0], self.c1[agent] - self.step_sizes[0]) ,2) 
        else:
            self.c1[agent] = round(min(self.high[0], self.c1[agent] + self.step_sizes[0]),2)
        if action[1] >= 0.5:  # 增加  
            self.c1[agent] = round( max(self.low[0], self.c1[agent] / 2 )  ,2)
        else:
            self.c1[agent] = round(min(self.high[0], self.c1[agent] * 1.5),2)

        if action[2] >= 0.5:  # 减小  
            self.c2[agent] = round(max(self.low[1], self.c2[agent] - self.step_sizes[1]),2)   
        else:  
            self.c2[agent] = round(min(self.high[1], self.c2[agent] + self.step_sizes[1]),2) 
        if action[3] >= 0.5:  # 增加  
            self.c2[agent] = round(max(self.low[1], self.c2[agent] / 2 )  ,2) 
        else: 
            self.c2[agent] = round(min(self.high[1], self.c2[agent] * 1.5) ,2) 

        if action[4] >= 0.5:  # 减小  
            self.w[agent] = round( max(self.low[2], self.w[agent] - self.step_sizes[2]) ,2)  
        else: 
            self.w[agent] = round( min(self.high[2], self.w[agent] + self.step_sizes[2]),2) 
        if action[5] >= 0.5:  # 增加  
            self.w[agent] = round( max(self.low[2], self.w[agent] / 2 )  ,2) 
        else: 
            self.w[agent] = round( min(self.high[2], self.w[agent] * 1.5) ,2) 

        self.velocities[agent] = (self.w[agent] * self.velocities[agent] +  
                        self.c1[agent] * (self.best_positions[agent] - self.raw[agent]) +  
                        self.c2[agent] * (self.globe_best_positions - self.raw[agent]))
        
        self.state[agent] = self.raw[agent] + self.velocities[agent]  
        self.state[agent] = self._apply_constraints(self.state[agent]) 
        self.state[agent] =np.round(self.state[agent],2)
        self.current_score[agent] = self._evaluate(self.state[agent])
        # print(f"agent_now:{agent}")
        self.best_scores[agent]=self._evaluate(self.best_positions[agent])
        self.global_best_score=self._evaluate(self.globe_best_positions)

        if self.current_score[agent] < self.best_scores[agent]:  
            self.best_positions[agent] = self.state[agent]  
            self.best_positions[agent] =np.round(self.best_positions[agent],2)
            self.best_scores[agent] = self.current_score[agent]  
            self.rewards[agent]+=10
            # print(f"agent:{agent},\
            #       best_scores:{self.best_scores[agent]},\
            #       current_score:{current_score},\
            #       global_best_score:{self.global_best_score},\
            #       \nbest_positions:{self.best_positions[agent]},\
            #       globe_best_positions:{self.globe_best_positions}"

            #     )
        
        else:
            self.rewards[agent]-=1

            # 更新全局最优  
        if self.current_score[agent] < self.global_best_score:  
            self.globe_best_positions = self.state[agent] 
            self.globe_best_positions =np.round(self.globe_best_positions,2) 
            self.global_best_score = self.current_score[agent] 
            self.rewards[agent]+=20 
            # print("best change!")
            
        next_agent = self._agent_selector.next()
        self.agent_selection = next_agent

        self.observation = {  
            i: np.hstack([  
                    self.state[i].reshape(-1, 1),         # 当前状态 (5,1)  
                    self.best_positions[i].reshape(-1, 1), # 个体最优 (5,1)  
                    self.globe_best_positions.reshape(-1, 1)  # 全局最优 (5,1)  
                        ]).astype(np.float32)  # 最终形状 (5,3)  
                   
            for i in self.agents  
        }   
        obs = self.observation

        self.terminations[agent] = self.current_step >= self.n*len(self.agents)  # 回合是否因为达到最大步数而结束 
        # self.terminations[agent] = self.best_scores[agent] <= float(global_variables["target"] ) # 回合是否因为达到最大步数而结束
        # self.terminations[agent]=False
        # truncated = True  # 没有时间限制，设置为 False  
        # self._accumulate_rewards()
        if self.render_mode == "human":
            self.render()
        return obs,self.rewards,self.terminations,self.truncations,{}
    
    def render(self, mode="human"):  
        global a 
        self.jishu+=1
        if self.jishu%len(self.agents)==0 :
            a+=1 
            # 打印当前状态和总和  
            self.data.append(self._evaluate(self.globe_best_positions))
            position= np.array(self.globe_best_positions,dtype=np.float32)
            position = torch.tensor(position)                             # 将ndarry转换为tensor
            position = torch.unsqueeze(position, dim=0)                   # input.shape从[2]→[1, 2]
            position = torch.unsqueeze(position, dim=0)                   # 将input.shape从[1, 2]→[1, 1, 2]，才可作为网络输入用于预测
            draw = net(position).detach().numpy()  
            draw1 = net1(position).detach().numpy()  
            draw = np.squeeze(draw) 
            draw1 = np.squeeze(draw1)  
            # try:
            #     plt.figure(1)
            #     plt.plot(freq,draw,color=colors[a-self.a],label=f'num_{self.current_step//len(self.agents)}')
            #     plt.plot(freq,draw1,color=colors[a-self.a],label=f'num_{self.current_step//len(self.agents)}')
            # except:
            #     pass
            for id,agent in enumerate(self.agents):   
                position= np.array(self.state[agent],dtype=np.float32)
                position = torch.tensor(position)                             # 将ndarry转换为tensor
                position = torch.unsqueeze(position, dim=0)                   # input.shape从[2]→[1, 2]
                position = torch.unsqueeze(position, dim=0)                   # 将input.shape从[1, 2]→[1, 1, 2]，才可作为网络输入用于预测
                draw = net(position).detach().numpy()  
                draw1 = net1(position).detach().numpy()  
                draw = np.squeeze(draw) 
                draw1 = np.squeeze(draw1)  
                # try:
                #     plt.figure(2)
                #     plt.plot(freq,draw,color=color[id],label=f'{agent}_{self.current_step//len(self.agents)}') 
                #     plt.plot(freq,draw1,color=color[id],label=f'{agent}_{self.current_step//len(self.agents)}')    
                #     plt.figure(id+3)
                #     plt.bar(x+(jishu//len(self.agents) - self.n//len(self.agents)/2) * width + width/2  , self.state[agent],width, color=colors[jishu] )
                #     plt.xticks(x, jie_gou)
                #     plt.title(f'{agent}')

                #     plt.figure(a)
                #     if np.array_equal(self.state[agent], self.globe_best_positions):
                #         plt.bar(x + (id - len(self.agents) / 2) * width + width / 2, self.state[agent], width, color='red')
                #     else:
                #         plt.bar(x + (id - len(self.agents) / 2) * width + width / 2, self.state[agent], width, color="blue")
                #     plt.xticks(x, jie_gou)
                #     plt.title(f'第{self.current_step//len(self.agents)}轮')  
                # except:
                #     pass
            result = f"Step: {self.current_step//len(self.agents)}, \nState: {self.state}, \nCurrent_score:{self.current_score}\
                \nbest: {self.best_scores},\nbest_posion:{self.best_positions},\
                \nself.globe_best_positions:{self.globe_best_positions},\
                \nself.global_best_score:{self.global_best_score}\n"
            print(result)  

            with open(self.result_path,"a") as f:
                f.write(result+'\n')
            # print(f"Step: {self.current_step//len(self.agents)}, \nself.globe_best_positions:{self.globe_best_positions},\
            #     \nself.global_best_score:{self.global_best_score}\n")          

        
        # plt.plot(freq,draw,color=colors[jishu],label=f'num_{self.current_step}')
        if self.jishu == self.n*len(self.agents):
            # plt.figure(1)
            # plt.legend()
            # plt.xlabel('Iteration')
            # plt.ylabel('Global Best Score')
            # plt.title('globe_best_positions change')
            # plt.figure(2)
            # plt.legend()
            # plt.xlabel('Iteration')
            # plt.ylabel('agents change')
            # plt.title('agents change')
            # plt.show()
            # liebiao=list(range(1, len(self.data) + 1))  
            if not self.train:
                plt.plot(self.data, marker='o',label=f'PSO supported by SAC_{self.n_pistons}_not train')
            else:
                plt.plot(self.data, marker='o',label=f'PSO supported by SAC_{self.n_pistons}')
            # plt.legend()
            # plt.show()
            # self.data=[]
            self.jishu =0
            a = self.a


    def close(self):  
        pass  

    def _apply_constraints(self, list):
        """ 应用约束条件 """
        list[0] = np.clip(list[0], 3, 8)
        list[1] = np.clip(list[1], 2 * list[0], 3.25 * list[0])
        list[2] = np.clip(list[2], 10, 80-list[1])
        list[3] = np.clip(list[3], 10, 90 -list[2]-list[1])
        list[4] = 100 -list[3]-list[2]-list[1]
        return list
    def _evaluate(self, position: np.ndarray) -> float:  
        """ 评估适应度 """    
        position= np.array(position,dtype=np.float32)
        position = torch.tensor(position)                     
        position = torch.unsqueeze(position, dim=0)                 
        position = torch.unsqueeze(position, dim=0)                  
        pred_0 = net(position).detach().numpy() 
        pred_90 = net1(position).detach().numpy() 
        return  pred_0[0,4] 
    