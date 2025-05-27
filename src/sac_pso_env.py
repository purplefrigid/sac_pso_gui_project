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
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# print(sys.path)
from src.config import global_variables
import torch
# from src.F_model_8paras import F_Net_1D

def get_resource_path(relative_path):
    """获取打包后文件的路径"""
    if hasattr(sys, '_MEIPASS'):
        # 如果是打包后的环境
        base_path = sys._MEIPASS
    else:
        # 如果是开发环境
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# net = F_Net_1D()
# net1 = F_Net_1D()
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

jishu=0
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
from matplotlib import rcParams  
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体  
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题 

def env(**kwargs):
    env = PSO(**kwargs)
    # if env.continuous:
    #     env = wrappers.ClipOutOfBoundsWrapper(env)
    # else:
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class PSO(AECEnv, EzPickle):
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
    ):
        EzPickle.__init__(
            self,
            n_pistons=n_pistons,
            render_mode=render_mode,
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
        # self.action = spaces.MultiDiscrete([4] * len(self.variable_ranges))
        # self.action = spaces.Box(low=0, high=1, shape=(len(self.variable_ranges),2), dtype=np.float32)  

        self.n_pistons = n_pistons
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
        
        # self.action_spaces = {i: self.action for i in self.agents}
        # self.observation_spaces = {
        #     i: spaces.Dict(
        #         {
        #             "observation": spaces.Box(
        #             #     low=0, high=100, shape=(1, 5, 3), dtype=np.float32
        #             low=-np.inf, high=np.inf, shape=(5,3), dtype=np.float32  
        #             ),
        #         }
        #     )
        #     for i in self.agents
        # }
        self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    spaces.Box(
                    #     low=0, high=100, shape=(1, 5, 3), dtype=np.float32
                    low=-np.inf, high=np.inf, shape=(5,3), dtype=np.float32  )    
                ]
                * self.n_pistons,
            )
        )
        self.rewards = {i: 0 for i in self.agents}
        self.c1 = {i: 0.3 for i in self.agents}
        self.c2 = {i: 0.3 for i in self.agents}
        self.w = {i: 0.7 for i in self.agents}
        # self.raw = {i: np.array([0,0,0,0,0], dtype=np.float32) for i in self.agents}
        # self.state = {i: np.array([0,0,0,0,0], dtype=np.float32) for i in self.agents}
        # self.raw = {i: np.zeros(5, dtype=np.float32) for i in self.agents}  

        # self.velocities = {i: np.array([0,0,0,0,0], dtype=np.float32) for i in self.agents}
        # self.best_positions = {i: np.array([0,0,0,0,0], dtype=np.float32) for i in self.agents}
        self.velocities = {i: np.zeros(5, dtype=np.float32) for i in self.agents}
        self.best_positions = {i: np.zeros(5, dtype=np.float32) for i in self.agents}
        self.best_scores={i: float('inf') for i in self.agents}
        self.current_score={i: float('inf') for i in self.agents}
        self.global_best_score = float('inf')
        self.globe_best_positions = np.array([0,0,0,0,0], dtype=np.float32)


        # self.infos = {i: {"legal_moves": list(range(0, 9))} for i in self.agents}

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


        # self.maxepoch = 10*self.n_pistons  # 最大迭代次数
        self.maxepoch = 100
        # self.plot_interval = 50  # 每50步可视化一次  
        # self.state_history = {agent: [] for agent in self.agents}  
        # self.best_score_history = {agent: [] for agent in self.agents}  
        # self.global_steps = 0  # 全局步数计数器  
        self.n = 15
        self.a = 2+len(self.agents)
    def observe(self, agent):
        # array1=self.state[agent].reshape(1,5,1)
        # array2=self.best_positions[agent].reshape(1,5,1)
        # array3=self.globe_best_positions.reshape(1,5,1)
        # observation = np.concatenate([array1, array2, array3], axis=2)
        array1 = self.state[agent].reshape(-1, 1)  
        array2 = self.best_positions[agent].reshape(-1, 1)  
        array3 = self.globe_best_positions.reshape(-1, 1)  
        observation = np.hstack([array1, array2, array3]) 
        assert not np.isscalar(observation) , "error"
        # print(f"agent:{agent},observation:\n{observation}")
        # print(f"agent_obs:{agent}")
        # return {"observation": observation}
        return observation
        # if self.render_mode == "human":
        #     self.clock = pygame.time.Clock()
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
        indices = np.random.choice(len(existing_params), self.n_pistons)  
        self.raw = {i: np.array(existing_params[j]).astype(np.float32) for i,j in zip(self.agents,indices)}
        self.best_positions= deepcopy(self.raw)
        self.state = deepcopy(self.raw)
        # self.data.append(self._evaluate(self.globe_best_positions))
        self.current_score={i: self._evaluate(self.state[i]) for i in self.agents}
        min_agent = min(self.current_score, key=self.current_score.get)
        self.globe_best_positions = self.raw[min_agent]
        # self.globe_best_positions = self.raw[self.agents[0]]
        

        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        
        particles = np.zeros((self.n_pistons, 5))  
        velocities = np.zeros_like(particles)  
        self.velocities = {j: velocities[i].astype(np.float32) for i,j in enumerate(self.agents)}
        # obs = {  
        #     i: np.hstack([  
        #             self.state[i].reshape(-1, 1),         # 当前状态 (5,1)  
        #             self.best_positions[i].reshape(-1, 1), # 个体最优 (5,1)  
        #             self.globe_best_positions.reshape(-1, 1)  # 全局最优 (5,1)  
        #                 ]).astype(np.float32)  # 最终形状 (5,3)  
                   
        #     for i in self.agents  
        # }   
        # 从现有参数中随机采样初始化  
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
        current_score = self._evaluate(self.state[agent])
        # print(f"agent_now:{agent}")
        self.best_scores[agent]=self._evaluate(self.best_positions[agent])
        self.global_best_score=self._evaluate(self.globe_best_positions)

        if current_score < self.best_scores[agent]:  
            self.best_positions[agent] = self.state[agent]  
            # self.best_positions[agent] =np.round(self.best_positions[agent],2)
            self.best_scores[agent] = current_score  
            self.rewards[agent]+=2
            # print(f"agent:{agent},\
            #       best_scores:{self.best_scores[agent]},\
            #       current_score:{current_score},\
            #       global_best_score:{self.global_best_score},\
            #       \nbest_positions:{self.best_positions[agent]},\
            #       globe_best_positions:{self.globe_best_positions}"

            #     )
                # print("best change!")
        else:
            self.rewards[agent]-=1
        # self.global_steps += 1  
            # 更新全局最优  
        if current_score < self.global_best_score:  
            self.globe_best_positions = self.state[agent] 
            # self.globe_best_positions =np.round(self.globe_best_positions,2) 
            self.global_best_score = current_score 
            self.rewards[agent]+=4    
    # 记录所有agent的状态和分数  
        # if self.global_steps % self.n_pistons == 0: 
        #     for agent in self.agents:  
        #         self.state_history[agent].append(self.state[agent].copy())  
        #         self.best_score_history[agent].append(self.best_scores[agent])  
    
        # # 定期生成可视化  
        # if self.global_steps % self.plot_interval == 0:  
        #     # self._visualize_progress()  
        #     self.global_steps = 0
        #     self.state_history = {agent: [] for agent in self.agents}  
        #     self.best_score_history = {agent: [] for agent in self.agents}  

        next_agent = self._agent_selector.next()
        self.agent_selection = next_agent


        # obs = {  
        #     i: {  
        #             "observation": np.hstack([  
        #             self.state[i].reshape(-1, 1),         # 当前状态 (5,1)  
        #             self.best_positions[i].reshape(-1, 1), # 个体最优 (5,1)  
        #             self.globe_best_positions.reshape(-1, 1)  # 全局最优 (5,1)  
        #                 ]).astype(np.float32)  # 最终形状 (5,3)  
        #         }   
        #     for i in self.agents  
        # } 
        self.observation = {  
            i: np.hstack([  
                    self.state[i].reshape(-1, 1),         # 当前状态 (5,1)  
                    self.best_positions[i].reshape(-1, 1), # 个体最优 (5,1)  
                    self.globe_best_positions.reshape(-1, 1)  # 全局最优 (5,1)  
                        ]).astype(np.float32)  # 最终形状 (5,3)  
                   
            for i in self.agents  
        }   
        obs = self.observation

        self.terminations[agent] = self.current_step >= self.maxepoch  # 回合是否因为达到最大步数而结束 
        # self.terminations[agent]=False
        # truncated = True  # 没有时间限制，设置为 False  
        # self._accumulate_rewards()
        if self.render_mode == "human":
            self.render()
        return obs,self.rewards,self.terminations,self.truncations,{}
    
    # def _visualize_progress(self):  
    # # 创建两个并列子图  
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))  
        
    #     # 状态变化子图（每个step单独一图）  
    #     for agent in self.agents:  
    #         states = np.array(self.state_history[agent])  
    #         ax1.plot(states[:, 0], states[:, 1],   
    #                 marker='o', linestyle='--',   
    #                 label=f'Agent {agent} Trajectory')  
    #         ax1.scatter(states[-1, 0], states[-1, 1],   
    #                 s=100, edgecolor='black', zorder=5)  
        
    #     ax1.set_title(f'Agent States @ Step {self.global_steps}')  
    #     ax1.set_xlabel('X Position')  
    #     ax1.set_ylabel('Y Position')  
    #     ax1.grid(True)  
    #     ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  
        
    #     # 最佳分数变化子图（集中显示）  
    #     for agent in self.agents:  
    #         ax2.plot(self.best_score_history[agent],   
    #                 linewidth=2,   
    #                 label=f'Agent {agent}')  
        
    #     ax2.set_title('Best Scores Evolution')  
    #     ax2.set_xlabel('Global Steps')  
    #     ax2.set_ylabel('Score Value')  
    #     ax2.grid(True)  
    #     ax2.legend()  
        
    #     plt.tight_layout()  
    #     plt.show()  


    def render(self, mode="human"):  
        global jishu
        global a 
        jishu+=1
        # 打印当前状态和总和  
        self.data.append(self._evaluate(self.globe_best_positions))
        position= np.array(self.globe_best_positions,dtype=np.float32)
        position = torch.tensor(position)                             # 将ndarry转换为tensor
        position = torch.unsqueeze(position, dim=0)                   # input.shape从[2]→[1, 2]
        position = torch.unsqueeze(position, dim=0)                   # 将input.shape从[1, 2]→[1, 1, 2]，才可作为网络输入用于预测
        draw = net(position).detach().numpy()  
        draw = np.squeeze(draw)  
        
        plt.figure(1)
        plt.plot(freq,draw,color=colors[jishu],label=f'num_{self.current_step}')
        if jishu%len(self.agents)==0:
            a+=1 
            for id,agent in enumerate(self.agents):   
                position= np.array(self.state[agent],dtype=np.float32)
                position = torch.tensor(position)                             # 将ndarry转换为tensor
                position = torch.unsqueeze(position, dim=0)                   # input.shape从[2]→[1, 2]
                position = torch.unsqueeze(position, dim=0)                   # 将input.shape从[1, 2]→[1, 1, 2]，才可作为网络输入用于预测
                draw = net(position).detach().numpy()  
                draw = np.squeeze(draw)  
                plt.figure(2)
                plt.plot(freq,draw,color=color[id],label=f'{agent}_{self.current_step//len(self.agents)}')     
                plt.figure(id+3)
                plt.bar(x+(jishu/len(self.agents) - self.n/len(self.agents)/2) * width + width/2  , self.state[agent],width, color=colors[jishu] )
                plt.xticks(x, jie_gou)
                plt.title(f'{agent}')
                plt.figure(a)
                if np.array_equal(self.state[agent], self.globe_best_positions):
                    plt.bar(x + (id - len(self.agents) / 2) * width + width / 2, self.state[agent], width, color='red')
                else:
                    plt.bar(x + (id - len(self.agents) / 2) * width + width / 2, self.state[agent], width, color=colors[jishu])
                plt.xticks(x, jie_gou)
                plt.title(f'第{self.current_step//len(self.agents)}轮')              
            print(f"Step: {self.current_step//len(self.agents)}, State: {self.state}, \nbest: {self.best_scores},best_posion:{self.best_positions},\nself.c1: {self.c1},self.c2: {self.c2},self.w: {self.w}")  

        
        # plt.plot(freq,draw,color=colors[jishu],label=f'num_{self.current_step}')
        if jishu == self.n:
            plt.figure(1)
            plt.legend()
            plt.xlabel('Iteration')
            plt.ylabel('Global Best Score')
            plt.title('Optimization Progress')
            plt.figure(2)
            plt.legend()
            plt.xlabel('Iteration')
            plt.ylabel('agents change')
            plt.title('Optimization Progress')
            plt.show()
            liebiao=list(range(1, len(self.data) + 1))  
            plt.plot(self.data,color="r")
            plt.show()
            # self.data=[]
            jishu =0
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
    