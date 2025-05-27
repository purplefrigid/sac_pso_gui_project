import argparse
import os
import warnings

import gymnasium as gym
import numpy as np
import torch
# from pettingzoo.butterfly import pistonball_v6

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import sac_pso_env
from src import sac_pso_env_watch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, CollectStats, InfoStats, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, SACPolicy, MultiAgentPolicyManager
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.exploration import OUNoise
from src.config import global_variables
import matplotlib.pyplot as plt
import ast  # 用 于安全地解析字符串为 Python 对象
from pso import PSOOptimizer
from datetime import datetime

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=global_variables["seed"])

    parser.add_argument("--eps-test", type=float, default=global_variables["eps_test"])
    parser.add_argument("--eps-train", type=float, default=global_variables["eps_train"])
    parser.add_argument("--buffer-size", type=int, default=global_variables["buffer_size"])
    # parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--actor-lr", type=float, default=global_variables["actor_lr"])
    parser.add_argument("--critic-lr", type=float, default=global_variables["critic_lr"])
    parser.add_argument("--alpha-lr", type=float, default=global_variables["alpha_lr"])

    parser.add_argument("--noise_std", type=float, default=global_variables["noise_std"])
    parser.add_argument("--gamma",type=float,default=global_variables["gamma"],help="a smaller gamma favors earlier win",)
    parser.add_argument("--tau", type=float, default=global_variables["tau"])
    parser.add_argument("--auto_alpha", type=int, default=global_variables["auto_alpha"])
    parser.add_argument("--alpha", type=float, default=global_variables["alpha"])

    parser.add_argument("--n-pistons",type=int,default=global_variables["n_pistons"],help="Number of pistons(agents) in the env",)
    parser.add_argument("--n-step", type=int, default=global_variables["n_step"])
    parser.add_argument("--target-update-freq", type=int, default=global_variables["target_update_freq"])
    parser.add_argument("--epoch", type=int, default=global_variables["epoch"])
    parser.add_argument("--step-per-epoch", type=int, default= global_variables["step_per_epoch"])
    parser.add_argument("--step-per-collect", type=int, default=global_variables["step_per_collect"])
    parser.add_argument("--update-per-step", type=float, default=global_variables["update_per_step"])
    parser.add_argument("--batch-size", type=int, default=global_variables["batch_size"])
    parser.add_argument("--hidden-sizes", type=ast.literal_eval, nargs="*", default=global_variables["hidden_sizes"])
    parser.add_argument("--training-num", type=int, default=global_variables["training_num"])
    parser.add_argument("--test-num", type=int, default=global_variables["test_num"])
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--sac_model_dir", type=str, default="sac_model")
    parser.add_argument("--render", type=float, default=global_variables["render"])

    parser.add_argument("--watch",default=False,action="store_true",help="no training, watch the play of pre-trained models",)
    parser.add_argument("--device",type=str,default="cuda" if torch.cuda.is_available() else "cpu",)
    # parser.add_argument("--device",type=str,default="cpu",)
    parser.add_argument("--train", type=ast.literal_eval, default=True)
    return parser



def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


def get_env(args: argparse.Namespace = get_args()) -> PettingZooEnv:
    return PettingZooEnv(sac_pso_env.PSO(n_pistons=args.n_pistons))
def get_env_watch(args: argparse.Namespace ) -> PettingZooEnv:
    return PettingZooEnv(sac_pso_env_watch.PSO_watch(n_pistons=args.n_pistons,train=args.train))

def get_agents(
    args: argparse.Namespace = get_args(),
    agents: list[BasePolicy] | None = None,
    actor_optims: list[torch.optim.Optimizer] | None = None,
    critic1_optims: list[torch.optim.Optimizer] | None = None,
    critic2_optims: list[torch.optim.Optimizer] | None = None,
) -> tuple[BasePolicy, list[torch.optim.Optimizer] | None,list[torch.optim.Optimizer] | None,list[torch.optim.Optimizer] | None,list[torch.optim.Optimizer] | None, list]:
    env = get_env()

    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )
    args.state_shape = observation_space.shape or int(observation_space.n)
    args.action_shape = env.action_space.shape or int(env.action_space.n)
    if agents is None:
        agents = []
        actor_optims = []
        critic1_optims= []
        critic2_optims= []
        for _ in range(args.n_pistons):
            # model
            net = Net(
                state_shape=args.state_shape,
                action_shape=args.action_shape,
                hidden_sizes=args.hidden_sizes,
                device=args.device,
            ).to(args.device)


            actor = ActorProb(net, args.action_shape, device=args.device, unbounded=True).to(args.device)
            actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
            net_c1 = Net(
                state_shape=args.state_shape,
                action_shape=args.action_shape,
                hidden_sizes=args.hidden_sizes,
                concat=True,
                device=args.device,
            )
            critic1 = Critic(net_c1, device=args.device).to(args.device)
            critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
            
            net_c2 = Net(
                state_shape=args.state_shape,
                action_shape=args.action_shape,
                hidden_sizes=args.hidden_sizes,
                concat=True,
                device=args.device,
            )
            critic2 = Critic(net_c2, device=args.device).to(args.device)
            critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

            agent: SACPolicy = SACPolicy(
                actor=actor,
                actor_optim=actor_optim,  # No need for optimizer during testing
                critic=critic1,
                critic_optim=critic1_optim,  # No need for optimizer during testing
                critic2=critic2,
                critic2_optim=critic2_optim,  # No need for optimizer during testing
                tau=args.tau,
                gamma=args.gamma,
                alpha=args.alpha,
                exploration_noise=OUNoise(0.0, args.noise_std),
                action_space=env.action_space,
                # action_scaling=False,
            )

            agents.append(agent)
            actor_optims.append(actor_optim)
            critic1_optims.append(critic1_optim)
            critic2_optims.append(critic2_optim)

    policy = MultiAgentPolicyManager(policies=agents, env=env,action_scaling=True)
    return policy, actor_optims,critic1_optims,critic2_optims, env.agents


def train_agent(
    args: argparse.Namespace = get_args(),
    agents: list[BasePolicy] | None = None,
    actor_optims: list[torch.optim.Optimizer] | None = None,
    critic1_optims: list[torch.optim.Optimizer] | None = None,
    critic2_optims: list[torch.optim.Optimizer] | None = None,
) -> tuple[InfoStats, BasePolicy]:
    train_envs = DummyVectorEnv([get_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    policy, actor_optim,critic1_optim,critic2_optim, agents = get_agents(
        args, agents=agents, 
        actor_optims=actor_optims,
        critic1_optims=critic1_optims,
        critic2_optims=critic2_optims
    )

    # collector
    train_collector = Collector[CollectStats](
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector[CollectStats](
        policy, 
        test_envs, 
        exploration_noise=True
    )
    train_collector.reset()
    train_collector.collect(n_step=args.batch_size * args.training_num)#2*1
    # log
    log_path = os.path.join(args.logdir, "sac_pso_env", "sac")
    sac_model_path = os.path.join(args.sac_model_dir, "sac")
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_best_fn(policy: BasePolicy) -> None:
        # torch.save(policy.state_dict(), os.path.join(sac_model_path, "policy"+"_"+str(args.n_pistons)+f'_{10*args.n_pistons}'+'_5.14'+".pth"))
        from datetime import datetime
            # ...existing code...
        current_date = datetime.now().strftime("%Y%m%d")
        model_name = f"policy_{args.n_pistons}_{current_date}.pth"
        torch.save(policy.state_dict(), os.path.join(sac_model_path, model_name))

    def stop_fn(mean_rewards: float) -> bool:
        return False

    # def train_fn(epoch: int, env_step: int) -> None:
    #     [agent.set_eps(args.eps_train) for agent in policy.policies.values()]

    # def test_fn(epoch: int, env_step: int | None) -> None:
    #     [agent.set_eps(args.eps_test) for agent in policy.policies.values()]

    # def reward_metric(rews: np.ndarray) -> np.ndarray:
    #     return rews[:, 0]

    # trainer
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        # test_collector=None,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        # train_fn=train_fn,
        # test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=args.update_per_step,
        logger=logger,
        # test_in_train=False,
        # reward_metric=reward_metric,
    ).run()

    return result, policy


def watch(args: argparse.Namespace = get_args(), policy_path: str | None = None, policy: BasePolicy | None = None) -> None:
    if policy_path:
        import re
        # 匹配 policy 文件名格式
        match = re.search(r"_(\d+)(?:_(\d+))?", policy_path)
        if match:
            args.n_pistons = int(match.group(1))  # 提取第一个下划线后的数字并设置为 n_pistons
            if match.group(2) is None:  # 如果没有第二个数字，设置 train=False
                args.train = False
        else:
            raise ValueError(f"Invalid policy file name format: {policy_path}")

    env = get_env_watch(args)
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )
    args.state_shape = observation_space.shape or int(observation_space.n)
    args.action_shape = env.action_space.shape or int(env.action_space.n)
    agents = []
    actor_optims = []
    critic1_optims= []
    critic2_optims= []
    for _ in range(args.n_pistons):
        # model
        net = Net(
            state_shape=args.state_shape,
            action_shape=args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
        ).to(args.device)


        actor = ActorProb(net, args.action_shape, device=args.device, unbounded=True).to(args.device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
        net_c1 = Net(
            state_shape=args.state_shape,
            action_shape=args.action_shape,
            hidden_sizes=args.hidden_sizes,
            concat=True,
            device=args.device,
        )
        critic1 = Critic(net_c1, device=args.device).to(args.device)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
        net_c2 = Net(
            state_shape=args.state_shape,
            action_shape=args.action_shape,
            hidden_sizes=args.hidden_sizes,
            concat=True,
            device=args.device,
        )
        critic2 = Critic(net_c2, device=args.device).to(args.device)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

        agent: SACPolicy = SACPolicy(
            actor=actor,
            actor_optim=actor_optim,  # No need for optimizer during testing
            critic=critic1,
            critic_optim=critic1_optim,  # No need for optimizer during testing
            critic2=critic2,
            critic2_optim=critic2_optim,  # No need for optimizer during testing
            tau=args.tau,
            gamma=args.gamma,
            alpha=args.alpha,
            exploration_noise=OUNoise(0.0, args.noise_std),
            action_space=env.action_space,
            # action_scaling=False,
        )

        agents.append(agent)
        actor_optims.append(actor_optim)
        critic1_optims.append(critic1_optim)
        critic2_optims.append(critic2_optim)

    policy = MultiAgentPolicyManager(policies=agents, env=env,action_scaling=True)
    # log_path = os.path.join(args.logdir, "sac_pso_env", "sac")
    sac_model_path = os.path.join(args.sac_model_dir, "sac")
    policy.load_state_dict(torch.load(os.path.join(sac_model_path, policy_path)))
    env = DummyVectorEnv([lambda: get_env_watch(args)])
    if not policy:
        warnings.warn(
            "watching random agents, as loading pre-trained policies is currently not supported",
        )
        policy, _, _ = get_agents(args)
    [agent for agent in policy.policies.values()]
    collector = Collector[CollectStats](policy, env, exploration_noise=True)
    collector.reset()
    result = collector.collect(n_episode=1, render=args.render)
    # plt.show()
    # result.pprint_asdict()

def test_global():
    print(global_variables)


# args = get_args() 
# result, agent = train_agent(args)
# import shutil
# while True:
#     current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
#     output_dir = f"C:\\Users\\pc\\Desktop\\新建文件夹\\{current_time}"
#     os.makedirs(output_dir, exist_ok=True) 
#     args = get_args() 
#     # result, agent = train_agent(args)
#     watch(args,"policy_7_100.pth")
#     watch(args,"policy_10_100.pth")
#     watch(args,"policy_10.pth")
#     optimizer = PSOOptimizer("库.xlsx", num_particles=10, max_iter=10)  
#     optimizer.optimize() 
#     optimizer1 = PSOOptimizer("库.xlsx", num_particles=20, max_iter=10)  
#     optimizer1.optimize() 
#     plt.legend()
#     output_path = os.path.join(output_dir, "plot.png")
#     plt.savefig(output_path, dpi=300)  # dpi=300 设置高分辨率
#     plt.clf() 
#     # plt.show()
#     txt_files_to_move = ["PSO supported by SAC_7.txt", "PSO supported by SAC_10.txt", "PSO supported by SAC_10_not train.txt"]

#     # 遍历文件名列表并转移文件
#     for txt_file in txt_files_to_move:
#         source_path = os.path.join(".", txt_file)
#         destination_path = os.path.join(output_dir, txt_file)
        
#         # 检查文件是否存在
#         if os.path.exists(source_path):
#             shutil.move(source_path, destination_path)  # 转移文件
#             print(f"已转移: {source_path} -> {destination_path}")
#         else:
#             print(f"文件未找到: {source_path}")