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

import ast  # 用于安全地解析字符串为 Python 对象
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
    parser.add_argument("--logdir", type=str, default=global_variables["logdir"])
    parser.add_argument("--render", type=float, default=global_variables["render"])

    parser.add_argument("--watch",default=False,action="store_true",help="no training, watch the play of pre-trained models",)
    # parser.add_argument("--device",type=str,default="cuda" if torch.cuda.is_available() else "cpu",)
    parser.add_argument("--device",type=str,default="cpu",)
    # parser.add_argument("--Window", type=ast.literal_eval, default=None)
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]
# import argparse


# def get_args():
#     parser = argparse.ArgumentParser()
#     # 其他参数...
#     parser.add_argument(
#         "--hidden-sizes",
#         type=ast.literal_eval,  # 使用 ast.literal_eval 将字符串解析为列表
#         default=[64, 64],  # 默认值
#         help="Hidden layer sizes (e.g., [64, 64])"
#     )
#     return parser.parse_args()

def get_env(args: argparse.Namespace = get_args()) -> PettingZooEnv:
    return PettingZooEnv(sac_pso_env.PSO(n_pistons=args.n_pistons,main_window=global_variables["window_show"]))


def get_agents(
    args: argparse.Namespace = get_args(),
    agents: list[BasePolicy] | None = None,
    actor_optims: list[torch.optim.Optimizer] | None = None,
    critic1_optims: list[torch.optim.Optimizer] | None = None,
    critic2_optims: list[torch.optim.Optimizer] | None = None,
) -> tuple[BasePolicy, list[torch.optim.Optimizer] | None,list[torch.optim.Optimizer] | None,list[torch.optim.Optimizer] | None,list[torch.optim.Optimizer] | None, list]:
    env = get_env()
    # print("env",env)
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )
    args.state_shape = observation_space.shape or int(observation_space.n)
    args.action_shape = env.action_space.shape or int(env.action_space.n)
    # print("cccc")
    if agents is None:
        agents = []
        actor_optims = []
        critic1_optims= []
        critic2_optims= []
        # print(f"State shape: {args.state_shape}, Action shape: {args.action_shape}")
        # print(f"Hidden sizes: {args.hidden_sizes}")
        for _ in range(args.n_pistons):
            # model
            try:
                net = Net(
                    state_shape=args.state_shape,
                    action_shape=args.action_shape,
                    hidden_sizes=args.hidden_sizes,
                    device=args.device,
                ).to(args.device)
                # dummy_input = torch.randn(1,5, 3)
                # _ = net(dummy_input)
            except Exception as e:
                print("Error creating Net:", e)
                raise
            # print("dddd")

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



            # action_dim = space_info.action_info.action_dim
            # if args.auto_alpha:
            #     target_entropy = -action_dim
            #     log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
            #     alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
            #     args.alpha = (target_entropy, log_alpha, alpha_optim)


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

            # optim = torch.optim.Adam(net.parameters(), lr=args.lr)
            # agent: SACPolicy = SACPolicy(
            #     model=net,
            #     optim=optim,
            #     action_space=env.action_space,
            #     discount_factor=args.gamma,
            #     estimation_step=args.n_step,
            #     target_update_freq=args.target_update_freq,
            # )
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
    # print("window_show",window_show)
    # seed
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # train_envs.seed(args.seed)
    # test_envs.seed(args.seed)

    policy, actor_optim,critic1_optim,critic2_optim, agents = get_agents(
        args, agents=agents, 
        actor_optims=actor_optims,
        critic1_optims=critic1_optims,
        critic2_optims=critic2_optims
    )
    # print("bbbb")
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
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_best_fn(policy: BasePolicy) -> None:
        pass

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


def watch(args: argparse.Namespace = get_args(), policy: BasePolicy | None = None) -> None:
    env = DummyVectorEnv([get_env])
    if not policy:
        warnings.warn(
            "watching random agents, as loading pre-trained policies is currently not supported",
        )
        policy, _, _ = get_agents(args)
    [agent for agent in policy.policies.values()]
    collector = Collector[CollectStats](policy, env, exploration_noise=True)
    collector.reset()
    result = collector.collect(n_episode=int(global_variables["n_episode"]), render=args.render)
    result.pprint_asdict()


def test_global():
    print(global_variables)
# args = get_args()
# result, agent = train_agent(args)
# watch(args, agent)