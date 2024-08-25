import argparse
import os
import pickle
import torch
import torch.optim as optim
import time
import numpy as np

from pathlib import Path
from learning.ray_model import SimulationNN_Ray
from learning.ray_model import MuscleNN
from core.env import Env as MyEnv
from learning.ray_ppo import CustomPPOTrainer

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import register_env

torch, nn = try_import_torch()

## Supervised learner for muscle model
class MuscleLearner:
    def __init__(
        self,
        num_tau_des,
        num_muscles,
        num_reduced_JtA,
        learning_rate=1e-4,
        num_epochs=3,
        batch_size=128,
        model_weight=None,
        device="cuda",
        sizes = [256,256,256],
        learningStd=True,
    ):
        self.device = device
        if not torch.cuda.is_available():
            self.device = "cpu"
        
        self.num_tau_des = num_tau_des
        self.num_muscles = num_muscles
        self.num_reduced_JtA = num_reduced_JtA
        self.num_epochs_muscle = num_epochs
        self.muscle_batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = MuscleNN(self.num_reduced_JtA, 
                              self.num_tau_des, 
                              self.num_muscles, 
                              device = self.device, 
                              config={"sizes":sizes, "learningStd":learningStd})
        
        if model_weight:
            self.model.load_state_dict(model_weight)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for param in self.model.parameters():
            param.register_hook(lambda grad: torch.clamp(grad, -0.5, 0.5))

        self.stats = {}
        self.model.train()
    
    def train(self, reduced_JtA, net_tau_des, full_JtA):
        idx_all = np.asarray(range(len(reduced_JtA)))
        stats = {}
        stats["loss"] = {}
        
        for iter in range(self.num_epochs_muscle):
            ## shuffle
            np.random.shuffle(idx_all)

            ## Efficient shuffle 
            for i in range(len(reduced_JtA) // self.muscle_batch_size):
                mini_batch_idx = torch.from_numpy(idx_all[i * self.muscle_batch_size : (i + 1) * self.muscle_batch_size]).cuda()
                batch_reduced_JtA = torch.index_select(reduced_JtA, 0, mini_batch_idx)
                batch_net_tau_des = torch.index_select(net_tau_des, 0, mini_batch_idx)
                batch_full_JtA = torch.index_select(full_JtA, 0, mini_batch_idx)

                self.optimizer.zero_grad()
                a_pred = self.model.forward(batch_reduced_JtA, batch_net_tau_des)
                
                # mse loss
                mse_loss = torch.nn.functional.mse_loss(batch_net_tau_des, (a_pred.unsqueeze(1) @ batch_full_JtA.transpose(1,2)).squeeze(1))
                mse_loss /= 10000.0

                reg_loss = a_pred.pow(2).mean()
                
                loss = mse_loss + 0.01 * reg_loss

                ## put loss to stats
                if iter == self.num_epochs_muscle - 1 and i == (len(reduced_JtA) // self.muscle_batch_size - 1):
                    stats["loss"]["mse_loss"] = mse_loss.item()
                    stats["loss"]["reg_loss"] = reg_loss.item()
                    stats["loss"]["total_loss"] = loss.item()
                loss.backward()
                self.optimizer.step()

        return stats
    
    def get_model_weights(self, device=None):
        if device:
            return {k: v.to(device) for k, v in self.model.state_dict().items()}
        else:
            return self.model.state_dict()
    
    def set_model_weights(self, weights):
        self.model.load_state_dict(weights)


def create_my_trainer(rl_algorithm: str):
    if rl_algorithm == "PPO":
        RLTrainer = CustomPPOTrainer
    else:
        raise RuntimeError(f"Invalid algorithm {rl_algorithm}!")

    class MyTrainer(RLTrainer):
        def setup(self, config):
            self.env_str = config.pop("env_str")

            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )      
                
            self.policy_nn_config = config.pop("policyNN") 
            self.policy_nn_config["sizes"] = config.pop("sizes")
            self.policy_nn_config["learningStd"] = config.pop("learningStd")
            
            self.actuator_type = config.pop("actuator_type")
            self.trainer_config = config.pop("trainer_config")

            if self.actuator_type.find("mass") != -1:
                self.muscle_nn_config = config.pop("muscleNN") # [config.pop("num_reduced_JtA"), config.pop("num_muscles"), config.pop("num_tau_des")]
            
            RLTrainer.setup(self, config=config)       
            
            ## Initialize mass
            if self.actuator_type.find("mass") != -1:
                self.remote_workers = self.workers.remote_workers()
                self.muscle_learner = MuscleLearner(
                    self.muscle_nn_config["num_tau_des"],
                    self.muscle_nn_config["num_muscles"],
                    self.muscle_nn_config["num_reduced_JtA"],
                    learning_rate=self.trainer_config["muscle"]["lr"],
                    num_epochs=self.trainer_config["muscle"]["num_epochs"],
                    batch_size=self.trainer_config["muscle"]["sgd_minibatch_size"],
                    sizes=self.trainer_config["muscle"]["sizes"],
                    learningStd=self.trainer_config["muscle"]["learningStd"],
                )
            
                model_weights = ray.put(self.muscle_learner.get_model_weights(device=torch.device("cpu")))
                muscle_nn_config = {"sizes":self.trainer_config["muscle"]["sizes"], 
                                    "learningStd":self.trainer_config["muscle"]["learningStd"]}
                for worker in self.remote_workers:
                    worker.foreach_env.remote(lambda env: env.set_muscle_network(muscle_nn_config))
                    worker.foreach_env.remote(lambda env: env.load_muscle_model_weight(model_weights))
            
            self.max_reward = -float("inf")
            self.idx = 0
            

        def step(self):
            result = RLTrainer.step(self)
            current_reward = result["episode_reward_mean"]
            result["sampler_results"].pop("hist_stats")
            result["loss"] = {}
            if self.actuator_type.find("mass") != -1:
                start = time.perf_counter()
                mts = []    # muscle tuples
                muscle_transitions = []

                ## Collect Muscle Tuples 
                for idx in range(3):
                    mts.append(ray.get([worker.foreach_env.remote(lambda env: env.get_muscle_tuples(idx)) for worker in self.remote_workers]))
                    muscle_transitions.append([])
                    
                [muscle_transitions[mts_i].append(mts[mts_i][worker_i][env_i][i]) for mts_i in range(len(mts)) for worker_i in range(len(mts[mts_i]))  for env_i in range(len(mts[mts_i][worker_i])) for i in range(len(mts[mts_i][worker_i][env_i]))]
                loading_time = (time.perf_counter() - start) * 1000

                converting_time = time.perf_counter()
                # convert to cuda
                muscle_transitions[0] = torch.tensor(np.array(muscle_transitions[0], dtype=np.float32), device=self.device)
                muscle_transitions[1] = torch.tensor(np.array(muscle_transitions[1], dtype=np.float32), device=self.device)
                muscle_transitions[2] = torch.tensor(np.array(muscle_transitions[2], dtype=np.float32), device=self.device)
                converting_time = (time.perf_counter() - converting_time) * 1000

                learning_time = time.perf_counter()
                stats = self.muscle_learner.train(muscle_transitions[0], muscle_transitions[1], muscle_transitions[2])
                learning_time = (time.perf_counter() - learning_time) * 1000
                distribute_time = time.perf_counter()
                model_weights = ray.put(
                    self.muscle_learner.get_model_weights(device=torch.device("cpu"))
                )
                for worker in self.remote_workers:
                    worker.foreach_env.remote(
                        lambda env: env.load_muscle_model_weight(model_weights)
                    )

                distribute_time = (time.perf_counter() - distribute_time) * 1000
                total_time = (time.perf_counter() - start) * 1000

                result["timers"]["muscle_learning"] = {}
                result["timers"]["muscle_learning"]["distribute_time_ms"] = distribute_time
                result["timers"]["muscle_learning"]["learning_time_ms"] = learning_time
                result["timers"]["muscle_learning"]["loading_time_ms"] = loading_time
                result["timers"]["muscle_learning"]["converting_time_ms"] = converting_time
                result["timers"]["muscle_learning"]["total_ms"] = total_time
                result["loss"]["muscle"] = stats["loss"]

            if self.max_reward < current_reward:
                self.max_reward = current_reward
                self.save_checkpoint(self._logdir, "max")
            self.save_checkpoint(self._logdir, "last")
            self.idx += 1

            return result

        def __getstate__(self):
            state = RLTrainer.__getstate__(self)
            state['env_str'] = self.env_str
            state['policyNN'] = self.policy_nn_config
    
            if self.actuator_type.find("mass") != -1:
                state["muscle"] = self.muscle_nn_config
                state["muscle"]["weights"] = self.muscle_learner.get_model_weights(torch.device("cpu"))
                state["muscle"]["sizes"] = self.muscle_learner.model.config["sizes"]
                state["muscle"]["learningStd"] = self.muscle_learner.model.config["learningStd"]
    
            return state

        def __setstate__(self, state):
            RLTrainer.__setstate__(self, state)
            if self.actuator_type.find("mass") != -1:
                self.muscle_nn_config = state["muscle"]
                self.muscle_learner = MuscleLearner(
                    self.muscle_nn_config["num_tau_des"],
                    self.muscle_nn_config["num_muscles"],
                    self.muscle_nn_config["num_reduced_JtA"],
                    learning_rate=self.trainer_config["muscle"]["lr"],
                    num_epochs=self.trainer_config["muscle"]["num_epochs"],
                    batch_size=self.trainer_config["muscle"]["sgd_minibatch_size"],
                    model_weight=state["muscle"]["weights"],
                    sizes=state["muscle"]["sizes"],
                    learningStd=state["muscle"]["learningStd"],
                )
            
        def save_checkpoint(self, checkpoint_path, str=None):
            if str == None:
                print(f"Saving checkpoint at path {checkpoint_path}")
                RLTrainer.save_checkpoint(self, checkpoint_path)
            else:
                with open(Path(checkpoint_path) / f"{str}_checkpoint", "wb") as f:
                    pickle.dump(self.__getstate__(), f)
            return checkpoint_path

        def load_checkpoint(self, checkpoint_path):
            print(f"Loading checkpoint at path {checkpoint_path}")
            checkpoint_file = list(Path(checkpoint_path).glob("checkpoint-*"))
            if len(checkpoint_file) == 0:
                raise RuntimeError("Missing checkpoint file!")
            RLTrainer.load_checkpoint(self, checkpoint_file[0])

    return MyTrainer


CONFIG = None
def get_config_from_file(filename: str, config: str):
    exec(open(filename).read(), globals())
    config = CONFIG[config]
    return config


parser = argparse.ArgumentParser()
parser.add_argument("--cluster", action="store_true")
parser.add_argument("--config", type=str, default="ppo_small_pc")
parser.add_argument("--config-file", type=str, default="learning/ray_config.py")
parser.add_argument("-n", "--name", type=str)
parser.add_argument("--env", type=str, default="data/env.xml")
parser.add_argument("--checkpoint", type=str, default=None)

if __name__ == "__main__":
    env_path = None
    checkpoint_path = None
    args = parser.parse_args()
    print("Argument : ", args)

    env_xml = Path(args.env).resolve()

    # read all text from the file
    env_str = None
    with open(env_xml, "r") as file:
        env_str = file.read()

    if args.cluster:
        ray.init(address=os.environ["ip_head"])
    else:
        if "node" in args.config:
            ray.init(num_cpus=128)
        else:
            ray.init()

    print("Nodes in the Ray cluster:")
    print(ray.nodes())

    config = get_config_from_file(args.config_file, args.config)
    ModelCatalog.register_custom_model("MyModel", SimulationNN_Ray)

    register_env("MyEnv", lambda config: MyEnv(env_str))
    print(f"Loading config {args.config} from config file {args.config_file}.")

    config["rollout_fragment_length"] = config["train_batch_size"] / (config["num_workers"] * config["num_envs_per_worker"])
    config["env_str"] = env_str

    with MyEnv(env_str) as env:
        config["actuator_type"] = env.actuator_type
        config["policyNN"] = {"num_obs" : env.num_obs, "num_actions" : env.num_action}

        if config["actuator_type"].find("mass") != -1:
            config["muscleNN"] = {}
            config["muscleNN"]["num_tau_des"] = len(env.get_zero_action())
            config["muscleNN"]["num_muscles"] = env.muscles.getNumMuscles()
            config["muscleNN"]["num_reduced_JtA"] = env.muscles.getNumMuscleRelatedDofs()

    local_dir = "./ray_results"
    algorithm = config["trainer_config"]["algorithm"]
    MyTrainer = create_my_trainer(algorithm)

    from ray.tune import CLIReporter

    tune.run(
        MyTrainer,
        name=args.name,
        config=config,
        local_dir=local_dir,
        restore=checkpoint_path,
        progress_reporter=CLIReporter(max_report_frequency=25),
        checkpoint_freq=200,
    )

    ray.shutdown()
