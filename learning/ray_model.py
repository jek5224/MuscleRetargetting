import torch
import torch.nn as nn
import numpy as np

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import convert_to_torch_tensor

MultiVariateNormal = torch.distributions.Normal
temp = MultiVariateNormal.log_prob
MultiVariateNormal.log_prob = lambda self, val: temp(self, val).sum(-1, keepdim=True)

temp2 = MultiVariateNormal.entropy
MultiVariateNormal.entropy = lambda self: temp2(self).sum(-1)
MultiVariateNormal.mode = lambda self: self.mean

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()

class MuscleNN(nn.Module):
    def __init__(
        self, 
        num_total_muscle_related_dofs, 
        num_dofs, 
        num_muscles, 
        device = "cuda", 
        config={"sizes" : [256,256,256], "learningStd" : False}# [256,256,256]
    ):
        super(MuscleNN, self).__init__()

        self.num_total_muscle_related_dofs = num_total_muscle_related_dofs
        self.num_dofs = num_dofs  
        self.num_muscles = num_muscles
        self.config = config

        layers = []
        prev_size = num_total_muscle_related_dofs + num_dofs
        for size in self.config["sizes"]:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_size = size
        layers.append(nn.Linear(prev_size, num_muscles))
        
        self.fc = nn.Sequential(*layers)

        # Normalization
        self.std_muscle_tau = torch.ones(num_total_muscle_related_dofs) * 200
        self.std_tau = torch.ones(num_dofs) * 200
        
        if self.config["learningStd"]:
            self.std_muscle_tau = nn.Parameter(self.std_muscle_tau)
            self.std_tau = nn.Parameter(self.std_tau)

        if torch.cuda.is_available() and device=="cuda":
            self.cuda()
            self.std_tau = self.std_tau.cuda()
            self.std_muscle_tau = self.std_muscle_tau.cuda()
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.fc.apply(weights_init) ## initialize 

    def forward(self, reduced_JtA, tau) -> torch.Tensor:
        if reduced_JtA.shape == self.std_muscle_tau.shape:
            reduced_JtA = reduced_JtA / self.std_muscle_tau
        else:
            print(reduced_JtA.shape, self.std_muscle_tau.shape)
            print("Dimension of reduced_JtA and tau doesn't match")
            return torch.zeros(self.num_muscles)
        # reduced_JtA = reduced_JtA / self.std_muscle_tau
        tau = tau / self.std_tau

        return torch.relu(torch.tanh(self.fc(torch.cat([reduced_JtA, tau], dim=-1))))

    def get_activation(self, reduced_JtA, tau_des) -> np.array:   
        if not isinstance(reduced_JtA, torch.Tensor):
            reduced_JtA = torch.tensor(reduced_JtA, device=self.fc[0].weight.device, dtype=torch.float32)
        if not isinstance(tau_des, torch.Tensor):
            tau_des = torch.tensor(tau_des, device=self.fc[0].weight.device, dtype=torch.float32) 
        with torch.no_grad():
            return self.forward(reduced_JtA, tau_des).cpu().detach().numpy()


class SimulationNN(nn.Module):
    def __init__(self, 
                 num_states, 
                 num_actions, 
                 config={"sizes" : {"policy" : [512,512,512], "value" : [512,512,512]}, "learningStd" : False} ,
                 device="cuda",
        ):
        super(SimulationNN, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.config = config
        
        p_layers = []
        prev_size = num_states
        for size in self.config["sizes"]["policy"]:
            p_layers.append(nn.Linear(prev_size, size))
            p_layers.append(nn.ReLU(inplace=True))
            prev_size = size

        p_layers.append(nn.Linear(prev_size, num_actions))
        self.p_fc = nn.Sequential(*p_layers)

        v_layers = []
        prev_size = num_states
        for size in self.config["sizes"]["value"]:
            v_layers.append(nn.Linear(prev_size, size))
            v_layers.append(nn.ReLU(inplace=True))
            prev_size = size

        v_layers.append(nn.Linear(prev_size, 1))
        self.v_fc = nn.Sequential(*v_layers)

        self.log_std = None
        if self.config["learningStd"]:
            self.log_std = nn.Parameter(torch.ones(num_actions))
        else:
            self.log_std = torch.ones(num_actions)

        if torch.cuda.is_available() and device == "cuda":
            if not self.config["learningStd"]:
                self.log_std = self.log_std.cuda()
            self.cuda()
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # initialize
        self.p_fc.apply(weights_init)
        self.v_fc.apply(weights_init)

    def forward(self, x):
        p_out = MultiVariateNormal(self.p_fc.forward(x), self.log_std.exp())
        v_out = self.v_fc.forward(x)
        return p_out, v_out


class SimulationNN_Ray(TorchModelV2, SimulationNN):
    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        # from IPython import embed; embed()
        num_states = np.prod(obs_space.shape)
        num_actions = np.prod(action_space.shape)
        SimulationNN.__init__(self, 
                              num_states, 
                              num_actions, 
                              {"sizes" : model_config["custom_model_config"]["sizes"], "learningStd" : model_config["custom_model_config"]["learningStd"]}, 
                              "cuda" if torch.cuda.is_available() else "cpu")
        
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, {}, "SimulationNN_Ray"
        )
        num_outputs = 2 * np.prod(action_space.shape)
        self._value = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        x = obs.reshape(obs.shape[0], -1)
        action_dist, self._value = SimulationNN.forward(self, x)
        action_tensor = torch.cat([action_dist.loc, action_dist.scale.log()], dim=1)
        return action_tensor, state

    def value_function(self):
        return self._value.squeeze(1)


## This class is for integration of simulationNN and ray filter
class PolicyNN:
    def __init__(
        self,
        num_states,
        num_actions,
        policy_state,
        filter_state,
        device,
        config={"sizes" : {"policy" : [512,512,512], "value" : [512,512,512]}, "learningStd" : False} 
    ):
        self.device = device
        self.policy = SimulationNN(num_states=num_states, num_actions=num_actions, config=config, device=device)
        self.policy.load_state_dict(convert_to_torch_tensor(policy_state))
        self.policy.eval()
        self.filter = filter_state

    def get_action(self, obs, is_random=False) -> np.ndarray:
        with torch.no_grad():
            obs = self.filter(obs, update=False)
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
            return (
                self.policy.p_fc.forward(obs).cpu().detach().numpy()
                if not is_random
                else self.policy.forward(obs)[0].sample().cpu().detach().numpy()
            )

import pickle
def loading_network(path, device="cuda") -> (SimulationNN, MuscleNN, str):
    device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    mus_nn = None 
    env_str = None

    state = pickle.load(open(path, "rb"))
    if 'env_str' in state.keys():
        env_str = state['env_str']
        
    worker_state = pickle.loads(state["worker"])
    policy_state = worker_state["state"]["default_policy"]["weights"]
    filter_state = worker_state["filters"]["default_policy"]
    
    policy_config = state["policyNN"]
    if "sizes" not in policy_config.keys():
        policy_config["sizes"] = {"policy" : [512,512,512], "value" : [512,512,512]}
    if "learningStd" not in policy_config.keys():
        policy_config["learningStd"] = False
    
    policy = PolicyNN(
        policy_config["num_obs"], 
        policy_config["num_actions"], 
        policy_state, 
        filter_state, 
        device,    
        {"sizes" : policy_config["sizes"], "learningStd" : policy_config["learningStd"]} 
    )

    if 'muscle' in state.keys():
        mus_nn = MuscleNN(
            state["muscle"]["num_reduced_JtA"],
            state["muscle"]["num_tau_des"],
            state["muscle"]["num_muscles"],
            config={"sizes": state["muscle"]["sizes"], "learningStd": state["muscle"]["learningStd"]}
        )
        mus_nn.load_state_dict(state["muscle"]["weights"])

    return policy, mus_nn, env_str
