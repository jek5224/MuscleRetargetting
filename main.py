from viewer.viewer import GLFWApp
from core.env import Env
import sys

## Arg parser
import argparse
parser = argparse.ArgumentParser(description='Muscle Simulation')
parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint_path')
parser.add_argument('--env_path', type=str, default='data/env.xml', help='Env_path')

if __name__ == "__main__":
    args = parser.parse_args()
    app = GLFWApp()

    if args.checkpoint:
        app.loadNetwork(args.checkpoint)
    else:
        env_str = None
        with open(args.env_path, "r") as file:
            env_str = file.read()
        app.setEnv(Env(env_str))

    app.startLoop()
