from core.env import Env
import time
import numpy as np

def main():
    start = time.time()

    env_str = None
    with open("data/env.xml", "r") as file:
        env_str = file.read()
    env = Env(env_str)
    print("Time to create env: ", time.time() - start)
    zero_action = np.zeros(env.num_action)
    start_time = time.time()
    for i in range(10):
        env.reset()
        for i in range(300):
            _, _, done, _ = env.step(zero_action)
        print(time.time() - start_time)
    print("Time: ", time.time() - start_time)
## Arg parser
if __name__ == "__main__":
    # cProfile.run('main()')
    main()
