from wordle_env import WordleEnv
from tqdm import tqdm
from time import time
import random
import pprofile
import sys
import logging
import numpy as np


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
profiler = pprofile.Profile()


env = WordleEnv(ordered=True, simple_reward=True)
start = time()
for i in tqdm(range(2315)):
    obs = env.reset()
    done = False
    total_reward = 0
    # with profiler:
    while not done:
        # valid_indexes = [x for x in range(len(obs['valid_avail_actions_mask'])) if obs['valid_avail_actions_mask'][x]==1]
        # obs, reward, done, _ = env.step(random.choice(valid_indexes))
        obs, reward, done, _ = env.step(random.choice(np.where(obs['valid_avail_actions_mask']==1)[0].tolist()))
        total_reward += reward
    # print(total_reward)
    # profiler.dump_stats("profiler_stats3.txt")
    # exit()
end = time()
print('Time taken: ' + str(end - start))