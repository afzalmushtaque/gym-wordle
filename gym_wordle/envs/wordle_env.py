import gym
from gym import spaces
import numpy as np
import random
import pkg_resources
from time import time
import colorama
from colorama import Fore, Style
import logging


colorama.init(autoreset=True)


GAME_LENGTH = 6
WORD_LENGTH = 5
ALPHABET_LENGTH = 26
STATES_LENGTH = 3 # A character can either be wrong [0], unknown [1] or correct [2]


def visualize(observation):
    formed_observation = np.reshape(observation[:-1], newshape=(WORD_LENGTH, ALPHABET_LENGTH))
    header = 'Step: ' + str(observation[-1] + 1) + ' | '
    for i in range(ALPHABET_LENGTH):
        header += chr(97 + i) + ' | '
    print('-' * len(header))
    print(header)
    print('-' * len(header))
    for i in range(5):
        print_string = 'Char: ' + str(i + 1) + ' |'
        for j in range(ALPHABET_LENGTH):
            if formed_observation[i][j] == 0:
                encoding = '  '
            elif formed_observation[i][j] == 1:
                encoding = ' ?'
            elif formed_observation[i][j] == 2:
                encoding = ' ✓'
            else:
                raise Exception('Invalid state')
            print_string += encoding + ' |'
        print(print_string)
    print('-' * len(header))


def decode(numpy_array):
    return ''.join([chr(char_code + 97) for char_code in numpy_array])


def encode(string):
    return np.array([ord(char) - 97 for char in string])


class WordleEnv(gym.Env):
    def __init__(self, ordered=False, simple_reward=False, mask_actions=False, seed=None):
        super(WordleEnv, self).__init__()
        if seed:
            np.random.seed(seed)
            random.seed(seed)
        self.ordered = ordered
        self.simple_reward = simple_reward
        self.mask_actions = mask_actions
        if self.ordered:
            self.game_index = -1
        self.guess_npy = np.load(pkg_resources.resource_filename('gym_wordle', 'data/guess_list.npy')).astype(np.int32) - 1
        self.solution_npy = np.load(pkg_resources.resource_filename('gym_wordle', 'data/solution_list.npy')).astype(np.int32) - 1

        self.guess_words = []
        for i in range(self.guess_npy.shape[0]):
            self.guess_words.append(decode(self.guess_npy[i]))

        self.solution_words = []
        for i in range(self.solution_npy.shape[0]):
            self.solution_words.append(decode(self.solution_npy[i]))
        
        self.action_space = spaces.Discrete(self.guess_npy.shape[0])
        
        observation_space_vector = [STATES_LENGTH] * WORD_LENGTH * ALPHABET_LENGTH
        observation_space_vector.append(GAME_LENGTH + 1)
        self.observation_space = spaces.Dict(
            {
                'valid_avail_actions_mask': gym.spaces.Box(0, 1, shape=(self.guess_npy.shape[0],)),
                'observation': spaces.MultiDiscrete(nvec=observation_space_vector)
            }
        )
        
    def reset(self, game_number=None):
        if game_number:
            logging.debug('Loading specified game number...')
            game_index = game_number
        else:
            if self.ordered:
                self.game_index += 1
                if self.game_index==self.solution_npy.shape[0]:
                    self.game_index = 0
                logging.debug('Loading ordered game number...')
                game_index = self.game_index
            else:
                logging.debug('Loading random game number...')
                game_index = np.random.randint(self.solution_npy.shape[0])
        logging.debug('Loaded game number {0}.'.format(game_index))
        self.hidden_word = self.solution_words[game_index]
        self.board_row_idx = 0
        self.state = np.ones(shape=(WORD_LENGTH, ALPHABET_LENGTH), dtype=np.int32)
        self.board = np.negative(np.ones(shape=(GAME_LENGTH, WORD_LENGTH,), dtype=np.int32))
        self.guesses = []
        self.remaining_words: list[str] = self.solution_words.copy()
        self.valid_avail_actions_mask = np.array([1.0] * self.guess_npy.shape[0], dtype=np.float32)
        return self._get_obs()

    def step(self, action):
        info = {}
        if self.valid_avail_actions_mask[action]==0:
            raise ValueError(
                "Chosen action was not one of the non-zero action embeddings",
                action,
                self.valid_avail_actions_mask[action],
            )
        action = self.guess_words[action]        
        hw = list(self.hidden_word)
        solved_indexes = []
        valid_words = self.remaining_words.copy()
        assert len(valid_words) > 0
        for idx, char in enumerate(action):
            if char == self.hidden_word[idx]:
                # in the correct location
                self.board[self.board_row_idx, idx] = 2
                hw.remove(char)
                solved_indexes.append(idx)
                valid_words = [valid_word for valid_word in valid_words if valid_word[idx]==char]
            elif char in self.hidden_word:
                valid_words = [valid_word for valid_word in valid_words if valid_word[idx]!=char and char in valid_word]
            else:
                valid_words = [valid_word for valid_word in valid_words if char not in valid_word]
            assert len(valid_words) > 0

        for idx, char in enumerate(action):
            if idx in solved_indexes:
                continue
            if char not in hw:
                self.board[self.board_row_idx, idx] = 0
            else:
                # alphabet is present in the word
                assert char != self.hidden_word[idx]
                # in the wrong location
                self.board[self.board_row_idx, idx] = 1
                hw.remove(char)
                
        # update guesses remaining tracker
        self.board_row_idx += 1
        # update previous guesses made
        self.guesses.append(action)

        if all(self.board[self.board_row_idx - 1, :] == 2):
            done = True
            won = True
        else:
            if self.board_row_idx == GAME_LENGTH:
                done = True
                won = False
            else:
                done = False
                won = None
        if self.mask_actions:
            self.valid_avail_actions_mask = np.array([0.0] * self.guess_npy.shape[0], dtype=np.float32)
            self.valid_avail_actions_mask[[self.guess_words.index(valid_word) for valid_word in valid_words]] = 1
            assert np.sum(self.valid_avail_actions_mask) > 0 , "At least one valid action must remain in mask"
        else:
            self.valid_avail_actions_mask = np.array([1.0] * self.guess_npy.shape[0], dtype=np.float32)
        self.state = np.zeros(shape=(WORD_LENGTH, ALPHABET_LENGTH), dtype=np.int32)
        info['remaining_words'] = valid_words
        for i in range(WORD_LENGTH):
            valid_ith_alphabets = [ord(valid_word[i]) - 97 for valid_word in valid_words]
            self.state[i, valid_ith_alphabets] = 1
            # Setting state = 2 for alphabets which are the only possibilities at a given character index
            assert np.sum(self.state[i, :]) > 0, "At least one character from the alphabet must be present for character index " + str(i + 1)
            if np.sum(self.state[i, :])==1:
                self.state[i, :] = np.where(self.state[i, :]==1, 2, 0)
        self.remaining_words = valid_words
        if self.simple_reward:
            reward = -1
            if done and not won:
                reward = -2
        else:
            info_eff_cost = len(self.remaining_words) / (2315 * 6)
            act_eff_cost = (10 - np.sum(self.board[self.board_row_idx - 1])) / 60
            total_cost = info_eff_cost + act_eff_cost
            if done and won:
                total_cost -= 10
            
            # logging.debug('info_eff_cost: {:,.4f}'.format(info_eff_cost))
            # logging.debug('act_eff_cost: {:,.4f}'.format(act_eff_cost))
            # logging.debug('total_cost: {:,.4f}'.format(total_cost))
            reward = -total_cost
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        return {
            'valid_avail_actions_mask': self.valid_avail_actions_mask,
            'observation': np.hstack((self.state.flatten(), self.board_row_idx)) ,
        }
    
    def render(self, mode="human"):
        assert mode in ["human"], "Invalid mode, must be \"human\""
        print('##########')
        for i in range(len(self.guesses)):
            for j in range(WORD_LENGTH):
                letter =self.guesses[i][j]
                if self.board[i][j] == 0:
                    print(Fore.BLACK + Style.BRIGHT + letter + " ", end='')
                elif self.board[i][j] == 1:
                    print(Fore.YELLOW + Style.BRIGHT + letter + " ", end='')
                elif self.board[i][j] == 2:
                    print(Fore.GREEN + Style.BRIGHT + letter + " ", end='')
            print()
        print('##########')
        print()

if __name__ == "__main__":
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    env = WordleEnv(ordered=True, simple_reward=True, seed=42)
    obs = env.reset()
    visualize(obs['observation'])
    step = 0
    print('Hidden word:')
    print(env.hidden_word)
    done = False
    total_reward = 0
    while not done:
        is_input_valid = False
        while not is_input_valid:        
            guess = input('Enter your guess: ')
            try:
                action = env.guess_words.index(guess)
                is_input_valid = True      
            except ValueError:
                print('\'' + guess + '\' is not a valid guess. Please try again')
        start = time()
        obs, reward, done, info = env.step(action)
        end = time()
        time_taken = end - start
        print("Time consumed in step: " + str(time_taken))
        visualize(obs['observation'])
        print(info)
        step += 1
        total_reward += reward
        print('Guesses left: {:,.0f}'.format(GAME_LENGTH - env.board_row_idx))
        print('Reward: {0:,.3f}, Total Reward: {1:,.3f}'.format(reward, total_reward))
        print('Attempted ' + decode(env.guess_npy[action]))
        env.render()
    print('Hidden word was \'' + env.hidden_word + '\'')