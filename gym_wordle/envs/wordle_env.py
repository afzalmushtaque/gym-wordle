import gym
from gym import spaces
import numpy as np
import random
import pkg_resources
from typing import Optional
import colorama
from colorama import Fore, Style
import logging


colorama.init(autoreset=True)


# global game variables
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
    def __init__(self, ordered=False, simple_reward=False):
        super(WordleEnv, self).__init__()
        self.ordered = ordered
        self.simple_reward = simple_reward
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
        
    def reset(self, game_number=None, seed = None):
        if seed:
            np.random.seed(seed)
            random.seed(seed)
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
        self.hidden_word = self.solution_npy[game_index]
        self.board_row_idx = 0
        self.state = np.ones(shape=(WORD_LENGTH, ALPHABET_LENGTH), dtype=np.int32)
        self.board = np.negative(np.ones(shape=(GAME_LENGTH, WORD_LENGTH,), dtype=np.int32))
        self.guesses = []
        self.remaining_words = self.solution_words.copy()
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
        action = self.guess_npy[action]        
        hw = list(self.hidden_word)
        solved_indexes = []
        for idx, char in enumerate(action):
            if char == self.hidden_word[idx]:
                # in the correct location
                self.board[self.board_row_idx, idx] = 2
                hw.remove(char)
                solved_indexes.append(idx)
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

        mask = self.board[self.board_row_idx - 1]
        guess_word = decode(action)
        solution_words_copy = self.remaining_words.copy()
        
        for solution_word in self.remaining_words:
            tracking_word = list(solution_word)
            for index, character in enumerate(guess_word):
                if mask[index]==2:
                    if character!=solution_word[index]:
                        solution_words_copy.remove(solution_word)
                        break
                    else:
                        try:
                            tracking_word.remove(character)
                        except Exception as e:
                            logging.error(e)
                            raise e
            if solution_word not in solution_words_copy:
                continue
            for index, character in enumerate(guess_word):    
                if mask[index]==1:
                    if character not in tracking_word or character==solution_word[index]:
                        solution_words_copy.remove(solution_word)
                        break
                    else:
                        try:
                            tracking_word.remove(character)
                        except Exception as e:
                            logging.error(e)
                            raise e
                elif mask[index]==0:
                    if character in tracking_word:
                        solution_words_copy.remove(solution_word)
                        break
                # else:
                    # raise Exception('Invalid mask "' + mask[index] + '" specified')

        self.remaining_words = solution_words_copy
        self.valid_avail_actions_mask = np.array([0.0] * self.guess_npy.shape[0], dtype=np.float32)
        for index, word in enumerate(self.guess_words):
            if word in self.remaining_words:
                self.valid_avail_actions_mask[index] = 1
        self.state = np.zeros(shape=(WORD_LENGTH, ALPHABET_LENGTH), dtype=np.int32)
        info['remaining_words'] = self.remaining_words
        for word in self.remaining_words:
            for i in range(WORD_LENGTH):
                char_value = ord(word[i]) - 97
                self.state[i, char_value] = 1

        # Setting state = 2 for alphabets which are the only possibilities at a given character index
        for char_index in range(WORD_LENGTH):
            assert np.sum(self.state[char_index, :]) > 0, "At least one character from the alphabet must be present for character index " + str(char_index + 1)
            if np.sum(self.state[char_index, :])==1:
                self.state[char_index, :] = np.where(self.state[char_index, :]==1, 2, 0)

        if self.simple_reward:
            reward = -1
            if done and not won:
                reward = -2
        else:
            info_eff_cost = len(self.remaining_words) * 1000 / (2315 * 6)
            act_eff_cost = (10 - np.sum(self.board[self.board_row_idx - 1])) / 60
            total_cost = info_eff_cost + act_eff_cost
            logging.debug('info_eff_cost: {:,.4f}'.format(info_eff_cost))
            logging.debug('act_eff_cost: {:,.4f}'.format(act_eff_cost))
            logging.debug('total_cost: {:,.4f}'.format(total_cost))
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
                letter = chr(97 + self.guesses[i][j])
                if self.board[i][j] == 0:
                    print(Fore.BLACK + Style.BRIGHT + letter + " ", end='')
                elif self.board[i][j] == 1:
                    print(Fore.YELLOW + Style.BRIGHT + letter + " ", end='')
                elif self.board[i][j] == 2:
                    print(Fore.GREEN + Style.BRIGHT + letter + " ", end='')
            print()
        print()
        print('##########')
        print()

if __name__ == "__main__":
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    env = WordleEnv(ordered=True)
    obs = env.reset()
    visualize(obs['observation'])
    step = 0
    print('Hidden word:')
    print(decode(env.hidden_word))
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

        obs, reward, done, _ = env.step(action)
        visualize(obs['observation'])
        step += 1
        total_reward += reward
        print('Guesses left: {:,.0f}'.format(GAME_LENGTH - env.board_row_idx))
        print('Reward: {0:,.3f}, Total Reward: {1:,.3f}'.format(reward, total_reward))
        print('Attempted ' + decode(env.guess_npy[action]))
        env.render()
    print('Hidden word was \'' + decode(env.hidden_word) + '\'')