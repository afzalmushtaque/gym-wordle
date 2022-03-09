import gym
from gym import spaces
import numpy as np
import random
import pkg_resources
from typing import Optional
import colorama
from colorama import Fore
from colorama import Style
import logging


colorama.init(autoreset=True)


# global game variables
GAME_LENGTH = 6
WORD_LENGTH = 5


def visualize(observation):
    formed_observation = np.reshape(observation[:-1], newshape=(5, 26))
    # formed_observation = np.argmax(formed_observation, axis=-1)
    # formed_observation = observation
    header = 'Step: ' + str(observation[-1] + 1) + ' | '
    for i in range(26):
        header += chr(97 + i) + ' | '
    print('-' * len(header))
    print(header)
    print('-' * len(header))
    for i in range(5):
        print_string = 'Word: ' + str(i + 1) + ' |'
        for j in range(26):
            if formed_observation[i][j] == 0:
                encoding = '  '
            elif formed_observation[i][j] == 1:
                encoding = ' X'
            else:
                raise Exception('Invalid state')
            print_string += encoding + ' |'
        print(print_string)
    print('-' * len(header))

def encodeToStr(encoding):
    return ''.join([chr(char + 97) for char in encoding])


class WordleEnv(gym.Env):
    def __init__(self, **kwargs):
        super(WordleEnv, self).__init__()
        self.guess_npy = np.load(pkg_resources.resource_filename('gym_wordle', 'data/guess_list.npy')).astype(np.int32) - 1
        self.solution_npy = np.load(pkg_resources.resource_filename('gym_wordle', 'data/solution_list.npy')).astype(np.int32) - 1
        self.guess_words = []
        for i in range(self.guess_npy.shape[0]):
            self.guess_words.append(''.join([chr(97+x) for x in self.guess_npy[i]]))
        self.solution_words = []
        for i in range(self.solution_npy.shape[0]):
            self.solution_words.append(''.join([chr(97+x) for x in self.solution_npy[i]]))
        
        self.action_space = spaces.Discrete(self.guess_npy.shape[0])
        observation_space_vector = [4] * WORD_LENGTH * 26
        observation_space_vector.append(GAME_LENGTH + 1)
        self.observation_space = spaces.Dict({
            'valid_avail_actions_mask': gym.spaces.Box(0, 1, shape=(self.guess_npy.shape[0],)),
            'observation': spaces.MultiDiscrete(nvec=observation_space_vector)
        })
        
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.hidden_word = random.choice([self.solution_npy[x] for x in seed])
        else:
            self.hidden_word = self.solution_npy[np.random.randint(self.solution_npy.shape[0])]
        self.board_row_idx = 0
        self.state = np.ones(shape=(WORD_LENGTH, 26), dtype=np.int32)
        self.board = np.negative(np.ones(shape=(GAME_LENGTH, WORD_LENGTH,), dtype=np.int32))
        self.guesses = []
        self.remaining_words = self.solution_words.copy()
        self.valid_avail_actions_mask = np.array([1.0] * self.guess_npy.shape[0], dtype=np.float32)
        return self._get_obs()

    def step(self, action):
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
                # logging.debug('Alphabet "' + chr(97 + char) + '" does not exist in the remaining word at any non solved index. Rejecting it across the word.')
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
        else:
            if self.board_row_idx == GAME_LENGTH:
                done = True
            else:
                done = False

        mask = self.board[self.board_row_idx - 1]
        guess_word = encodeToStr(action)
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
                            logging.error('Exception occured for mask==2')
                            logging.error(e)
                            logging.error('self.hidden_word:')
                            logging.error(encodeToStr(self.hidden_word))
                            logging.error('guess_word:')
                            logging.error(guess_word)
                            logging.error('mask')
                            logging.error(mask)
                            logging.error('self.remaining_words:')
                            logging.error(self.remaining_words)
                            logging.error('solution_words_copy:')
                            logging.error(solution_words_copy)
                            logging.error('solution_word:')
                            logging.error(solution_word)
                            logging.error('tracking_word:')
                            logging.error(tracking_word)
                            logging.error('index:')
                            logging.error(index)
                            logging.error('character:')
                            logging.error(character)
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
                            logging.error('Exception occured for mask==1')
                            logging.error(e)
                            logging.error('self.hidden_word:')
                            logging.error(encodeToStr(self.hidden_word))
                            logging.error('guess_word:')
                            logging.error(guess_word)
                            logging.error('mask')
                            logging.error(mask)
                            logging.error('self.remaining_words:')
                            logging.error(self.remaining_words)
                            logging.error('solution_words_copy:')
                            logging.error(solution_words_copy)
                            logging.error('solution_word:')
                            logging.error(solution_word)
                            logging.error('tracking_word:')
                            logging.error(tracking_word)
                            logging.error('index:')
                            logging.error(index)
                            logging.error('character:')
                            logging.error(character)
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
        self.state = np.zeros(shape=(WORD_LENGTH, 26), dtype=np.int32)
        logging.debug('{:,.0f} words remaining:'.format(len(self.remaining_words)))
        logging.debug(self.remaining_words)
        for word in self.remaining_words:
            for i in range(WORD_LENGTH):
                char_value = ord(word[i]) - 97
                self.state[i, char_value] = 1
        info_eff_cost = len(self.remaining_words) * 1000 / (2315 * 6)
        act_eff_cost = (10 - np.sum(self.board[self.board_row_idx - 1])) / 60
        total_cost = info_eff_cost + act_eff_cost
        logging.debug('info_eff_cost: {:,.4f}'.format(info_eff_cost))
        logging.debug('act_eff_cost: {:,.4f}'.format(act_eff_cost))
        logging.debug('total_cost: {:,.4f}'.format(total_cost))
        return self._get_obs(), -total_cost, done, {}

    def _get_obs(self):
        return {
            'valid_avail_actions_mask': self.valid_avail_actions_mask,
            'observation': np.hstack((self.state.flatten(), self.board_row_idx)) ,
        }
        
    
    def render(self, mode="human"):
        assert mode in ["human"], "Invalid mode, must be \"human\""
        print('###################################################')
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
        print('###################################################')
        print()

if __name__ == "__main__":
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    env = WordleEnv()
    obs = env.reset(seed=[10])
    visualize(obs['observation'])
    step = 0
    print('Hidden word:')
    print(encodeToStr(env.hidden_word))
    done = False
    total_reward = 0
    while not done:
        guess = input('Enter your guess: ')
        act = int(guess)
        # act = np.array(strToEncode([guess])[0])
        # act = np.array([ord(x) - 97 for x in guess])
        obs, reward, done, _ = env.step(act)
        visualize(obs['observation'])
        step += 1
        total_reward += reward
        print('Guesses left: {:,.0f}'.format(GAME_LENGTH - env.board_row_idx))
        print('Reward: {0:,.3f}, Total Reward: {1:,.3f}'.format(reward, total_reward))
        print('Attempted ' + encodeToStr(env.guess_npy[act]))
        env.render()
    print('Hidden word:')
    print(encodeToStr(env.hidden_word))