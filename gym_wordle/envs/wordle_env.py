from dis import dis
from math import dist
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
from ray.rllib import VectorEnv


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
                encoding = ' -'
            elif formed_observation[i][j] == 2:
                encoding = ' ?'
            elif formed_observation[i][j] == 3:
                encoding = ' R'
            else:
                raise Exception('Invalid state')
            print_string += encoding + ' |'
        print(print_string)
    print('-' * len(header))

def encodeToStr(encoding):
    return ''.join([chr(char + 97) for char in encoding])



class WordleEnv(gym.Env):
    """
    Simple Wordle Environment

    Wordle is a guessing game where the player has 6 guesses to guess the
    5 letter hidden word. After each guess, the player gets feedback on the
    board regarding the word guessed. For each character in the guessed word:
        * if the character is not in the hidden word, the character is
          grayed out (encoded as 0 in the environment)
        * if the character is in the hidden word but not in correct
          location, the character is yellowed out (encoded as 1 in the
          environment)
        * if the character is in the hidden word and in the correct
          location, the character is greened out (encoded as 2 in the
          environment)

    The player continues to guess until they have either guessed the correct
    hidden word, or they have run out of guesses.

    The environment is structured in the following way:
        * Action Space: the action space is a length 5 MulitDiscrete where valid values
          are [0, 25], corresponding to characters [a, z].
        * Observation Space: the observation space is dict consisting of
          two objects:
          - board: The board is 6x5 Box corresponding to the history of
            guesses. At the start of the game, the board is filled entirely
            with -1 values, indicating no guess has been made. As the player
            guesses words, the rows will fill up with values in the range
            [0, 2] indicating whether the characters are missing in the
            hidden word, in the incorrect position, or in the correct position
			based on the most recent guess.
          - alphabet: the alphabet is a length 26 Box corresponding to the guess status
            for each letter in the alaphabet. As the start, all values are -1, as no letter
            has been used in a guess. As the player guesses words, the letters in the
            alphabet will change to values in the range [0, 2] indicating whether the
            characters are missing in the hidden word, in the incorrect position,
            or in the correct position.
    """

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
        self.observation_space = spaces.MultiDiscrete(nvec=observation_space_vector)
        
    def shortlist_remaining_words(self, guess_word, mask):
        solution_words_copy = self.remaining_words.copy()
        for solution_word in self.remaining_words:
            tracking_word = list(solution_word)
            for index, character in enumerate(guess_word):
                if mask[index]==2:
                    if character!=solution_word[index]:
                        solution_words_copy.remove(solution_word)
                        break
                    else:
                        tracking_word.remove(character)
                elif mask[index]==1:
                    if character not in tracking_word:
                        solution_words_copy.remove(solution_word)
                        break
                    else:
                        tracking_word.remove(character)
                elif mask[index]==0:
                    if character in tracking_word:
                        solution_words_copy.remove(solution_word)
                        break
                else:
                    raise Exception('Invalid mask "' + mask[index] + '" specified')
        self.remaining_words = solution_words_copy
        print(self.remaining_words)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.hidden_word = random.choice([self.solution_npy[x] for x in seed])
        else:
            self.hidden_word = self.solution_npy[np.random.randint(self.solution_npy.shape[0])]
        self.board_row_idx = 0
        self.state = np.zeros(shape=(WORD_LENGTH, 26), dtype=np.int32)
        self.board = np.negative(np.ones(shape=(GAME_LENGTH, WORD_LENGTH,), dtype=np.int32))
        self.guesses = []
        self.remaining_words = self.solution_words.copy()
        return self._get_obs()

    def step(self, action):
        action = self.guess_npy[action]        
        hw = list(self.hidden_word)
        solved_indexes = []
        for idx, char in enumerate(action):
            if char == self.hidden_word[idx]:
                # in the correct location
                self.board[self.board_row_idx, idx] = 2
                self.state[:, char] = np.where(self.state[:, char]==2, 0, self.state[:, char]) # maybes for the same alphabet become unknowns
                self.state[idx, :] = 1 # all other alphabets become incorrect on this word index
                self.state[idx, char] = 3 
                hw.remove(char)
                solved_indexes.append(idx)
        for idx, char in enumerate(action):
            if idx in solved_indexes:
                continue
            if char not in hw:
                logging.debug('Alphabet "' + chr(97 + char) + '" does not exist in the remaining word at any non solved index. Rejecting it across the word.')
                for i in range(WORD_LENGTH):
                    self.state[:, char] = np.where(self.state[:, char]==0, 1, self.state[:, char])
                    self.state[idx, char] = 1
                self.board[self.board_row_idx, idx] = 0
            else:
                # alphabet is present in the word
                assert char != self.hidden_word[idx]
                # in the wrong location
                self.board[self.board_row_idx, idx] = 1
                hw.remove(char)
                self.state[idx, char] = 1
                self.state[:, char] = np.where(self.state[:, char]==0, 2, self.state[:, char])

        deductions_pending = True
        while(deductions_pending):
            deductions_pending = False
            for i in range(26):
                if (self.state[:, i]==1).sum()==4 and (self.state[:, i]==2).sum()==1:
                    for j in range(WORD_LENGTH):
                        if self.state[j, i]==2:
                            logging.debug('Deducing alphabet "' + chr(97 + i) + '" at position ' + str(j + 1) + ' by elimination and marking all other alphabets at this location as incorrect.')
                            self.state[j, :] = 1
                            self.state[j, i] = 3
                            deductions_pending = True
                            break # can only happen once with an alphabet so no need to continue the for loop
            
        # update guesses remaining tracker
        self.board_row_idx += 1
        # update previous guesses made
        self.guesses.append(action)
        # information_gain = 0
        # for i in range(WORD_LENGTH):
        #     for j in range(26):
        #         if self.state[i, j]==self.prev_state[i, j]:
        #             information_gain -=0.01
        #         else:
        #             if self.state[i, j]==3:
        #                 information_gain += 0.5
        #             elif self.state[i, j]==2:
        #                 information_gain += 0.1
        #             elif self.state[i, j]==1:
        #                 information_gain += 0.03
        #             elif self.state[i, j]==0:
        #                 information_gain += 0.01
        #             else:
        #                 raise Exception('Impossible situation while calculating information gain.')
        if all(self.board[self.board_row_idx - 1, :] == 2):
            done = True
        else:
            if self.board_row_idx == GAME_LENGTH:
                done = True
            else:
                done = False
        self.prev_state = self.state.copy()
        # if done:
        #     reward = np.sum(self.board[self.board_row_idx - 1])
        # else:
        #     reward = 0
        # scaled_rewards = np.round(np.power(np.exp(self.board[self.board_row_idx-1]), 2.308), 0) - 1 # [0, 9, 100]
        # reward -= ((WORD_LENGTH * 100) - scaled_rewards.sum()) / (WORD_LENGTH * 100 * GAME_LENGTH)
        
        self.shortlist_remaining_words(encodeToStr(action), self.board[self.board_row_idx - 1])

        return self._get_obs(), -len(self.remaining_words), done, {}

    def _get_obs(self):
        return np.hstack((self.state.flatten(), self.board_row_idx)) 
    
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
    visualize(obs)
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
        visualize(obs)
        step += 1
        total_reward += reward
        print('Guesses left: {:,.0f}'.format(GAME_LENGTH - env.board_row_idx))
        print('Reward: {0:,.3f}, Total Reward: {1:,.3f}'.format(reward, total_reward))
        print('Attempted ' + encodeToStr(env.guess_npy[act]))
        env.render()
    print('Hidden word:')
    print(encodeToStr(env.hidden_word))