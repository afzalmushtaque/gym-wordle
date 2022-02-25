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

ALPHABET_STATUS_TO_INT_LOOKUP = {}
INT_TO_ALPHABET_STATUS_LOOKUP = {}

for i in range(4):
    for j in range(27):
        ALPHABET_STATUS_TO_INT_LOOKUP[(i, j)] = 27 * i + j
        INT_TO_ALPHABET_STATUS_LOOKUP[27 * i + j] = {'status': i, 'alphabet': j}


# load words and then encode
filename = pkg_resources.resource_filename(
    'gym_wordle',
    'data/5_words.txt'
)

def custom_distance(source, target):
    distance = 0
    for index, char in enumerate(source):
        if char in target:
            if char!=target[index]:
                distance += 1
        else:
            distance += 2
    return distance

def encodeToStr(encoding):
    string = ""
    for enc in encoding:
        string += chr(97 + enc)
    return string

def strToEncode(lines):
    encoding = []
    for line in lines:
        encoding.append(tuple(ord(char) - 97 for char in line.strip()))
    return encoding


with open(filename, "r") as f:
    WORDS = strToEncode(f.readlines())
    WORDS = [WORDS[x] for x in [1000, 2000, 3000, 4000, 5000, 6000]]
    # random.shuffle(WORDS)
    # WORDS = WORDS[:1]


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

    def __init__(self):
        super(WordleEnv, self).__init__()
        self.action_space = spaces.MultiDiscrete([26] * WORD_LENGTH)
        self.observation_space = spaces.Box(low=-1, high=2, shape=(5, 26,))

    def reset(self, seed: Optional[int] = None):
        # super().reset(seed=seed)
        if seed:
            self.hidden_word = random.choice([WORDS[x] for x in seed])
        else:
            self.hidden_word = random.choice(WORDS)
        # self.guesses_left = GAME_LENGTH
        self.board_row_idx = 0
        self.state = np.negative(np.ones(shape=(WORD_LENGTH, 26), dtype=int))
        self.board = np.negative(np.ones(shape=(GAME_LENGTH, WORD_LENGTH,), dtype=int))
        self.guesses = []
        self.scores = [self.state.sum()]
        return self._get_obs()

    def step(self, action):
        # action = WORDS[action]
        # assert self.action_space.contains(action)

        # Action must be a valid word
        # if not tuple(action) in WORDS:
        #     return self._get_obs(), -11, True, {}
        # update game board
        hw = list(self.hidden_word)
        solved_indexes = []
        for idx, char in enumerate(action):
            if char not in hw:
                logging.debug('Alphabet "' + chr(97 + char) + '" does not exist in the remaning word at any non solved index. Rejecting it across the word.')
                for i in range(WORD_LENGTH):
                    if i not in solved_indexes:
                        self.state[i, char] = 0
                self.board[self.board_row_idx, idx] = 0
            else:
                if char != self.hidden_word[idx]:
                    self.board[self.board_row_idx, idx] = 1
                    hw.remove(char)
                    self.state[idx, char] = 0
                    self.state[:, char] = np.where(self.state[:, char]==-1, 1, self.state[:, char])
                else:
                    # guessed ith alphabet correctly
                    self.board[self.board_row_idx, idx] = 2
                    self.state[:, char] = np.where(self.state[:, char]==1, -1, self.state[:, char]) # maybes for the same alphabet become unknowns
                    self.state[idx, :] = 0 # all other alphabets become incorrect on this word index
                    self.state[idx, char] = 2 
                    hw.remove(char)
                    solved_indexes.append(idx)
        deductions_pending = True
        while(deductions_pending):
            deductions_pending = False
            for i in range(26):
                if (self.state[:, i]==0).sum()==4 and (self.state[:, i]==1).sum()==1:
                    for j in range(WORD_LENGTH):
                        if self.state[j, i]==1:
                            logging.debug('Deducing alphabet "' + chr(97 + i) + '" at position ' + str(j + 1) + ' by elimination and marking all other alphabets at this location as incorrect.')
                            self.state[j, :] = 0
                            self.state[j, i] = 2
                            deductions_pending = True
                            break # can only happen once with an alphabet so no need to continue the for loop
        
        possible_alphabets = []
        known_alphabets = []
        for i in range(26):
            if (self.state[:, i]==1).any():            
                possible_alphabets.append(i)
            for j in range(5):
                if self.state[j, i]==2:            
                    known_alphabets.append(i)
        if len(possible_alphabets) == 5 - len(known_alphabets):
            for i in range(26):
                if i not in possible_alphabets and i not in known_alphabets:
                    logging.debug('Eliminating alphabet "' + chr(97 + i) + '" from consideration as all correct alphabets are either known or possible.')
                    self.state[:, i] = 0
        # reward = self.score - self.previous_score
        # reward -= 1/6
        
        # update guesses remaining tracker
        self.board_row_idx += 1

        # update previous guesses made
        self.guesses.append(action)

        # check to see if game is over
        if all(self.board[self.board_row_idx - 1, :] == 2):
            done = True
        else:
            if self.board_row_idx == GAME_LENGTH:
                done = True
            else:
                done = False
        # if done:
            # reward += self.board[self.board_row_idx, :].sum()
        self.scores.append(self.state.sum())
        reward = -custom_distance(encodeToStr(action), encodeToStr(self.hidden_word))
        # reward = self.scores[-1] - self.scores[-2]
        # reward -= 40/6
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return self.state
    

    def render(self, mode="human"):
        assert mode in ["human"], "Invalid mode, must be \"human\""
        print('###################################################')
        for i in range(len(self.guesses)):
            for j in range(WORD_LENGTH):
                letter = chr(ord('a') + self.guesses[i][j])
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
