import gym
from gym import spaces
import numpy as np
import random
import pkg_resources
from typing import Optional
import colorama
from colorama import Fore
from colorama import Style
from visualizer import visualize
import logging


colorama.init(autoreset=True)

# global game variables
GAME_LENGTH = 6
WORD_LENGTH = 5

# ALPHABET_STATUS_TO_INT_LOOKUP = {}
# INT_TO_ALPHABET_STATUS_LOOKUP = {}

# for i in range(4):
#     for j in range(27):
#         ALPHABET_STATUS_TO_INT_LOOKUP[(i, j)] = 27 * i + j
#         INT_TO_ALPHABET_STATUS_LOOKUP[27 * i + j] = {'status': i, 'alphabet': j}


# load words and then encode
filename = pkg_resources.resource_filename(
    'gym_wordle',
    'data/5_words.txt'
)

def custom_distance(source, target):
    distance = 0
    for index, char in enumerate(source):
        if char in target:
            # alphabet present in hidden word
            if char!=target[index]:
                # but at the wrong location
                distance += 1
        else:
            # alphabet not present in hidden word
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
    WORDS = [WORDS[x] for x in [ 5934,  1408,  5487,  2871, 12660, 12612,  4394,  2145,  3507,
        6281,  2322, 13812,  2695, 10663,  9023,  6413, 14815,  9200,
         346, 12534, 12937,  3055,  8307,  9481, 12548, 12499, 11114,
        9218, 10979, 13564,  5067,  3354,  6375,  4079,  8779,  3440,
       11819,  5872,  9188,  9712,  7459,  3329,  9195, 14081, 14330,
       13477,   771, 10790,  4887,  3538,  4578,  2594,  1111,  2142,
         516,  7287, 10601,  9404, 14485, 13777, 10377, 12240,  2687,
        9043,   418,  4581,   932,  1495,  2859,   148,  8626,  6288,
        4483,  6015,   468,  7038,  9845,   562, 14139,  5879,   394,
       11004,  4948,  8004,  4005,  6121, 14364, 10350, 14237,  5168,
       13685, 12396,  4451,  9456,  9503, 14883,  9217,  6768,  4920,
        5791]
    ]
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
        self.observation_space = spaces.Box(low=-1, high=2, shape=(WORD_LENGTH, 26,))

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
        # self.scores = [self.state.sum()]
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
            if char == self.hidden_word[idx]:
                # in the correct location
                self.board[self.board_row_idx, idx] = 2
                self.state[:, char] = np.where(self.state[:, char]==1, -1, self.state[:, char]) # maybes for the same alphabet become unknowns
                self.state[idx, :] = 0 # all other alphabets become incorrect on this word index
                self.state[idx, char] = 2 
                hw.remove(char)
                solved_indexes.append(idx)

        for idx, char in enumerate(action):
            if idx in solved_indexes:
                continue
            if char not in hw:
                logging.debug('Alphabet "' + chr(97 + char) + '" does not exist in the remaning word at any non solved index. Rejecting it across the word.')
                for i in range(WORD_LENGTH):
                    if i not in solved_indexes:
                        self.state[i, char] = 0
                if not idx in solved_indexes:
                    self.board[self.board_row_idx, idx] = 0
            else:
                # alphabet is present in the word
                if char != self.hidden_word[idx]:
                    # in the wrong location
                    self.board[self.board_row_idx, idx] = 1
                    hw.remove(char)
                    self.state[idx, char] = 0
                    self.state[:, char] = np.where(self.state[:, char]==-1, 1, self.state[:, char])
                    
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
        # self.scores.append(self.state.sum())
        # reward = -custom_distance(encodeToStr(action), encodeToStr(self.hidden_word))
        reward = -custom_distance(action, self.hidden_word)
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

if __name__ == "__main__":
    env = WordleEnv()
    obs = env.reset()
    step = 0
    print('Hidden word:')
    print(encodeToStr(env.hidden_word))
    done = False
    total_reward = 0
    while not done:
        guess = input('Enter your guess: ')
        act = np.array(strToEncode([guess])[0])
        # act = int(guess)
        # make a random guesgoofss
        # act = env.action_space.sample()
        # take a step
        obs, reward, done, _ = env.step(act)
        visualize(obs)
        step += 1
        total_reward += reward
        print('Guesses left: {:,.0f}'.format(GAME_LENGTH - env.board_row_idx))
        print('Reward: {0:,.2f}, Total Reward: {1:,.2f}'.format(reward, total_reward))
        print('Attempted ' + encodeToStr(act))
        env.render()
    print('Hidden word:')
    print(encodeToStr(env.hidden_word))