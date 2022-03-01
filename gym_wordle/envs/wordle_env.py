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


colorama.init(autoreset=True)

# global game variables
GAME_LENGTH = 6
WORD_LENGTH = 5

# load words and then encode
filename = pkg_resources.resource_filename(
    'gym_wordle',
    'data/5_words.txt'
)

def visualize(observation):
    formed_observation = np.reshape(observation[:-1], newshape=(5, 26))
    # formed_observation = np.argmax(formed_observation, axis=-1)
    # formed_observation = observation
    header = 'Step: ' + str(observation[-1] + 1) + ' | '
    for i in range(26):
        header += chr(97+i) + ' | '
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

VOCAB_SIZE = len(WORDS)

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
        self.action_space = spaces.MultiDiscrete([26] * WORD_LENGTH)
        # self.observation_space = spaces.Box(low=-1, high=2, shape=(WORD_LENGTH, 26,))
        observation_space_vector = [4] * WORD_LENGTH * 26
        observation_space_vector.append(GAME_LENGTH)
        self.observation_space = spaces.MultiDiscrete(nvec=observation_space_vector)
        # self.observation_space = spaces.MultiBinary(n=[4] * WORD_LENGTH * 26)
        # self.syllabus = list(np.arange(100))
        self.syllabus = list(np.arange(VOCAB_SIZE))
        self.cheat_mode = kwargs.get('cheat_mode', True)
        

    def expand_syllabus(self):
        if len(self.syllabus) < VOCAB_SIZE:
            logging.info('Expanding syllabus to ' + str(self.syllabus[-1] + 1) + ' words...')
            self.syllabus.append(self.syllabus[-1] + 1)
        else:
            logging.debug('Skipping syllabus expansion as vocab limit ' + str(self.syllabus[-1] + 1) + ' reached...')
    
   
    def reset(self, seed: Optional[int] = None):
        # super().reset(seed=seed)
        if seed is not None:
            self.hidden_word = random.choice([WORDS[x] for x in seed])
        else:
            self.hidden_word = random.choice([WORDS[x] for x in self.syllabus])
        # self.guesses_left = GAME_LENGTH
        self.board_row_idx = 0
        self.prev_state = np.zeros(shape=(WORD_LENGTH, 26), dtype=int)
        self.state = np.zeros(shape=(WORD_LENGTH, 26), dtype=int)
        self.board = np.negative(np.ones(shape=(GAME_LENGTH, WORD_LENGTH,), dtype=int))
        self.guesses = []
        if self.cheat_mode:
            _, _, _, _ = self.evaluate_action(self.hidden_word, cheat_mode=True)
        # self.scores = [self.state.sum()]
        return self._get_obs()

    def evaluate_action(self, action, cheat_mode):
        if cheat_mode:
            action = self.hidden_word
        hw = list(self.hidden_word)
        solved_indexes = []
        distance = 0
        for idx, char in enumerate(action):
            self.prev_state = self.state.copy()
            if char == self.hidden_word[idx]:
                # in the correct location
                if not cheat_mode:
                    self.board[self.board_row_idx, idx] = 2
                self.state[:, char] = np.where(self.state[:, char]==2, 0, self.state[:, char]) # maybes for the same alphabet become unknowns
                self.state[idx, :] = 1 # all other alphabets become incorrect on this word index
                self.state[idx, char] = 3 
                hw.remove(char)
                solved_indexes.append(idx)
                if self.state[idx, char]!=self.prev_state[idx, char]:
                    distance -= 10
        for idx, char in enumerate(action):
            if idx in solved_indexes:
                continue
            self.prev_state = self.state.copy()
            if char not in hw:
                logging.debug('Alphabet "' + chr(97 + char) + '" does not exist in the remaining word at any non solved index. Rejecting it across the word.')
                for i in range(WORD_LENGTH):
                    self.state[:, char] = np.where(self.state[:, char]==0, 1, self.state[:, char])
                    # if i not in solved_indexes:
                    #     self.state[i, char] = 0
                distance += 0.2
                if not idx in solved_indexes and not cheat_mode:
                    self.board[self.board_row_idx, idx] = 0
                
            else:
                # alphabet is present in the word
                if char != self.hidden_word[idx]:
                    # in the wrong location
                    if not cheat_mode:
                        self.board[self.board_row_idx, idx] = 1
                    hw.remove(char)
                    self.state[idx, char] = 1
                    self.state[:, char] = np.where(self.state[:, char]==0, 2, self.state[:, char])
                    distance += 0.1
            # check for changes
            if self.state[idx, char]==self.prev_state[idx, char]:
                # did not gain new information    
                distance += 0.1

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
        
        # possible_alphabets = []
        # known_alphabets = []
        # for i in range(26):
        #     if (self.state[:, i]==1).any():            
        #         possible_alphabets.append(i)
        #     for j in range(5):
        #         if self.state[j, i]==2:            
        #             known_alphabets.append(i)
        # if len(possible_alphabets) == 5 - len(known_alphabets):
        #     for i in range(26):
        #         if i not in possible_alphabets and i not in known_alphabets:
        #             logging.debug('Eliminating alphabet "' + chr(97 + i) + '" from consideration as all correct alphabets are either known or possible.')
        #             self.state[:, i] = 0
        
        # update guesses remaining tracker
        if not cheat_mode:
            self.board_row_idx += 1
            # update previous guesses made
            self.guesses.append(action)
        
        # assert -50 <= np.round(distance, 6) <= 1.5
        # scaled_distance = (distance + 50) / (1.5 + 50)
        # reward = -((scaled_distance * 2) - 1)
        # check to see if game is over
        # reward = -distance / 50
        if all(self.board[self.board_row_idx - 1, :] == 2):
            done = True
        else:
            if self.board_row_idx == GAME_LENGTH:
                done = True
            else:
                done = False
        
        self.prev_state = self.state.copy()
        # reward = -custom_distance(action, self.hidden_word) / 10
        reward = -(10 - self.board[self.board_row_idx-1].sum()) / 60
        return self._get_obs(), reward, done, {}
    
    def step(self, action):
        obs, reward, done, info = self.evaluate_action(action, cheat_mode=False)
        if self.cheat_mode:
            # undo state changes
            state_backup = self.state.copy()
            obs, _, _, _ = self.evaluate_action(action, cheat_mode=True)
            self.state = state_backup.copy()
        return obs, reward, done, info
        

    def _get_obs(self):
        # agent_state = np.empty(shape=(5, 26, 4), dtype=np.bool)
        # for i in range(WORD_LENGTH):
        #     for j in range(26):
        #         agent_state[i, j] = np.eye(4)[self.state[i, j]]
        # return agent_state.reshape(-1, 4)
        self.board_row_idx
        return np.hstack((self.state.flatten(), self.board_row_idx)) 
    
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
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    env = WordleEnv(cheat_mode=True)
    obs = env.reset(seed=[10])
    visualize(obs)
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