import numpy as np


lookup = {i: chr(97 + i) for i in range(26)}

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div


def visualize(observation):
    # formed_observation = np.reshape(observation, newshape=(6, 5, 27, 3))
    formed_observation = observation
    header = 'Step: n | '
    for i in range(26):
        header += chr(97+i) + ' | '
    print('-' * len(header))
    print(header)
    print('-' * len(header))
    for i in range(5):
        print_string = 'Step: ' + str(i) + ' |'
        for j in range(26):
            if formed_observation[i][j] == 0:
                encoding = ' -'
            elif formed_observation[i][j] == 1:
                encoding = ' ?'
            elif formed_observation[i][j] == 2:
                encoding = ' R'
            else:
                encoding = '  '
            print_string += encoding + ' |'
        print(print_string)
    print('-' * len(header))
