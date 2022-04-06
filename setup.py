from setuptools import setup

setup(name='gym_wordle',
      version='2.1.0',
      install_requires=['gym==0.17.2',
                        'numpy>=1.19.2',
                        'colorama>=0.4.4'],
      package_data={'gym_wordle': ['data/*.npy', 'envs/*']},
      packages=['gym_wordle'],
)
