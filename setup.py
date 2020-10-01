import os

from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="Model-Free-Metareasoning",
    version="1.0.0",
    author="Justin Svegliato, Abhinav Bhatia",
    author_email="justin.svegliato@gmail.com, abhinav.bhatia.me@gmail.com",
    description=("Model Free Metareasoning Experiments"),
    keywords="Metareasoning, Planning, Reinforcement Learning",
    url="https://github.com/bhatiaabhinav/model-free-metareasoning",
    packages=['MFMR'],
    long_description=read('README.md'),
    python_requires='>=3.6',
    install_requires=[
        'gym>=0.17.2',
        'matplotlib>=3.2.0',
        'numpy>=1.19.2',
        'pytest>=5.4.1',
        'scipy>=1.5.2',
        'RL-v2>=3.0.0',
        'wandb>=0.10.2'
    ]
)
