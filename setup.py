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
    version="0.2.0a",
    author="Justin Svegliato, Abhinav Bhatia",
    author_email="justin.svegliato@gmail.com, bhatiaabhinav93@gmail.com",
    description=("Model Free Metareasoning Experiments"),
    keywords="Metareasoning, Planning, Reinforcement Learning",
    url="https://github.com/justinsvegliato/model-free-metareasoning",
    packages=['MFMR'],
    long_description=read('README.md'),
    python_requires='>=3.6',
    install_requires=[
        'gym>=0.17.1',
        'matplotlib>=3.2.0',
        'numpy>=1.18.1',
        'pytest>=5.4.1',
        'scipy>=1.4.1',
    ]
)
