from setuptools import setup

requirements = open('requirements.txt').readlines()

setup(
    name='causation',
    version='0.0.1',
    author='Ricardo Corral Corral',
    author_email='ricardo@suggestic.com',
    description='Suggestic causality dicovery routines',
    py_modules=['causation'],
    install_requires=requirements,
    url='https://github.com/Suggestic/causation_engine.git'
)
