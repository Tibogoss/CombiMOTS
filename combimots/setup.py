from pathlib import Path
from setuptools import find_packages, setup

# Load version number
__version__ = ''
version_file = Path(__file__).parent.absolute() / 'pmcts' / '_version.py'

with open(version_file) as fd:
    exec(fd.read())

# Load README
with open('../README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='combimots',
    version=__version__,
    author='Anonymous Authors - ICML2025 Submission n°16227',
    description='Combinatorial Pareto MCTS implementation for dual-target molecule generation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://anonymous.4open.science/r/CombiMOTS-0FEB/',
    download_url=f'',
    license='MIT',
    packages=find_packages(),
    package_data={'pmcts': ['py.typed', 'resources/**/*', 'docking/*.pdbqt']},
    entry_points={
        'console_scripts': [
            'pmcts=pmcts.generate.generate:generate_command_line',
            'pmcts-validate-docking=pmcts.docking.validate:main',
            'combimots-setup-docking=pmcts.docking.setup_quickvina:main',
            'combimots-preprocess=preprocess.runner:main'
        ]
    },
    install_requires=[
        'chemfunc',
        'chemprop==1.5.2',
        'descriptastorus',
        'matplotlib',
        'numpy',
        'pandas',
        'paretoset==1.2.4',
        'pytdc',
        'torch',
        'scikit-learn',
        'scipy',
        'tqdm',
        'typed-argument-parser>=1.8.0'
    ],
    python_requires='>=3.10',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Typing :: Typed'
    ],
    keywords=[
        'Dual-target Molecule Generation', 
        'Fragment-based Drug Discovery', 
        'Monte-Carlo Tree Search', 
        'Pareto Optimization', 
        'Search Space Reduction'
    ]
)
