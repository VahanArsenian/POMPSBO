from distutils.core import setup

setup(
    name='npsem',
    packages=['npsem', 'npsem.experiments'],
    version="0.1.0",
    author='Sanghack Lee',
    author_email='sanghack.lee@gmail.com', requires=['numpy', 'scipy', 'joblib', 'matplotlib', 'seaborn', 'networkx']
)
