from setuptools import setup

setup(
    name='spell-scrier',
    version='1.0',
    py_modules=['cli'],
    install_requires=[
        'Click',
    ],
    entry_points='''
        [console_scripts]
        scry=cli:scry
        '''
)
