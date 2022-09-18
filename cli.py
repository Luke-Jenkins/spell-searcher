# sets ups the CLI search tool for the user
import click
from pathlib import Path
import os
# import csv


class Config(object):
    def __init__(self):
        self.db_path = Path(os.getcwd() + r'/spells.csv')


pass_config = click.make_pass_decorator(Config, ensure=True)


# @click.group()
# @click.option('--verbose', is_flag=True)
# @click.option('--home-directory', type=click.Path())
# @pass_config
# def scry(config, verbose, home_directory):
#     config.verbose = verbose
#     if home_directory is None:
#         home_directory = '.'
#     config.home_directory = home_directory


@click.command()
@click.option(
    '--description', default=False,
    help='If true, searches by description.')
# @click.option(
#     '--repeat', default=1,
#     help='How many times you should be greeted.')
@click.argument('term')
@pass_config
def scry(Config, description, term):
    """This script searches for a spell.\n
    By default, spells are searched by title."""
    # if config.verbose:
    #     click.echo('We are in verbose mode.')
    # click.echo(f'Home directory is {config.home_directory}')
    if description is True:
        click.echo('Search by description.')
    else:
        click.echo('Search by title.')
