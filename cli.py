# sets ups the CLI search tool for the user
import click


# class Config(object):
#     def __init__(self):
#         self.verbose = False


# pass_config = click.make_pass_decorator(Config, ensure=True)


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
    '--string', default='World',
    help='This is the thing that is greeted.')
@click.option(
    '--repeat', default=1,
    help='How many times you should be greeted.')
@click.argument('out', type=click.File('w'), default='-', required=False)
# @pass_config
def scry(string, repeat, out):
    """This script greets you"""
    # if config.verbose:
    #     click.echo('We are in verbose mode.')
    # click.echo(f'Home directory is {config.home_directory}')
    for x in range(repeat):
        click.echo(f'Hello {string}!', file=out)
