import configparser
import os


def get_config(section, option):
    """
    get data config
    :param section: section
    :param option: option
    :return: value config
    """
    config = configparser.ConfigParser()

    # dir_file = os.path.dirname(__file__)
    config.read('config.ini')

    return config.get(section, option)
