import os
import sys
import logging
import functools
from termcolor import colored

@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=''):
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ':' + \
                colored('%(levelname)s', 'red') + ' %(message)s'
    
    # Create console handler
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        )
        logger.addHandler(console_handler)

    # Create file handler
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank_{dist_rank}.log'), 
                                       mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger