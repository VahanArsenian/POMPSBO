import logging
import argparse


parser = argparse.ArgumentParser(description='Arguments for POMPS example graph')
parser.add_argument('--smoke', action='store_true', help='Used to test the code')
parser.add_argument('--no-smoke', dest='smoke', action='store_false')
parser.set_defaults(smoke=True)
parser.add_argument('--n_iter', type=int, help='Number of iterations to be run', default=1500)
parser.add_argument('--seed', type=int, help='Seed for torch, python, and numpy', default=42)
parser.add_argument('--log_file', type=str, help='Log file path',
                    default="pomps_paper_graph0.log")
parser.add_argument('--experiment_name', type=str, help='Experiment name. Used to define artefact names',
                    default="pomps_paper_graph0")

args = vars(parser.parse_args())
print(args)
smoke_test = args['smoke']
n_iter = args['n_iter']
seed = args['seed']
log_file = args['log_file']
experiment_name = args["experiment_name"]


logger = logging.getLogger('pomps_logger')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(log_file)
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(process)d-%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)