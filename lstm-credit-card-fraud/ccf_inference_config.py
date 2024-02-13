import argparse
import os
from dataclasses import dataclass


DIR_THIS_FILE        : str = os.path.normpath(os.path.join(os.path.realpath(__file__), '..'))
DIR_UDF              : str = os.path.join(DIR_THIS_FILE, 'udf')
DIR_US_BANK_WORKLOAD : str = os.path.join(DIR_THIS_FILE, 'usBankWorkload')


FILE_EXT_TFL  = 'h5'
FILE_EXT_ONNX = 'onnx'


@dataclass
class InferenceConfig:
    model_name      : str  = 'model_160batch'
    csv_file        : str  = 'test_220_100k.csv'
    skip_batches    : int  = 0
    batch_count     : int  = 40
    use_onnxruntime : bool = False


def get_mapper_file_path(file_name: str = 'fitted_mapper.pkl'):
    return os.path.join(DIR_UDF, file_name)


def get_model_file_path(config: InferenceConfig):
    if config.use_onnxruntime:
        file_ext = FILE_EXT_ONNX
    else:
        file_ext = FILE_EXT_TFL
    return os.path.join(DIR_UDF, f'{config.model_name}.{file_ext}')


def get_csv_file_path(config: InferenceConfig):
    return os.path.join(DIR_US_BANK_WORKLOAD, config.csv_file)


def get_config_from_args(description: str) -> InferenceConfig:
    # create default config
    config: InferenceConfig = InferenceConfig()
    # define command line
    arg_parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument('-m', '--model-name',
                            dest='model_name', metavar='<string>',
                            help=f'Model name, which is a file with extension:\n'
                                 f'    tensorflow/keras : {FILE_EXT_TFL}\n'
                                 f'    ONNX runtime     : {FILE_EXT_ONNX}\n'
                                 f'in the directory:\n'
                                 f'    {DIR_UDF}\n'
                                 f'(default: {config.model_name})')
    arg_parser.add_argument('-c', '--csv-file',
                            dest='csv_file', metavar='<string>',
                            help=f'CSV file name in directory:\n'
                                 f'    {DIR_US_BANK_WORKLOAD}\n'
                                 f'(default: {config.csv_file})')
    arg_parser.add_argument('-s', '--skip-batches',
                            dest='skip_batches', metavar='<int>', type=int,
                            help=f'Number of batches to skip\n'
                                 f'(default: {config.skip_batches})')
    arg_parser.add_argument('-b', '--batch-count',
                            dest='batch_count', metavar='<int>', type=int,
                            help=f'Batch count\n'
                                 f'(default: {config.batch_count})')
    arg_parser.add_argument('-o', '--onnxruntime',
                            dest='use_onnxruntime', action='store_true',
                            help=f'use the ONNX runtime instead of tensorflow/keras')
    # get command line args
    args = arg_parser.parse_args()
    if args.model_name:
        config.model_name = args.model_name
    if args.csv_file:
        config.csv_file = args.csv_file
    if args.skip_batches:
        config.skip_batches = args.skip_batches
    if args.batch_count:
        config.batch_count = args.batch_count
    if args.use_onnxruntime:
        config.use_onnxruntime = True
    # return config
    return config
