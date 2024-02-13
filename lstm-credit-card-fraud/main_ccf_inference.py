import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import joblib
import pandas as pd

# this import must be in the main otherwise the pickle (fitted_mapper.pkl) does not find it
from data_frame_mapper import fraudEncoder, decimalEncoder, timeEncoder, amtEncoder

from ccf_inference_config import (
    InferenceConfig,
    get_mapper_file_path,
    get_model_file_path,
    get_csv_file_path,
    get_config_from_args,
)
from ccf_inference import (
    SEQ_LENGTH,
    BATCH_SIZE,
    load_model,
    run_inference,
)


def read_batch_data_from_csv(data_file, skip_batches, num_batches):
    skip_rows = BATCH_SIZE * SEQ_LENGTH * skip_batches
    num_rows  = BATCH_SIZE * SEQ_LENGTH * num_batches
    data_rows = pd.read_csv(data_file, skiprows=skip_rows, nrows=num_rows)
    return data_rows.values


if __name__ == '__main__':
    config: InferenceConfig = get_config_from_args('Credit Card Fraud Inference')

    print('=================================================================')

    model_file = get_model_file_path(config)
    mapper_file = get_mapper_file_path()
    data_file = get_csv_file_path(config)

    model = load_model(model_file, config.use_onnxruntime)
    mapper = joblib.load(open(mapper_file,'rb'))
    data = read_batch_data_from_csv(data_file, config.skip_batches, config.batch_count)

    num_cases = data.shape[0] // SEQ_LENGTH

    start_time = time.time()
    result = run_inference(model, mapper, data, config.use_onnxruntime)
    inference_time = time.time() - start_time

    print('--------------------------------------')
    num_errors = 0
    for fraud in result:
        print(f'fraud = {fraud[0]:8d}  {fraud[1]:4d}  {fraud[3]}-{fraud[4]:02d}-{fraud[5]:02d}  {fraud[6]}  {fraud[7]:8s}  {fraud[15]:3s}  {fraud[10]}-{fraud[11]}-{fraud[12]}')
        if fraud[15] != 'Yes':
            num_errors += 1

    print('--------------------------------------')
    print(f'Number of cases = {num_cases}')
    print(f'Total number of frauds = {len(result)}')
    print(f'Number of prediction errors = {num_errors}')
    print(f'Inference time = {inference_time:.2f} secs = {1000*inference_time/num_cases:.2f} msecs/case')
