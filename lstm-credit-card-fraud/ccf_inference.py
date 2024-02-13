import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import onnxruntime as ort
import pandas as pd
import tensorflow as tf

from data_frame_mapper import fraudEncoder, decimalEncoder, timeEncoder, amtEncoder


SEQ_LENGTH = 7     # Six past transactions followed by current transaction
BATCH_SIZE = 160

COLUMN_TITLES = [
    'Index', 'User', 'Card', 'Year', 'Month', 'Day', 'Time', 'Amount',
    'Use Chip', 'Merchant Name', 'Merchant City',
    'Merchant State', 'Zip', 'MCC', 'Errors?', 'Is Fraud?',
]
NUM_COLUMNS = len(COLUMN_TITLES)


# 1 batch = 1120 rows
# 1120 rows is 160 sets of 7 rows
# each set of 7 rows is 6 history rows and a 7th row to be predicted
# the batch prediction gives 160 cases of fraud/no-fraud, one for each set of 7
# given a batch index B in range 0-159, then the predicted row index R = (7 * B + 6)

# can I reorder the data by keeping the sets of 7 together, but sort them by
# increasing date/time of the row to be predicted (i.e. the last row of each set)
# then each batch of 160 sets of 7 will have a date/time range


def load_model(model_file, use_onnxruntime):
    if use_onnxruntime:
        model = ort.InferenceSession(model_file)
        #__print_onnx_inputs_and_outputs(model)
    else:
        model = tf.keras.models.load_model(model_file)
    return model


def run_model(model, mapped_batch_array, use_onnxruntime):
    if use_onnxruntime:
        onnx_inputs = model.get_inputs()
        input_name = onnx_inputs[0].name
        result_list = model.run(None, {input_name: mapped_batch_array.astype(np.float32)})
        prediction = result_list[0]
    else:
        prediction = model.predict(mapped_batch_array)
    return prediction


def run_inference(model, mapper, data, use_onnxruntime):
    #print(f'DEBUG> data shape = {data.shape}')
    #print(f'DEBUG> COLUMN_TITLES = {COLUMN_TITLES}')
    #print(f'DEBUG> data row 0 =\n{data[0]}')
    #print(f'DEBUG> data row 1 =\n{data[1]}')
    #print(f'DEBUG> data row 159 =\n{data[159]}')
    #print(f'DEBUG> data row 160 =\n{data[160]}')
    #print(f'DEBUG> data row 1119 =\n{data[1119]}')
    #print(f'DEBUG> data row 1120 =\n{data[1120]}')

    tdf = pd.DataFrame(data, columns=COLUMN_TITLES)
    #print(tdf.info())
    #print(f'DEBUG> tdf shape = {tdf.shape}')
    # example raw data row = 34,0,0,2002,9,11,19:30,$80.18,Swipe Transaction,-1605794445852049456,La Verne,CA,91750.0,5812,,No

    __clean_data_frame(tdf)
    #print(tdf.info())

    num_rows = tdf.shape[0]
    num_batches = num_rows // (BATCH_SIZE * SEQ_LENGTH)
    #print(f'DEBUG> num_rows = {num_rows}, num_batches = {num_batches}')

    result = []
    for batch_index in range(num_batches):
        #print(f'DEBUG> batch shape {batch_index}')
        batch_tdf = __extract_batch(tdf, batch_index)
        batch_predicted_frauds = __run_batch(model, mapper, batch_tdf, use_onnxruntime)
        result.extend(batch_predicted_frauds)
    return result


def __clean_data_frame(tdf):
    tdf.set_index('Index')
    tdf["Zip"] = tdf["Zip"].astype(float)
    tdf["MCC"] = tdf["MCC"].astype(int)
    tdf["Merchant Name"] = tdf["Merchant Name"].astype(str)
    #tdf["Merchant City"].replace('ONLINE', 'ONLINE', regex=True, inplace=True)
    tdf["Merchant State"].fillna(np.nan, inplace=True)
    tdf["Errors?"].fillna('missing_value', inplace=True)


def __extract_batch(tdf, batch_index):
    batch_start = batch_index * BATCH_SIZE * SEQ_LENGTH
    batch_end = (batch_index + 1) * BATCH_SIZE * SEQ_LENGTH
    batch_tdf = tdf.iloc[batch_start:batch_end, ]
    return batch_tdf


def __map_batch(mapper, batch_tdf):
    #print(f'DEBUG> batch_tdf shape = {batch_tdf.shape}')
    new_df = mapper.transform(batch_tdf).drop(['Is Fraud?'],axis=1)
    #print(f'DEBUG> new_df shape    = {new_df.shape}')
    xbatch = new_df.to_numpy().reshape(BATCH_SIZE, SEQ_LENGTH, -1)
    #print(f'DEBUG> xbatch shape    = {xbatch.shape}')
    xbatch_t = np.transpose(xbatch, axes=(1,0,2))
    #print(f'DEBUG> xbatch_t shape  = {xbatch_t.shape}')
    mapped_batch_array = np.asarray(xbatch_t)
    return mapped_batch_array


def __predict_batch(model, mapped_batch_array, use_onnxruntime):
    if use_onnxruntime:
        onnx_inputs = model.get_inputs()
        input_name = onnx_inputs[0].name
        result_list = model.run(None, {input_name: mapped_batch_array.astype(np.float32)})
        prediction = result_list[0]
    else:
        prediction = model.predict(mapped_batch_array)
    #print(f'DEBUG> prediction.shape = {prediction.shape}')
    # prediction shape = (7, 160, 1)
    batch_prediction = [float(prediction[SEQ_LENGTH-1][k][0]) for k in range(BATCH_SIZE)]
    return batch_prediction


def __get_batch_predicted_frauds(batch_prediction, batch_tdf):
    batch_predicted_frauds = []
    for j in range(len(batch_prediction)):
        prediction_value = batch_prediction[j]
        if prediction_value > 0.5:
            #peterw_pred = [batch_tdf[k][j][0] for k in range(SEQ_LENGTH)]
            #print(f'DEBUG> index in batch {j:4d}: value = {prediction_value:5.3f}, pred = {peterw_pred}')
            entry_index = SEQ_LENGTH * j + (SEQ_LENGTH - 1)
            #fraud_entry = batch_tdf.iloc[entry_index]
            fraud_entry = batch_tdf.values[entry_index]
            #print(f'DEBUG> index in batch {j:4d}: pred-value = {prediction_value:5.3f}, fraud_entry =\n{fraud_entry}')
            #fraud_entry[14] = prediction_value
            batch_predicted_frauds.append(fraud_entry)
    return batch_predicted_frauds


def __run_batch(model, mapper, batch_tdf, use_onnxruntime):
    #print(f'DEBUG> batch_tdf.shape = {batch_tdf.shape} (= {BATCH_SIZE} x {SEQ_LENGTH} of 16 values)')
    mapped_batch_array = __map_batch(mapper, batch_tdf)
    #print(f'DEBUG> mapped_batch_array.shape = {mapped_batch_array.shape} (batch size of sequence are flipped and values are mapped)')
    batch_prediction = __predict_batch(model, mapped_batch_array, use_onnxruntime)
    batch_predicted_frauds = __get_batch_predicted_frauds(batch_prediction, batch_tdf)
    #for fraud_entry in batch_predicted_frauds:
    #    print(f'DEBUG> fraud_entry = {fraud_entry}')
    #print(f'DEBUG> NUMBER OF FRAUDS = {len(batch_predicted_frauds)}')
    return batch_predicted_frauds


def __print_onnx_inputs_and_outputs(onnx_session) -> None:
    print('ONNX file inputs and outputs')

    onnx_inputs = onnx_session.get_inputs()
    print(f'  number of inputs = {len(onnx_inputs)}')
    for index in range(0, len(onnx_inputs)):
        print(f'    input {index}: name = {onnx_inputs[index].name}, shape = {onnx_inputs[index].shape}, type = {onnx_inputs[index].type}')

    onnx_outputs = onnx_session.get_outputs()
    print(f'  number of outputs = {len(onnx_outputs)}')
    for index in range(0, len(onnx_outputs)):
        print(f'    output {index}: name = {onnx_outputs[index].name}, shape = {onnx_outputs[index].shape}, type = {onnx_outputs[index].type}')
