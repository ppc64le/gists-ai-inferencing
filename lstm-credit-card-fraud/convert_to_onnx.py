import os
import tensorflow as tf
import tf2onnx


BATCH_SIZE = 160
NUM_MAPPED_COLUMNS = 220


MODEL_NAME = 'model_160batch'


if __name__ == '__main__':
    print('=================================================================')
    dir_udf: str = os.path.normpath(os.path.join(os.path.realpath(__file__), '..', 'udf'))

    model_file = os.path.join(dir_udf, f'{MODEL_NAME}.h5')
    model = tf.keras.models.load_model(model_file)

    onnx_file = os.path.join(dir_udf, f'{MODEL_NAME}.onnx')
    spec = (tf.TensorSpec((None, BATCH_SIZE, NUM_MAPPED_COLUMNS), name='input'),)
    tf2onnx.convert.from_keras(model, input_signature=spec, output_path=onnx_file)
