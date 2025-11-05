import tensorflow as tf
import numpy as np

def convert_and_optimize_model(input_path, output_path, is_h5=True, representative_data_gen=None, quantization_type=None):
    """
    Converts a Keras .h5 model or an existing .tflite model into an optimized TFLite model
    with specified quantization.

    Args:
        input_path (str): Path to the input model (.h5 or .tflite).
        output_path (str): Path where to save the optimized .tflite model.
        is_h5 (bool): True if input_path is a .h5 Keras model, False if it's a .tflite.
        representative_data_gen (callable, optional): Dataset generator for full integer quantization.
        quantization_type (str, optional): 'int8' for full int quantization,
                                           'float16' for float16 quantization,
                                           or None for no quantization.
    """
    if is_h5:
        model = tf.keras.models.load_model(input_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    else:
        converter = tf.lite.TFLiteConverter.from_saved_model(input_path)

    # Set optimization for quantization if specified
    if quantization_type == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_data_gen is None:
            raise ValueError("Representative dataset required for int8 quantization.")
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    elif quantization_type == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    else:
        # No quantization, just convert
        pass

    # Convert the model
    tflite_model = converter.convert()

    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model saved to {output_path} with quantization: {quantization_type}")


# Example usage:

# For full integer quantization, define a representative dataset generator:
def representative_dataset():
    for _ in range(100):
        # Replace with your sample input data shape
        # For example, if your model expects input shape (1, 224, 224, 3):
        #dummy_input = np.random.rand(1, 32, 44, 1).astype(np.float32)
        dummy_input = np.random.rand(1, 32, 44, 1).astype(np.float16)
        yield [dummy_input]

"""
# Convert from a .h5 to optimized .tflite with quantization
convert_and_optimize_model(
    input_path='/home/zord/PycharmProjects/AI-Belha/Results/Custom_split/STFT/best_model-bee_presence_AI_Belha.h5',
    output_path='/home/zord/PycharmProjects/AI-Belha/Results/Custom_split/STFT/best_model-bee_presence_AI_Belha_optimized.tflite',
    is_h5=True,
    representative_data_gen=representative_dataset
)
"""


"""
# For 8-bit integer quantization
convert_and_optimize_model(
    input_path='/home/zord/PycharmProjects/AI-Belha/Results/Custom_split/STFT/best_model-bee_presence_AI_Belha.h5',
    output_path='/home/zord/PycharmProjects/AI-Belha/Results/Custom_split/STFT/best_model-bee_presence_AI_Belha_optimized_int_8_bit.tflite',
    is_h5=True,
    representative_data_gen=representative_dataset,
    quantization_type='int8'
)
"""
# For float16 quantization
convert_and_optimize_model(
    input_path='/home/zord/PycharmProjects/SBCM_4_classes/best_model-bee_presence_SBCM.h5',
    output_path='/home/zord/PycharmProjects/SBCM_4_classes/best_model-bee_presence_SBCM_optimized_float_16_bit.tflite',
    is_h5=True,
    quantization_type='float16'
)
"""
# For no quantization
convert_and_optimize_model(
    input_path='/home/zord/PycharmProjects/AI-Belha/Results/Custom_split/STFT/best_model-bee_presence_AI_Belha.h5',
    output_path='/home/zord/PycharmProjects/AI-Belha/Results/Custom_split/STFT/best_model-bee_presence_AI_Belha_optimized_no_quantized.tflite',
    is_h5=True,
    quantization_type=None
)
"""