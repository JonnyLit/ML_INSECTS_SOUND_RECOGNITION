import flatbuffers
import sys

sys.path.append('tflite')  # folder 'tflite' containing the generated Python files

from tflite.Model import Model
from tflite.SubGraph import SubGraph
from tflite.Operator import Operator
from tflite.OperatorCode import OperatorCode
from tflite.Tensor import Tensor

# Path to your .tflite model
#model_path = '/home/zord/PycharmProjects/AI-Belha/Results/Custom_split/STFT/best_model-bee_presence_AI_Belha.h5'
#model_path = '/home/zord/PycharmProjects/AI-Belha/Results/Custom_split/STFT/best_model-bee_presence_AI_Belha.tflite'
#model_path = '/home/zord/PycharmProjects/AI-Belha/Results/Custom_split/STFT/best_model-bee_presence_AI_Belha_optimized.tflite'
#model_path = '/home/zord/PycharmProjects/AI-Belha/Results/Custom_split/STFT/best_model-bee_presence_AI_Belha_optimized_int_8_bit.tflite'
#model_path = '/home/zord/PycharmProjects/AI-Belha/Results/Custom_split/STFT/best_model-bee_presence_AI_Belha_optimized_float_16_bit.tflite'
model_path = '/home/zord/PycharmProjects/SBCM_4_classes/Results/16bit_data_results/mfcc/Config14_40Ksamples/epoch110_batch105_patience4/best_model-bee_presence_SBCM_optimized_float_16_bit.tflite'


# Load the TFLite model as bytes
with open(model_path, 'rb') as f:
    tflite_bytes = f.read()

# Parse the model
model = Model.GetRootAsModel(tflite_bytes, 0)

# Access the buffer data (assuming only one buffer, which contains custom options)
buffer = model.Buffers(0)
buffer_bytes = buffer.Data # get raw bytes

# Retrieve operator codes
operator_codes = [model.OperatorCodes(i) for i in range(model.OperatorCodesLength())]

# Iterate over subgraphs
for subgraph_idx in range(model.SubgraphsLength()):
    subgraph = model.Subgraphs(subgraph_idx)
    print(f"Subgraph {subgraph_idx}:")

    # List tensors
    for tensor_idx in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(tensor_idx)
        print(f"  Tensor {tensor_idx}: {tensor.Name().decode('utf-8')}")

    # List operators (layers)
    for op_idx in range(subgraph.OperatorsLength()):
        operator = subgraph.Operators(op_idx)
        opcode_index = operator.OpcodeIndex()

        # Get the operator code object from the model's OperatorCodes list
        opcode = operator_codes[opcode_index]

        # Get operator type
        builtin_code = opcode.BuiltinCode()
        print(f"\nOperator {op_idx}: {builtin_code}")

        # Extract custom options if present
        custom_options_offset = operator.CustomOptions(0)
        if custom_options_offset != 0:
            # The custom options are stored at this offset in the buffer
            custom_options_bytes = buffer_bytes[custom_options_offset:]
            print(f"Custom options (raw bytes): {custom_options_bytes}")
        else:
            print("No custom options.")
        # You can further parse 'custom_options_bytes' if you have a schema for those custom options