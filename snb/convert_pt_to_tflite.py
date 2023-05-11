import torch
import tensorflow_probability
import onnx_tf
import onnx

def tf_to_tf_lite(tf_path, tf_lite_path):
    """
    From pytorch_to_tflite, but it won't install.

    Converts TF saved model into TFLite model
    :param tf_path: TF saved model path to load
    :param tf_lite_path: TFLite model path to save
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)  # Path to the SavedModel directory
    tflite_model = converter.convert()  # Creates converter instance
    with open(tf_lite_path, 'wb') as f:
        f.write(tflite_model)


def onnx_to_tf(onnx_path, tf_path):
    """
    From pytorch_to_tflite, but it won't install.

    Converts ONNX model to TF 2.X saved file
    :param onnx_path: ONNX model path to load
    :param tf_path: TF path to save
    """
    onnx_model = onnx.load(onnx_path)

    # onnx.checker.check_model(onnx_model)  # Checks signature
    tf_rep = onnx_tf.backend.prepare(onnx_model)  # Prepare TF representation
    tf_rep.export_graph(tf_path)  # Export the model

def pt_to_tflite(pt_path, input_shape, out_path=None):
    if out_path is None:
        out_path = pt_path.replace(".pt", ".tflite")
    model = torch.load(pt_path)
    onnx_path = pt_path.replace(".pt", ".onnx")
    torch.onnx.export(model, torch.randn(input_shape), onnx_path)
    tf_path = onnx_path.replace(".onnx", ".tf")
    onnx_to_tf(onnx_path=onnx_path, tf_path=tf_path)
    tf_to_tf_lite(tf_path, out_path)
    return [onnx_path, tf_path, out_path]
