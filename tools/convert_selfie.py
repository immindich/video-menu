"""Convert MediaPipe's selfie segmenter (TFLite) to OpenVINO IR for the daemon.

OpenVINO's TFLite frontend can't read the model as-is: its final upsampling
layer is MediaPipe's custom op Convolution2DTransposeBias (a transposed
convolution with a bias, fused). This registers a translation for it, then
adds preprocessing so the model takes the daemon's frame layout directly:
three planes (NCHW) of f16 in 0..1, instead of interleaved (NHWC) f32.

usage: python convert_selfie.py IN.tflite OUT.xml
  (writes OUT.xml and OUT.bin)
"""
import sys

import openvino as ov
import openvino.opset13 as ops
from openvino.frontend import ConversionExtension
from openvino.preprocess import PrePostProcessor


def convolution_2d_transpose_bias(node):
    """MediaPipe's transposed convolution + bias, on NHWC tensors.

    Inputs: data [N, H, W, C_in], weights [C_out, kH, kW, C_in], bias
    [C_out]. The op's stride and padding live in custom options the frontend
    can't read; MediaPipe uses it for 2x upsampling with a 2x2 kernel and
    stride 2, where the windows don't overlap and padding has no effect.
    Refuse anything else rather than guess.
    """
    data, weights, bias = (node.get_input(i) for i in range(3))
    kernel = weights.get_partial_shape()
    if not kernel.is_static or [kernel[1].get_length(), kernel[2].get_length()] != [2, 2]:
        raise ValueError(f"Convolution2DTransposeBias: expected a 2x2 kernel, got {kernel}")
    nchw = ops.transpose(data, ops.constant([0, 3, 1, 2]))
    # ConvolutionBackpropData wants filters as [C_in, C_out, kH, kW].
    filters = ops.transpose(weights, ops.constant([3, 0, 1, 2]))
    up = ops.convolution_backprop_data(
        nchw, filters, strides=[2, 2], pads_begin=[0, 0], pads_end=[0, 0], dilations=[1, 1]
    )
    up = ops.add(up, ops.reshape(bias, ops.constant([1, -1, 1, 1]), special_zero=False))
    return [ops.transpose(up, ops.constant([0, 2, 3, 1])).output(0)]


src, dst = sys.argv[1], sys.argv[2]
core = ov.Core()
core.add_extension(ConversionExtension("Convolution2DTransposeBias", convolution_2d_transpose_bias))
model = core.read_model(src)

ppp = PrePostProcessor(model)
ppp.input().tensor().set_element_type(ov.Type.f16).set_layout(ov.Layout("NCHW"))
ppp.input().model().set_layout(ov.Layout("NHWC"))
model = ppp.build()

ov.save_model(model, dst, compress_to_fp16=True)
ins = [(i.get_any_name(), str(i.get_partial_shape()), str(i.get_element_type())) for i in model.inputs]
outs = [(o.get_any_name(), str(o.get_partial_shape()), str(o.get_element_type())) for o in model.outputs]
print(f"wrote {dst}: inputs {ins}, outputs {outs}")
