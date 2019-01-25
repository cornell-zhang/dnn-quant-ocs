#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch.nn as nn

from .quantizer import Quantizer
from .q_utils import *

###
# Range-based linear quantization
###


class RangeLinearQuantWrapper(nn.Module):
    """
    Base class for module which wraps an existing module with linear range-base quantization functionality

    Args:
        wrapped_module (torch.nn.Module): Module to be wrapped
        num_bits_acts (int): Number of bits used for inputs and output quantization
        num_bits_accum (int): Number of bits allocated for the accumulator of intermediate integer results
    """
    def __init__(self, wrapped_module, num_bits_acts, num_bits_accum=32):
        super(RangeLinearQuantWrapper, self).__init__()

        self.wrapped_module = wrapped_module
        self.num_bits_acts = num_bits_acts

        self.acts_min_q_val, self.acts_max_q_val = get_quantized_range(num_bits_acts, signed=True)
        self.accum_min_q_val, self.accum_max_q_val = get_quantized_range(num_bits_accum, signed=True)

    def forward(self, *inputs):
        in_scales = self.pre_quantized_forward(*inputs)

        # Quantize inputs
        inputs_q = []
        for idx, input in enumerate(inputs):
            input_q = linear_quantize_clamp(input.data, in_scales[idx], self.acts_min_q_val, self.acts_max_q_val,
                                            inplace=False)
            inputs_q.append(torch.autograd.Variable(input_q))

        # Forward through wrapped module
        accum = self.wrapped_module.forward(*inputs_q)
        clamp(accum.data, self.accum_min_q_val, self.accum_max_q_val, inplace=True)

        # Re-quantize accumulator to quantized output range
        requant_scale, out_scale = self.post_quantized_forward(accum)
        out_q = linear_quantize_clamp(accum.data, requant_scale, self.acts_min_q_val, self.acts_max_q_val, inplace=True)

        # De-quantize back to FP32
        out_f = linear_dequantize(out_q, out_scale, inplace=True)

        return torch.autograd.Variable(out_f)

    def pre_quantized_forward(self, *inputs):
        """
        Calculate input scale factors and perform any action required before quantization of inputs.

        Should be overridden by all subclasses

        :param inputs: Current input tensors passed to forward method
        :return: List of scale factors per input
        """
        raise NotImplementedError

    def post_quantized_forward(self, accumulator):
        """
        Calculate re-quantization scale factor (for converting the intermediate integer accumulator to output range),
        and output scale factor.

        :param accumulator: Tensor with accumulator values
        :return: Tuple of (re-quantization scale factor, output scale factor)
        """
        raise NotImplementedError


class RangeLinearQuantParamLayerWrapper(RangeLinearQuantWrapper):
    """
    Linear range-based quantization wrappers for layers with weights (namely torch.nn.ConvNd and
    torch.nn.Linear)

    Args:
        wrapped_module (torch.nn.Module): Module to be wrapped
        num_bits_acts (int): Number of bits used for inputs and output quantization
        num_bits_params (int): Number of bits used for parameters (weights, no bias) quantization
        num_bits_accum (int): Number of bits allocated for the accumulator of intermediate integer results
    """
    def __init__(self, wrapped_module, num_bits_acts, num_bits_params, num_bits_accum=32):
        super(RangeLinearQuantParamLayerWrapper, self).__init__(wrapped_module, num_bits_acts, num_bits_accum)

        if not isinstance(wrapped_module, (nn.Conv2d, nn.Linear)):
            raise ValueError(self.__class__.__name__ + ' can wrap only Conv2D and Linear modules')

        self.num_bits_params = num_bits_params
        self.params_min_q_val, self.params_max_q_val = get_quantized_range(num_bits_params, signed=True)

        # Quantize weights - overwrite FP32 weights
        self.w_scale = symmetric_linear_quantization_scale_factor(num_bits_params,
                                                                  get_tensor_max_abs(wrapped_module.weight))
        linear_quantize_clamp(wrapped_module.weight.data, self.w_scale, self.params_min_q_val, self.params_max_q_val,
                              inplace=True)

        self.current_accum_scale = 1

    def pre_quantized_forward(self, input):
        super(RangeLinearQuantParamLayerWrapper, self).forward(input)

    def pre_quantized_forward(self, input):
        in_scale = symmetric_linear_quantization_scale_factor(self.num_bits_acts, get_tensor_max_abs(input))
        self.current_accum_scale = in_scale * self.w_scale
        return [in_scale]

    def post_quantized_forward(self, accumulator):
        accum_max_abs = get_tensor_max_abs(accumulator)
        y_f_max_abs = accum_max_abs / self.current_accum_scale
        out_scale = symmetric_linear_quantization_scale_factor(self.num_bits_acts, y_f_max_abs)
        requant_scale = out_scale / self.current_accum_scale
        return requant_scale, out_scale

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '(\n'
        tmpstr += '  (wrapped_module): ' + self.wrapped_module.__repr__() + '\n'
        tmpstr += '  num_bits_activations={0}, num_bits_parameters={1}'.format(self.num_bits_acts,
                                                                               self.num_bits_params) + '\n'
        tmpstr += ')'
        return tmpstr


class SymmetricLinearQuantizer(Quantizer):
    """
    Applies symmetric, range-based linear quantization to a model.
    Currently, the following Modules are supported: torch.nn.Conv2d, torch.nn.Linear

    Args:
        model (torch.nn.Module): Model to be quantized
        bits_activations/parameters: Number of bits to be used when quantizing each tensor type
    """
    def __init__(self, model, bits_activations=8, bits_parameters=8):
        super(SymmetricLinearQuantizer, self).__init__(model, bits_activations=bits_activations,
                                                       bits_weights=bits_parameters,
                                                       train_with_fp_copy=False)

        self.model.quantizer_metadata = {'type': type(self),
                                         'params': {'bits_activations': bits_activations,
                                                    'bits_parameters': bits_parameters}}

        def replace_fn(module, name, qbits_map):
            return RangeLinearQuantParamLayerWrapper(module, qbits_map[name].acts, qbits_map[name].wts)

        self.replacement_factory[nn.Conv2d] = replace_fn
        self.replacement_factory[nn.Linear] = replace_fn
