import torch
import torch.nn as nn
import numpy as np

from .quantizer import Quantizer
from .q_utils import *
from .range_linear import RangeLinearQuantWrapper as RangeLinearActQuantWrapper
from .ocs_impl import ocs_wts
from .clip import find_clip_aciq, find_clip_mmse, find_clip_entropy

# For activations profiling, we do a hacky implementation where setting
# a global var puts each instance of OCSParamLayerWrapper into profiling
# mode. In this mode we collect activation stats and don't perform
# quantization or OCS.
PROFILE_MODE = False

def ocs_set_profile_mode(pm):
    global PROFILE_MODE
    PROFILE_MODE = pm

class OCSParamLayerWrapper(RangeLinearActQuantWrapper):
    """
    OCS quantization wrappers for layers with weights (namely torch.nn.ConvNd and
    torch.nn.Linear)

    Args:
        wrapped_module (torch.nn.Module): Module to be wrapped
        num_bits_acts (int): Number of bits used for inputs and output quantization
        num_bits_params (int): Number of bits used for parameters (weights, no bias) quantization
        num_bits_accum (int): Number of bits allocated for the accumulator of intermediate integer results
    """
    def __init__(self, wrapped_module, num_bits_acts, num_bits_params, num_bits_accum=32,
                 weight_expand_ratio=0.0, weight_clip_threshold=1.0,
                 act_expand_ratio=0.0, act_clip_threshold=1.0):
        super(OCSParamLayerWrapper, self).__init__(wrapped_module,
                                                        num_bits_acts, num_bits_accum)

        if not isinstance(wrapped_module, (nn.Conv2d, nn.Linear)):
            raise ValueError(self.__class__.__name__ +
                             ' can wrap only Conv2D an_wtsd Linear modules')

        self.num_bits_params = num_bits_params
        self.num_bits_acts = num_bits_acts
        self.params_min_q_val, self.params_max_q_val = get_quantized_range(num_bits_params, signed=True)

        self.weight_expand_ratio = weight_expand_ratio
        self.weight_clip_threshold = weight_clip_threshold
        self.act_expand_ratio = act_expand_ratio
        self.act_clip_threshold = act_clip_threshold
        self.split_threshold = 0.0

        self.current_accum_scale = 1

        # Profiling
        self.profile_info = None
        self.weight_orig = None

        self.channels_to_split = None

        if isinstance(wrapped_module, nn.Conv2d):
            weight_torch = wrapped_module.weight.data
            self.weight_orig = weight_torch
            weight_np = weight_torch.numpy()

            """ Avoid quantizing the input layer """
            num_channels = weight_torch.shape[1]
            if num_channels == 3:
                return

            # Perform prelim OCS
            q_weight_np, in_channels_to_split = ocs_wts(
                    weight_np,
                    self.weight_expand_ratio,
                    split_threshold=self.split_threshold,
                    grid_aware=False)

            # Find the clip threshold (alpha value in clipping papers)
            if self.weight_clip_threshold > 0.0:
                # Fixed threshold
                max_abs = get_tensor_max_abs(q_weight_np)
                clip_max_abs = self.weight_clip_threshold * max_abs
            else:
                print('Auto-tuning for weight clip threshold...')
                # Calculate threshold
                values = weight_np.flatten().copy()

                # Branch on clip method
                if self.weight_clip_threshold == 0.0:
                    clip_max_abs = find_clip_mmse(values, self.num_bits_params)
                elif self.weight_clip_threshold == -1.0:
                    clip_max_abs = find_clip_aciq(values, self.num_bits_params)
                elif self.weight_clip_threshold == -2.0:
                    clip_max_abs = find_clip_entropy(values, self.num_bits_params)
                else:
                    raise ValueError('Undefined weight clip method')

            self.w_scale = symmetric_linear_quantization_scale_factor(self.num_bits_params, clip_max_abs)

            # Grid aware OCS
            q_weight_np, in_channels_to_split = ocs_wts(
                    weight_np,
                    self.weight_expand_ratio,
                    split_threshold=self.split_threshold,
                    w_scale=self.w_scale,
                    grid_aware=True)

            # Save which channels got split
            if len(in_channels_to_split) > 0:
                assert(q_weight_np.shape[1] == num_channels + len(in_channels_to_split))
                self.channels_to_split = np.array(in_channels_to_split)
                self.channels_to_split = torch.from_numpy(self.channels_to_split).cuda()

            q_weight_torch = torch.from_numpy(q_weight_np)
            linear_quantize_clamp(q_weight_torch,
                                  self.w_scale,
                                  self.params_min_q_val,
                                  self.params_max_q_val,
                                  inplace=True)
            wrapped_module.weight.data = q_weight_torch

    def forward(self, *inputs):
        if PROFILE_MODE == True:
            """ Activation profiling """
            # Flag indicating whether profiling was done
            self.profile_info = True
            # Number of inputs should be 1
            assert(len(inputs) == 1)
            input = inputs[0]
            # Input shape is (Batch, C, H, W)
            self.input_shape = input.shape
            num_channels = self.input_shape[1]
            #print(input.shape)

            if num_channels > 3:

                torch_input_tensor = input.data.cpu()
                input_np = torch_input_tensor.numpy()

                """ Clipping """
                if self.act_clip_threshold > 0.0:
                    act_clip_max_abs = np.max(np.abs(input_np)) * self.act_clip_threshold
                else:
                    print('Auto-tuning for activation clip threshold...')
                    if self.act_clip_threshold == 0.0:
                        act_clip_max_abs = find_clip_mmse(input_np.flatten(), self.num_bits_acts)
                    elif self.act_clip_threshold == -1.0:
                        act_clip_max_abs = find_clip_aciq(input_np.flatten(), self.num_bits_acts)
                    elif self.act_clip_threshold == -2.0:
                        act_clip_max_abs = find_clip_entropy(input_np.flatten(), self.num_bits_acts)
                    else:
                        raise ValueError('Undefined act clip method')

                self.act_clip_max_abs = torch.tensor(act_clip_max_abs).cuda()

                """ Get channels to split """
                # Unused

            # For profiling, we use the original FP weights
            weight_q = self.wrapped_module.weight.data
            self.wrapped_module.weight.data = self.weight_orig.cuda()
            # Run the forward pass
            accum = self.wrapped_module.forward(*inputs)
            self.wrapped_module.weight.data = weight_q
            return accum
        else:
            assert(self.profile_info)
            assert(len(inputs) == 1)

            """ Avoid quantizing the input layer """
            input = inputs[0]
            num_channels = input.shape[1]
            if num_channels == 3:
                return self.wrapped_module.forward(*inputs)

            """ If we don't need OCS on the activations, skip the
                cuda->cpu->cuda transfer to save time """
            #if self.act_expand_ratio == 0.0:
            #    return super(OCSParamLayerWrapper, self).forward(input)

            """ Quantize inputs """
            inputs_q = []
            for idx, input in enumerate(inputs):
                # Clip
                new_max_abs = self.act_clip_max_abs

                # Determine scale factor for quantization
                in_scale = symmetric_linear_quantization_scale_factor(
                            self.num_bits_acts, new_max_abs)
                self.current_accum_scale = in_scale * self.w_scale

                input_q = linear_quantize_clamp(input.data, in_scale,
                                      self.acts_min_q_val, self.acts_max_q_val,
                                      inplace=False)

                # Duplicate channels
                if self.channels_to_split is not None:
                    N, _, H, W = input.shape

                    input_splits = torch.index_select(input_q, dim=1, index=self.channels_to_split)

                    input_ocs = torch.cat([input_q, input_splits], dim=1)
                else:
                    input_ocs = input_q

                inputs_q.append(torch.autograd.Variable(input_ocs))

            """ forward through wrapped module """
            accum = self.wrapped_module.forward(*inputs_q)
            clamp(accum.data, self.accum_min_q_val, self.accum_max_q_val, inplace=True)
            """ re-quantize accumulator to quantized output range """
            requant_scale, out_scale = self.post_quantized_forward(accum)
            out_q = linear_quantize_clamp(accum.data, requant_scale, self.acts_min_q_val, self.acts_max_q_val, inplace=True)
            """ de-quantize back to FP32 """
            out_f = linear_dequantize(out_q, out_scale, inplace=True)
            return torch.autograd.Variable(out_f)

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
        tmpstr += '  num_bits_activations={0}, num_bits_parameters={1}'.format(
            self.num_bits_acts, self.num_bits_params) + '\n'
        tmpstr += ')'
        return tmpstr


class OCSQuantizer(Quantizer):
    def __init__(self, model, bits_activations=8, bits_parameters=8,
                 weight_expand_ratio=0.0, weight_clip_threshold=1.0,
                 act_expand_ratio=0.0, act_clip_threshold=1.0):
        super(OCSQuantizer, self).__init__(model, bits_activations=bits_activations,
                                           bits_weights=bits_parameters,
                                           train_with_fp_copy=False)
        self.model.quantizer_metadata = {'type': type(self),
                                         'params': {'bits_activations': bits_activations,
                                                    'bits_parameters': bits_parameters}}

        def replace_fn(module, name, qbits_map):
            return OCSParamLayerWrapper(module, qbits_map[name].acts, qbits_map[name].wts,
                                        weight_expand_ratio=weight_expand_ratio, weight_clip_threshold=weight_clip_threshold,
                                        act_expand_ratio=act_expand_ratio, act_clip_threshold=act_clip_threshold)

        self.replacement_factory[nn.Conv2d] = replace_fn
        # self.replacement_factory[nn.Linear] = replace_fn
