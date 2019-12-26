# Code taken from https://gist.github.com/sergeyprokudin/429c61e6536f5af5d9b0e36c660b3ae9

import numpy as np


def count_conv_params_flops(conv_layer, verbose=1):
    # out shape is  n_cells_dim1 * (n_cells_dim2 * n_cells_dim3)
    out_shape = conv_layer.output.shape.as_list()
    n_cells_total = np.prod(out_shape[1:-1])

    n_conv_params_total = conv_layer.count_params()

    conv_flops = 2 * n_conv_params_total * n_cells_total

    if verbose:
        print("layer %s params: %s" % (conv_layer.name, "{:,}".format(n_conv_params_total)))
        print("layer %s flops: %s" % (conv_layer.name, "{:,}".format(conv_flops)))

    return n_conv_params_total, conv_flops


def count_dense_params_flops(dense_layer, verbose=1):
    # out shape is  n_cells_dim1 * (n_cells_dim2 * n_cells_dim3)
    out_shape = dense_layer.output.shape.as_list()
    n_cells_total = np.prod(out_shape[1:-1])

    n_dense_params_total = dense_layer.count_params()

    dense_flops = 2 * n_dense_params_total

    if verbose:
        print("layer %s params: %s" % (dense_layer.name, "{:,}".format(n_dense_params_total)))
        print("layer %s flops: %s" % (dense_layer.name, "{:,}".format(dense_flops)))

    return n_dense_params_total, dense_flops


def count_model_params_flops(model):
    total_params = 0
    total_flops = 0

    model_layers = model.layers

    for layer in model_layers:

        if any(conv_type in str(type(layer)) for conv_type in ['Conv1D', 'Conv2D', 'Conv3D']):
            params, flops = count_conv_params_flops(layer)
            total_params += params
            total_flops += flops
        elif 'Dense' in str(type(layer)):
            params, flops = count_dense_params_flops(layer)
            total_params += params
            total_flops += flops
        else:
            print("warning:: skippring layer: %s" % str(layer))

    print("total params (%s) : %s" % (model.name, "{:,}".format(total_params)))
    print("total flops  (%s) : %s" % (model.name, "{:,}".format(total_flops)))

    return total_params, total_flops
