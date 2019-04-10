import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
from cycler import cycler
import matplotlib
import matplotlib.pyplot as plt

lines = ['-', ':', '--', '-.']
linecycler = cycler(linestyle=lines)
matplotlib.rcParams['axes.prop_cycle'] = linecycler


def flatten_params_condense(params):
    key_maps = {
        'alpha': r'$\alpha$',
        'reg': r'$\lambda$',
        'beta1': r'$\beta1$',
        'beta2': r'$\beta2$',
        'epsilon': r'$\epsilon$',
    }

    def trim_leading_zero(v):
        if re.match(r'0.\d+', v):
            return v[1:]
        else:
            return v

    return ','.join(sorted('%s=%s' % (key_maps.get(k, k), trim_leading_zero(v)) for k, v in params.items()))

def flatten_params(params):
    return ','.join(sorted('%s=%s' % (k, v) for k, v in params.items()))


def plot(all_data, common_params, phases, types, path):
    for data_type in types:
        for data in all_data:
            for phase in phases:
                idx = '%s_%s' % (phase, data_type)
                plt.plot(
                    data['loss_data'].epoch,
                    data['loss_data'][idx],
                    label=','.join([phase, flatten_params_condense(data['params'])]),
                    linewidth=1,
                )

        plt.ylabel(data_type)
        plt.xlabel('epochs')
        plt.title(','.join([data_type, flatten_params_condense(common_params)]))
        plt.legend()
        if len(phases) == 1:
            filename = '%s~%s.png' % (data_type, flatten_params({'phase': phases[0], **common_params}))
        else:
            filename = '%s~%s.png' % (data_type, flatten_params(common_params))

        plt.savefig(os.path.join(path, filename))
        plt.close()


def parse_filename(filename):
    name = os.path.basename(filename)
    m = re.match(r'loss~(.*)~(.*)\.csv', name)
    if not m:
        sys.stderr.print('Error: file %s is not a correct filename.' % name)
        sys.exit()

    model_type = m.group(1)
    param_str = m.group(2)

    params = {}
    for p in param_str.split('_'):
        k, v = p.split('=')
        params[k] = v

    params['model'] = model_type

    return params


def extract_common_params(all_data):
    # Get all keys from the params
    keys = {k for data in all_data for k in data['params'].keys()}

    # Find common params
    common_params = {}
    for k in keys:
        values = {data['params'][k] for data in all_data}
        if len(values) == 1:
            common_params[k] = list(values)[0]

    # Remove common params from original params
    for data in all_data:
        keys = list(data['params'].keys())
        for k in keys:
            if k in common_params:
                del data['params'][k]

    return common_params


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True,
                        nargs=1, help='Output path')
    parser.add_argument('--filename', type=str,
                        nargs='+', help='Data filenames.')
    parser.add_argument('--phase', type=str, choices=[
                        'train', 'valid', 'test'], nargs='+', required=True, help='Phases to be plotted')
    parser.add_argument('--type', type=str, choices=[
                        'loss', 'accuracy'], nargs='+', required=True, help='Data types to be plotted')
    args = parser.parse_args()

    # Parse filenames and fetch data

    all_data = []
    for filename in args.filename:
        params = parse_filename(filename)
        loss_data = pd.read_csv(filename)

        all_data.append({
            'params': params,
            'loss_data': loss_data,
        })

    common_params = extract_common_params(all_data)

    plot(all_data, common_params, args.phase, args.type, args.path[0])
