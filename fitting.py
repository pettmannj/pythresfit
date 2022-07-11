from __future__ import annotations

import numpy as np
import pandas as pd
import lmfit

import os.path
from typing import Dict, Type
from itertools import accumulate, cycle

from settings import Settings
from dataset import DataSet
from exporthandler import ExportHandler
from parameteradaptor import ParameterAdaptor
from plotting import Plot
from models import Sigmoidal5PModel
from os import listdir

def get_minimisable_func(dataset: DataSet):
    # Get data as index + values to allow a separate index for each curve.
    data = dataset.listofdata

    # Flatten data
    x_data = np.hstack(([d.index.values for d in data]))
    y_data = np.hstack([d.values for d in data])

    # Infinite cycle of the positions where the data for each curve is stored in the 1D array.
    steps = [0] + list(accumulate([len(d) for d in data]))
    slices_gen = cycle([slice(f, s) for f, s in zip(steps[:-1], steps[1:])])

    res = list()
    models = dataset.models

    def fn2min(params: ParameterAdaptor):
        for m in models:
            sl = next(slices_gen)
            res[sl] = m.func(x_data[sl], params) - y_data[sl]

        return res
    
    return fn2min

#Read folder
def read_folder(settings) -> Dict[str, str]:
    files = {}

    if(os.path.isfile(settings.data_path)):
        if settings.data_path.endswith(settings.data_extension):
            f = os.path.split(settings.data_path)[-1]
            name = f[:-len(settings.data_extension)]
            files[name] = settings.data_path

    elif(os.path.isdir(settings.data_path)):
        for f in listdir(settings.data_path):
            if f.endswith(settings.data_extension):
                name = f[:-len(settings.data_extension)]
                files[name] = os.path.join(settings.data_path, f)

    else: raise Exception("Not a valid file or dir!")

    if not(len(files) > 0): raise Exception("No file with extension *" + settings.data_extension.upper() + " found!")
        
    return files

def parse_data(name: str, path: str, settings: Settings, model_cls: Type) -> DataSet:
    excl_value = '*'
    raw_data = pd.read_csv(path, index_col=0, dtype=str) # type: ignore

    index_mask = raw_data.index.map(lambda x: excl_value in str(x)) # Check if index contains exclusion values. This is not saved, but only used to calculate the excluded mask.
    excluded_mask = raw_data.applymap(lambda x: excl_value in str(x)).values # Check if data contains exlcusion values.

    # If an index value is excluded, mark all values in that row as excluded.
    idx = [i for i in range(len(index_mask)) if index_mask[i]]
    excluded_mask[idx] = True

    # Remove exlusion marks from values and index and parse as numbers.
    data = raw_data.applymap(lambda x: str(x).replace(excl_value, ''))
    data.index = data.index.map(lambda x: str(x).replace(excl_value, ''))
    data = data.astype('float64')
    data.index = data.index.astype('float64')

    # Create dataset. This is going to hold all important information/results.
    # Data is automatically sorted in the background.
    return DataSet(name, data, settings, model_cls, excluded_mask)

#Process data
def main(settings):
    
    files = read_folder(settings)
    handler = ExportHandler() # Handles combined saving of all data.
    abs_thres = None
    rel_thres = None

    # Print summary
    print(f'Unique ID: {handler.iD}\n')
    print(f'Datasets found in folder {settings.data_path}:')
    for f in files: print(f'\t{f}')
    print('\nStart fitting...\n')
        
    
    for f in files:
        dataset = parse_data(f, files[f], settings, Sigmoidal5PModel) # Read data and create dataset
        # return dataset
        
        # Fitting
        func2min = get_minimisable_func(dataset) # Initialise fitting func (closure).
        minimizer = lmfit.Minimizer(func2min, dataset.initial_params, nan_policy='propagate')
        r = minimizer.minimize(method=settings.fitting_method)
        if settings.calc_thres.lower().startswith('abs'): abs_thres = settings.potency_thres
        dataset.add_fitting_results(r.params, abs_thres)
        handler.add_dataset(dataset)

        # Plotting
        plot = Plot(dataset)
        plot.plot_fits(f)
        handler.add_plot(plot)

        # Normalisation
        if settings.normalise_data:
            if settings.calc_thres.lower().startswith('rel'): rel_thres = settings.potency_thres
            dataset = dataset.normalise(rel_thres)
            handler.add_dataset(dataset)

            # Plotting
            plot = Plot(dataset)
            plot.plot_fits(f)
            handler.add_plot(plot)

    handler.save() # Save all data
    return handler