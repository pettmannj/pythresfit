from __future__ import annotations

import numpy as np
import pandas as pd

from typing import List, Sequence, Type, Optional, cast
from collections import defaultdict
from copy import deepcopy

from settings import Settings
from parameteradaptor import ParameterAdaptor
from models import FittingModelBase, calc_prefix, norm


class DataSet:
    name: str

    # Data
    _data: pd.DataFrame
    _exclusion_mask: np.ndarray # Boolean mask of shape data that holds which datapoints are excluded.
    included_data: pd.DataFrame
    excluded_data: pd.DataFrame
    norm_col: Optional[str] = None # Column to be used for normalisation

    # Settings
    settings: Settings

    # Fitting
    models: List[FittingModelBase]
    prefixes: List[str]
    initial_params: ParameterAdaptor
    best_fit_params: ParameterAdaptor # Fitting results

    # Numeric results
    # Concentration needed to reach threshold activation. Can be empty.
    log_potency: List[float]
    # Empirical max and min (based on fitted curve without extrapolation). Can be empty.
    emp_emax: List[float]
    emp_emin: List[float]

    @property
    def listofdata(self) -> List[pd.Series]:
        return [self.included_data[c].copy().dropna() for c in self.included_data]

    @property
    def data(self): return self._data

    # Generates excluded and includes DataFrames from data and exlusion mask. Needs to be called if either data or exclusion mask is changed.
    def _update_data(self, v: pd.DataFrame) -> None:
        # Sort data and mask (this will automatically propagate to include and excluded data).
        sorted_idx = v.index.argsort()[::-1] # Reverse to sort descending
        self._data = v.iloc[sorted_idx] # type: ignore
        self._exclusion_mask = self._exclusion_mask[sorted_idx] # Needs to be after assigning data!

        # Make included and excluded data
        self.included_data = self._data.copy() # type: ignore
        self.included_data[self._exclusion_mask] = np.NaN

        self.excluded_data = self._data.copy() # type: ignore
        self.excluded_data[~self._exclusion_mask] = np.NaN

    @data.setter
    def data(self, v: pd.DataFrame): self._update_data(v)

    def _get_norm_col(self) -> str:
        # If user provided a numeric normalisation column
        if isinstance(self.settings.norm_column, int):
            ni = self.settings.norm_column

            if ni >= self.data.shape[1]:
                raise ValueError(f'Normalisation column with value {ni} is larger than the number of curves in dataset {self.name}.')

            if ni < -self.data.shape[1]:
                raise ValueError(f'Negative value in normalisation column cannot be smaller than the negative number of curves in dataset {self.name}.')

            return self.data.columns[ni] # type: ignore

        elif isinstance(self.settings.norm_column, str):
            # If the input in settings is already a string, try to find if that string exists in the data columns.
            nc = self.settings.norm_column
            if nc not in self.data.columns: raise ValueError(f'Normalisation column {nc} was not found in dataset {self.name}')
            return nc

        else: raise ValueError(f'Type {type(self.settings.norm_column)} is not supported for the norm_column setting.')

    # Exclude bell-shaped data. Threshold defines the sensitivity.
    def _exclude_bell(self, threshold: float = 0.2):
        sd0 = self.included_data.iloc[-1].std() / self.included_data.max()
        sd0 = sd0.values
        sd0[np.isnan(sd0)] = 0 # type: ignore In case 0 is a NaN. This can happen if there is less than two 0 datapoints (e.g. there is only 1 curve).
        
        diff = (self.included_data - self.included_data.max()) / self.included_data.max()
        diff = diff.values

        # Check if difference is big enough and the change is more than 2x the SD from the 0 values.
        mask = np.logical_and(np.abs(diff) > threshold, np.abs(diff) > (sd0 * 5)) # type: ignore

        # Get otherwise excluded values and check for continous series of excluded values.
        mask = np.logical_or(mask, self._exclusion_mask) # type: ignore

        for i in range(1, mask.shape[0]): # type: ignore
            mask[i] = np.logical_and(mask[i-1], mask[i]) # type: ignore

        # Save data
        self._exclusion_mask = np.logical_or(mask, self._exclusion_mask) # type: ignore
        self._update_data(self.data)

    def __init__(self, name: str, data: pd.DataFrame, settings: Settings, model_cls: Type, exclusion_mask: Optional[np.ndarray] = None) -> None:
        self.name = name
        self._exclusion_mask = exclusion_mask if exclusion_mask is not None else np.zeros_like(self.data, dtype=bool) # type: ignore
        self.data = data # Needs to be after exclusion mask assignment!
        self.settings = settings
        self.prefixes = calc_prefix(list(self.data.columns)) #type: ignore
        self.models = [model_cls(prefix=pre) for pre in self.prefixes]

        self.initial_params = ParameterAdaptor()
        self.initial_params.initialise(self.models, settings)
        self.initial_params.guess(self.models, self.included_data)

        self.log_potency = list()
        self.emp_emax = list()
        self.emp_emin = list()

        self.norm_col = self._get_norm_col()
        if self.settings.remove_bellshaped_data: self._exclude_bell()

    def add_fitting_results(self, fit_params: ParameterAdaptor, threshold: Optional[float] = None) -> None:
        self.best_fit_params = fit_params

        # Calc empirical stats & threshold
        self.calc_empirical_stats()
        if threshold: self.calc_potency(threshold)

    def normalise(self, threshold: Optional[float] = None) -> DataSet:
        ds = deepcopy(self)
        ds.name = f'{ds.name} (NORM)'

        ni = ds.included_data.columns.get_loc(ds.norm_col)

        range = ds.emp_emax[ni] - ds.emp_emin[ni]
        hill = np.median(np.array(ds.best_fit_params.get_param_values('hill')))
        add = 1 if hill < 0 else 0

        if hill > 0: miny = np.array(ds.best_fit_params.get_param_values('bottom'))
        else: miny = np.array(ds.best_fit_params.get_param_values('top'))
        

        ds.best_fit_params = ds.best_fit_params.normalise(miny, range, add) # type: ignore
        ds.data = norm(ds.data, miny, range) + add # type: ignore

        # Calc empirical stats & threshold again
        ds.calc_empirical_stats()
        if threshold: ds.calc_potency(threshold)

        return ds

    def export_results(self, curve_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
        result = defaultdict(list)
        # Use all curves if no selection is passed.
        curve_names = curve_names if curve_names else self.data.columns # type: ignore
        curve_names = cast(Sequence[str], curve_names) # No runtime function.
        prefixes = calc_prefix(curve_names)

        for pre, c in zip(prefixes, curve_names):
            result['curve'].append(c)
            
            # Add parameters
            for p in self.best_fit_params._p_names:
                result[p].append(self.best_fit_params[pre + p].value)

        if self.log_potency: result['log_potency'] = self.log_potency
        if self.emp_emin: result['empirical min'] = self.emp_emin
        if self.emp_emax: result['empirical max'] = self.emp_emax

        n_total = [self.data[c].notna().sum() for c in self.data]
        n_excl = self._exclusion_mask.sum(axis=0)
        
        result['Total N'] = n_total
        result['Excluded N'] = n_excl

        return pd.DataFrame(result)

    def calc_potency(self, threshold: float) -> None:
        potency = [m.solve(threshold, self.best_fit_params) for m in self.models] # type: ignore
        data = self.listofdata

        def get_interpolated(data: pd.Series, value: float) -> str:
            log_value = np.log10(value) # type: ignore
            xmin = np.log10(data.index[data.index > 0].min()) # type: ignore
            xmax = np.log10(data.index.max()) # type: ignore

            # Allow for extrapolation if extrapolate > 0.
            xrange = xmax  - xmin # type: ignore
            xmin = xmin - (self.settings.extrapolate * xrange)
            xmax = xmax + (self.settings.extrapolate * xrange)

            if pd.isna(value): return 'NaN'
            elif log_value > xmax: return f'>{xmax}' # type: ignore
            elif log_value < xmin: return f'<{xmin}' # type: ignore

            else: return str(log_value) # type: ignore

        # Check if extrapolated and if yes, replace with string >x or <x.
        self.log_potency = [get_interpolated(d, p) for d, p in zip(data, potency)] # type: ignore


    def calc_empirical_stats(self) -> None:
        data = self.listofdata
        self.emp_emax = list()
        self.emp_emin = list()

        for d, m in zip(data, self.models):
            if not m.monotonic: raise ValueError('Currently calculating empirical stats only work with monotonic fitting function.')

            # Calculate function value for xmin and xmax and assign Emin the smaller, and Emax the larger y value.
            x_values = np.array([d.index.values.min(), d.index.values.max()])
            y_res = m.func(x_values, self.best_fit_params)
            self.emp_emax.append(y_res.max())
            self.emp_emin.append(y_res.min())