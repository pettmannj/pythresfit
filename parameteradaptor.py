from __future__ import annotations
from copy import deepcopy
from settings import Settings
from lmfit.parameter import Parameters, Parameter
from models import FittingModelBase, norm
from typing import List, FrozenSet, Union
from pandas import DataFrame, Series
from numpy import ndarray

class ParameterAdaptor(Parameters):
    """Drop-in replacement for lmfit Parameters with added functionality to simplify access to the right parameters for each curve in a dataset.
    Specifically, this class keeps lists of fixed and shared parameters and all bounds separate.
    Changing any value automatically updates the underlying lmfit Parameter object that is used for the actual fitting.
    """

    _shared: List[str] # Shared parameters. Should be lower case.
    _fixed: List[str] # Fixed parameters. Should be lower case.
    _prefixes: List[str] # Prefixes from curve_names in dataset.
    _p_names: FrozenSet[str] # Set of names of parameters without prefixes from all models used.
    _model_name: str # Fitting model to be used.

    @property
    def shared(self) -> List[str]: return self._shared

    @property
    def fixed(self) -> List[str]: return self._fixed

    @shared.setter
    def shared(self, v: List[str]) -> None:
        self._shared = [vv.lower() for vv in v]

        # Check if given parameters exist
        for p in self._shared:
            if p not in self._p_names: raise ValueError(f"Could not find shared parameter '{p}' in model '{self._model_name}'.")

        # Set shared parameters
        for name in self._p_names:
            if name in self._shared:
                # Set all but first instance of parameter to be equal to the first instance.
                for pre in self._prefixes[1:]:
                    self[pre + name].expr = self[self._prefixes[0] + name].name
                    self[pre + name].vary = False

            else:
                for pre in self._prefixes[1:]:
                    self[pre + name].expr = None
                    if name not in self._fixed: self[pre + name].vary = True

    @fixed.setter
    def fixed(self, v: List[str]) -> None:
        self._fixed = [vv.lower() for vv in v]

        # Check if given parameters exist
        for p in self._fixed:
            if p not in self._p_names: raise ValueError(f"Could not find fixed parameter '{p}' in model '{self._model_name}'.")

        # Set fixed parameters
        for name in self._p_names:
            if name in self._fixed:
                # Set all but first instance of parameter to be equal to the first instance.
                for pre in self._prefixes:
                    self[pre + name].vary = False

            else:
                if name not in self._shared:
                    for pre in self._prefixes: self[pre + name].vary = True

                # Change first instance of parameter, since it is not touched when setting shared parameters.
                else: self[self._prefixes[0] + name].vary = True

    def get_param_values(self, name: str) -> List[float]:
        return [self[pre + name].value for pre in self._prefixes]

    def __init__(self, *args, **kwargs) -> None:
        self._shared = list()
        self._fixed = list()
        super().__init__(*args, **kwargs)

    def initialise(self, models: List[FittingModelBase], settings: Settings):
        if len(models) == 1:
            self._model_name = models[0]._name
            self._prefixes = [models[0]._prefix]
            self._p_names = frozenset(models[0]._p_names)

        else:
            # Create model name
            self._model_name = ' + '.join([m._name for m in models])
            for p, c in zip(models, models[1:]):
                if p._name != c._name: break

            else:
                self._model_name = models[0]._name # If all names are the same.

            # Set prefixes
            self._prefixes = [m._prefix for m in models]

            # Set parameter names
            self._p_names = frozenset([p for m in models for p in m._p_names])

        # Generate parameters with prefix for each column in dataset.
        for model in models:
            p = model.default_params()

            for k in p:
                self.add_many(p[k]) # List comprehension fails from a ValueError in Parameter

        # Set bounds
        for b in settings.bounds:
            if b not in self._p_names: raise ValueError(f"Could not set bounds for parameter {b}. Parameter not found in model '{self._model_name}'.")

            for pre in self._prefixes:
                self[pre + b].min = settings.bounds[b].min
                self[pre + b].max = settings.bounds[b].max

        # Set initial guesses
        for k in settings.initial_guess:
            if k not in self._p_names: raise ValueError(f"Could not set initial guess for parameter {k}. Parameter not found in model '{self._model_name}'.")

            for pre in self._prefixes:
                self[pre + k].value = settings.initial_guess[k]

        # Set fixed and shared parameters. This should be last to avoid it being overwritten.
        self.shared = [s.lower() for s in settings.shared]
        self.fixed = [f.lower() for f in settings.fixed]

    def guess(self, models: List[FittingModelBase], data: DataFrame) -> None:
        # Update parameter based on model guesses using the data.
        for c, m in zip(data, models):
            initial_guesses = m.guess(data[c])
            for k in initial_guesses:
                if self[k].vary: self[k].value = initial_guesses[k]

        # Make sure fixed and shared are set again.
        self.fixed = self.fixed
        self.shared = self.shared

    def normalise(self, miny: Union[float, ndarray, Series], range: float, add: float = 0) -> ParameterAdaptor:
        pa = deepcopy(self)

        for p in pa._p_names:
            if ('top' in p.lower()) or ('bottom' in p.lower()):
                for i, pre in enumerate(pa._prefixes):
                    # Add is to move descending curve up to it starts at 1 rather than at 0 going negative.
                    pa[pre + p].value = norm(pa[pre + p].value, miny[i], range) + add # type: ignore

        return pa

    def __deepcopy__(self, memo):
        """Implementation of Parameters.deepcopy().

        The method needs to make sure that asteval is available and that all
        individual Parameter objects are copied.

        """
        _pars = ParameterAdaptor(asteval=None)

        # find the symbols that were added by users, not during construction
        unique_symbols = {key: self._asteval.symtable[key]
                          for key in self._asteval.user_defined_symbols()}
        _pars._asteval.symtable.update(unique_symbols)

        # we're just about to add a lot of Parameter objects to the newly
        parameter_list = []
        for key, par in self.items():
            if isinstance(par, Parameter):
                param = Parameter(name=par.name,
                                  value=par.value,
                                  min=par.min,
                                  max=par.max)
                param.vary = par.vary
                param.brute_step = par.brute_step
                param.stderr = par.stderr
                param.correl = par.correl
                param.init_value = par.init_value
                param.expr = par.expr
                param.user_data = par.user_data
                parameter_list.append(param)

        _pars.add_many(*parameter_list)

        _pars._shared = self._shared
        _pars._fixed = self._fixed
        _pars._prefixes = self._prefixes
        _pars._p_names = self._p_names
        # _pars._curve_names = self._curve_names
        _pars._model_name = self._model_name

        return _pars
