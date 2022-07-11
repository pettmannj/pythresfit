from __future__ import annotations
from collections import OrderedDict
from pandas import DataFrame, Series
from numpy import ndarray, log10, inf
from lmfit import Parameters
from typing import Hashable, List, Mapping, Sequence, Union, Callable, Container

def calc_prefix(name: Union[str, Sequence[Hashable]]) -> Union[str, List[str]]:
    if isinstance(name, str): return f'c{str(abs(hash(name)))[:8]}_'

    if len(name) == 1: return [''] # If there is only 1 curve, we don't need a prefix.

    results = list()
    for item in name: results.append(f'c{str(abs(hash(item)))[:8]}_')
    return results


def norm(values: Union[float, ndarray, DataFrame], norm_values: Union[float, ndarray], range: float) -> Union[float, ndarray, DataFrame]:
    return (values - norm_values) / range

class FittingModelBase:
    _prefix: str
    _name: str
    func: Callable
    _p_names: List[str]
    monotonic: bool

    def __init__(self, func: Callable, name: str, p_names: List[str], monotonic: bool, prefix: str = ''):
        self._prefix = prefix
        self._name = name
        self._p_names = p_names
        self.func = func
        self.monotonic = monotonic

    def guess(self, data: Container) -> Parameters:
        raise NotImplementedError

    def solve(self, data: Union[float, ndarray], params: Mapping) -> Union[float, ndarray]:
        raise NotImplementedError

    def default_params(self) -> Parameters:
        raise NotImplementedError

class Sigmoidal5PModel(FittingModelBase):
    def __init__(self, prefix=''):
        super().__init__(func=self._sigmoidal_func, name='5P Sigmoidal model', p_names=['log_ec50', 'top', 'bottom', 'hill', 's'], monotonic=True, prefix=prefix)

    def _sigmoidal_func(self, x: ndarray, parameters: Mapping[str, float]) -> ndarray:
        d = (1+(2**((1/parameters[self._prefix + 's'])-1))*((10**parameters[self._prefix + 'log_ec50']/x)**parameters[self._prefix + 'hill']))**parameters[self._prefix + 's'] # type: ignore
        n = parameters[self._prefix + 'top'] - parameters[self._prefix + 'bottom']
        return parameters[self._prefix + 'bottom'] + (n/d)

    # See also https://books.google.com.au/books?id=xuYf6tcVdqYC&lpg=PA329&dq=4pl%20elisa%20curve%20fitting&pg=PA327#v=onepage&q&f=false
    def solve(self, y: Union[float, ndarray], parameters: Mapping[str, float]):
        n = ((parameters[self._prefix + 'top'] - parameters[self._prefix + 'bottom']) / (y - parameters[self._prefix + 'bottom']))**(1/parameters[self._prefix + 's']) - 1 # type: ignore
        d = 2**(1/parameters[self._prefix + 's'] - 1)
        return 10**parameters[self._prefix + 'log_ec50'] / (n/d)**(1/parameters[self._prefix + 'hill'])


    def default_params(self) -> Parameters:
        params = Parameters()
        params.add(self._prefix + 'log_ec50', value=1)
        params.add(self._prefix + 'hill', value=1)
        params.add(self._prefix + 'top', value=1)
        params.add(self._prefix + 'bottom', value=0)
        params.add(self._prefix + 's', value=1, vary=False)
        
        return params

    def guess(self, data: Series) -> OrderedDict:
        params = OrderedDict()
        params[self._prefix + 'top'] = data.max()
        params[self._prefix + 'bottom'] = data.min()

        # Calc estimated log_ec50. If this is between 0 and the lowest non-zero value set it to the lowest non-zero value to avoid log(0) = inf.
        greater50 = data[data > (0.5 * data.max())] # All values over 50 % Max.

        if not greater50.empty:
            est_log_ec50 = log10(greater50.index.values.min())
            params[self._prefix + 'log_ec50'] = est_log_ec50 if abs(est_log_ec50) != inf else data.index[data.index > 0].values.min() # type: ignore

        return params