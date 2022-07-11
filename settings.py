import pandas as pd
from numpy import Inf

from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Optional, Dict, List, DefaultDict, Union
import datetime

VERSION = '1.1.3'

@dataclass
class Bound:
    min: Union[str, float] = -Inf
    max: Union[str, float] = Inf


#Class to store all settings
@dataclass
class Settings:
    # Fitting parameters
    fixed: List[str]
    shared: List[str]
    initial_guess: Dict[str, float]
    bounds: DefaultDict

    #Inline output
    show_figures: bool = True
    plot_exclude: bool = True # Show excluded values on plot with a different style (otherwise don't show at all).
    # print_parameters: bool = False # TODO: Remove?
    print_settings: bool = False
    
    #Export data (settings are exported automatically if export_results or export_figures is turned on).
    export_results: bool = False
    export_figures: bool = False
    figure_type: str = '.pdf' # Supports pdf, svg, png and more.
    figure_resolution: int = 150 # Increasing this makes plotting take longer.

    #Threshold fitting
    calc_thres: str = 'none' # 'none', 'abs' or 'rel'
    potency_thres: float = 0.5

    # Normalisiation
    normalise_data: bool = False # Normalise data in dataset to 0-1 from top/bottom of fit of norm_column data.
    norm_column: Any = 0 # Supports strings or positive or negative integers (neg. indicates counting from the end).

    #Export/import paths
    data_path: str = r'data/'
    data_extension: str = r'.csv' # Extension of data files to read.
    result_path: str = r'results/'
        
    # Formatting figures
    cmap: str = 'jet' # Colormap to use
    reverse_colors: bool = False # Reverse order of colors?
    color_norm: Optional[str] = 'k' # Special color to use for norm data.
    xlabel: str = 'pMHC/peptide'
    ylabel: str = 'Response'

    # Fitting settings
    fitting_method: str = 'leastsq'
    remove_bellshaped_data: bool = False
    extrapolate: float = 0
    '''How much to extrapolate the potency.
    This is interpreted as fraction of the log range of x values that should be added to either side of the scale.
    For example for a x-scale from -3 to 3 (log) a value of 0.25 would add 25 %/1.5 logs to either side (-4.5 - 4.5).'''

    def __init__(self):
        self.fixed = list()
        self.shared = list()
        self.initial_guess = dict()
        self.bounds = defaultdict(Bound)

    def prepare_saving(self):
        '''
        Returns a Pandas Series with all the important settings for saving.
        '''

        #Arrange relevant settings
        data =  dict()
        data['pythresfitter version'] = VERSION
        data['Date of fitting'] = str(datetime.date.today())
        data['Normalise data'] = "Yes" if (self.normalise_data) else "No"
        data['Normalisisation column'] = self.norm_column
        data['Potency threshold'] = f'{self.potency_thres} ({self.calc_thres})' if self.calc_thres else "None"
        data['Extrapolate potency'] = f'Yes ({self.extrapolate*100} %)' if self.extrapolate != 0 else 'No'
        data['Fitting method'] = self.fitting_method
        data['Shared parameters'] = self.shared if (self.shared) else "None"
        data['Parameter bounds:'] = [(k, self.bounds[k]) for k in self.bounds] if self.bounds else 'Default bounds'

        try: data['Fixed parameters'] = [(f, self.initial_guess[f]) for f in self.fixed] if (self.fixed) else "None"
        except KeyError:
            print('Fixed parameters settings could not be saved, because parameter names were spelled differently in initial_guess and fixed_parameters.')
            data['Fixed parameters'] = f'{self.fixed} (error saving values)'
            
        return pd.Series(data)
    
    def update(self, kws: dict):
        for k in kws:
            setattr(self, k, kws[k]) # Ignore key checks, since uninitialised fields do not exist yet
