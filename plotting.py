from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.cm import get_cmap
from numpy import logspace, log10, ndarray
from typing import Optional

from matplotlib import rcParams
from matplotlib.font_manager import FontProperties, FontManager

from dataset import DataSet

# Change formatting of plots to be more similar to what GraphPad Prism (with Helvectica as font).
FONT_LABEL = FontProperties(fname=FontManager().findfont('Helvetica'), size=18)
FONT_TITLE = FontProperties(fname=FontManager().findfont('Helvetica'), size=20)
FONT_AXIS = FontProperties(fname=FontManager().findfont('Helvetica light'), size=16, weight='light')

# Changes font for plots to Helvetica or the system default if not present.
rcParams['figure.figsize'] = (7.6, 5.1)
rcParams['figure.dpi'] = 75
rcParams['figure.frameon'] = False

# Better fonts in exports of PDF/PostScript 
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['svg.fonttype'] = 'none' # Whene exporting as SVG render text as font not as path.


class Plot:
    figure: Optional[Figure] = None
    dataset: DataSet
    name: str

    def __init__(self, dataset: DataSet) -> None:
        self.dataset = dataset
        self.name = self.dataset.name

    @staticmethod
    def _get_x_space(x_values: ndarray):
        x_min = log10(x_values[x_values > 0].min()) #Smallest index excluding 0
        x_max = log10(x_values.max())
        return logspace(x_min, x_max, 100)

    def plot_fits(self, name):
        x_incl_data = self.dataset.included_data.index.values
        y_incl_data = self.dataset.included_data.values
        x_excl_data = self.dataset.excluded_data.index.values
        y_excl_data = self.dataset.excluded_data.values
        listofdata = self.dataset.listofdata

        #Plot results
        fig = plt.figure()
        
        # Add special color for norm column?
        if self.dataset.settings.color_norm:
            colors = list()
            cmap = get_cmap(self.dataset.settings.cmap, len(self.dataset.data.columns))
            n = len(self.dataset.data.columns) - 1
            if n >= 1: colors.extend([cmap(i) for i in range(n)])
            
        else:
            cmap = get_cmap(self.dataset.settings.cmap, len(self.dataset.data.columns))
            colors = [cmap(i) for i in range(len(self.dataset.data.columns))]
            
        # Reverse order of colors?
        if self.dataset.settings.reverse_colors: colors = colors[::-1]

        if self.dataset.settings.color_norm:
            ni = self.dataset.data.columns.get_loc(self.dataset.norm_col)
            # Position of norm color is irrespective of reversing.
            colors.insert(ni, self.dataset.settings.color_norm) # type: ignore

        colors = iter(colors)

        lines = []

        for i, _ in enumerate(self.dataset.data):
            color = next(colors)
            x_fit = self._get_x_space(listofdata[i].index.values)
            y_fit = self.dataset.models[i].func(x_fit, self.dataset.best_fit_params)
            lines.append(plt.plot(x_fit, y_fit, '-', color=color))
            plt.plot(x_incl_data, y_incl_data[:, i], '.', color=color)
            plt.plot(x_excl_data, y_excl_data[:, i], 'x', color=color)

        lines = [l[0] for l in lines]
        
        #Draw threshold line
        if self.dataset.log_potency:
            plt.hlines(self.dataset.settings.potency_thres, x_incl_data.min(), x_incl_data.max(), linestyles='dashed') # type: ignore
            
        ncol = 1 # Number of columns in legend
        if y_incl_data.shape[1] > 8: ncol = 2
        
        plt.gca().set_xscale('log')
        plt.xlabel(self.dataset.settings.xlabel, fontproperties=FONT_LABEL)
        plt.ylabel(self.dataset.settings.ylabel, fontproperties=FONT_LABEL)
        plt.gca().legend(lines, self.dataset.data.columns.values, prop=FONT_LABEL, ncol=ncol, bbox_to_anchor=(1,1))
        plt.gca().set_title(name, fontproperties=FONT_TITLE)
        
        # Remove figure frame on top and bottom
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_linewidth(1)
        plt.gca().spines['left'].set_linewidth(1)
        
        for label in plt.gca().xaxis.get_ticklabels():
            label.set_font_properties(FONT_AXIS)
            
        for label in plt.gca().yaxis.get_ticklabels():
            label.set_font_properties(FONT_AXIS)
         
        self.figure = fig
        plt.show(block=False)