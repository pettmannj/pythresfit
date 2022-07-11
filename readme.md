# Python threshold fitter
Version: v1.1.3

Date: 15/03/2021

Author: Johannes Pettmann

Changes:
* Unique run ID associated with all files.
* Settings can be saved alongside results.
* Normalised top fixed to 1.
* Better guessing of normalised bottom and non-normalised top
* Normalisation is now based on values at xmin and xmax of curve, not on 'Top' and 'Bottom' parameter. This might change results significantly.
* Added version and date of fitting to settings file.
* Updated style of plots to resemble Prism style with Helvetica as font and reversed jet as colormap.
* Norm data can now be plotted in a different color.
* All numerical results are now exported in one file (separate for normalised datat), rather than many.
* Values (or whole rows) can be excluded by placing an asterix in the cell of the CSV file.
* Automatic bell-shape removal to exclude data that is decreasing at high doses.
* Adaptive guessing based on data.

A python script to fit dose-response curves with a 4P/5P sigmoidal function and calculate a threshold. I was devleoped to fit T cells stimulated with a titration of ligands, but can also be used for other datasets. Further fitting models can be added where required. I developed this script to allow extracting an absolute or relative threshold value, rather than relying on an EC50 for characterizing the respose. In contrast to extracting the EC50, this method is independent of the Emax. This prevents significant errors when the latter is not well defined due to extrapolation. Either a absolute threshold can be applied, or data can be normalised to a given column and a relative threshold applied.

This code was used in our 2021 paper to fit the data: [Pettmann et al. The discriminatory power of the T cell receptor. eLife. 2021.](https://elifesciences.org/articles/67092)

### Data
This script imports data from CSV files, where the columns are referring to the curves and the index to the x-values. One file consistutes one dataset. All data within one dataset is fit together and therefore parameters can be shared, if desired.
Data is fitted on a linear scale and therefore this script accepts 0 as x-values. All the columns in each dataset (i.e. file) need to have the same number of x-values. If this is not the case (e.g. because values had to be excluded), one has to manually fill in nan_values (i.e. -99999). These values will be ignored for the actual fit.
The data path can be either pointing towards a single file or to a folder. If the path is a folder, all CSV files in that folder will be processed, but any subfolders will be ignored.
Note: Negative values can sometimes cause problems. For example, a curve from cytokine data that was all negative caused the whole fit for the dataset to not converge. Adjusting the baseline, replacing negatives with zeros, or removing that data resolves the problem.

### Parameters
This script fits a 5 parameter sigmoidal curve. Parameters can be constrained, shared between curves and fixed. For example S (asymetry parameter) can be constrained to 1, making is effectively a 4P sigmoidal. I recommend this as a default and all 3 default settings have S fixed to 1.

### Settings
Change settings such as:
* Data and results path
* Inline output (e.g. if figures should be shown)
* What data should be saved (e.g. fitted parameters or threshold data)
* Shared parameters in fit (usually bottom, top and/or hill)
* Bounds and initial guesses for parameters
* If and what parameters should be fixed (usually S=1 to make a 4P sigmoidal)
* Column to normalise to (can either be index (0 = first column) or the name of the column in the csv (e.g. '9V' or '1000')).
* Threshold value

### How to use
Make sure you have the Jupyter notebook, the main.py file and the data and results folders in the same directory. Requires at least Python 3.7.
1. Add data in the right format to data folder (subfolders don't work and can be used to hide data). Make sure to fill empty or excluded cells with the nan_value (e.g. -99999).
2. Select default setting to be used
3. Change between viewing and exporting mode
4. Adjust settings if necessary
5. Run
6. Analyse data in results folder (if exporting)
