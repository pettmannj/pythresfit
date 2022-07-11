
# %%
from fitting import main
from settings import Settings

# %%
settings = Settings()

# %%
# Parameter settings
# Shared and fixed parameters (case-insensitive)
# S is usually fixed. Hill is usually shared. Bottom is often shared.
settings.shared = ['top']
settings.fixed = ['s']

# Manually define initial guesses (otherwise they are automatically generated from the data).
# If there parameter is fixed, this initial guess is the value it will be fixed to.
# E.g. usually S is fixed to 1. Often the bottom is fixed to 0.
settings.initial_guess['s'] = 1
# settings.initial_guess['bottom'] = 0

# Optional constrains on parameters (bounds).
# Recommended only to be used if the fit otherwise doesn't work. Alternatively, using a different fitting method can sometimes work (see below).
# settings.bounds['log_ec50'].max = -3
# settings.bounds['log_ec50'].min = 3

# settings.bounds['top'].max = 100000
# settings.bounds['top'].min = 0
# settings.initial_guess['top'] = 500


# Hill settings
# MUST be positive for ascending curves (e.g. cytokines or CD69) and negative for descending curves (e.g. tetramers).
settings.initial_guess['hill'] = -1
settings.bounds['hill'].max = -0.3
settings.bounds['hill'].min = -1.5

# Export settings
settings.export_figures = False
settings.export_results = False

settings.normalise_data = True # Normalize data from [0-1] defined by normalization column.
# settings.norm_bottom = False # Align bottom during normalization (baseline correction)? This is useful if the bottom is not the same.
settings.calc_thres = 'rel' # 'none', 'rel', or 'abs' for no threshold, an relative threshold, respectively, an absolute threshold.
settings.potency_thres = 0.6 # Threshold (usually [0-1] for a relative threshold).

settings.remove_bellshaped_data = False # Remove bell-shaped datapoint automatically? Does NOT work with descending curves (i.e. tetramers).

settings.norm_column = 0 # Index or name of column to be used for normalization.
settings.reverse_colors = False # Reverse the order of colors that are used?
settings.color_norm = 'k' # Color to use for the data from the normalization column (e.g. 'k' for black).
settings.plot_exclude = True # True or False. If true, excluded (asterix in data or by automatic bell-shape removal) data will be shown as 'x'. If false, it will be completely omitted.
settings.extrapolate = 0.1 # How much to extrapolate the potency. 0 (default) indicates no extrapolation.
# settings.cmap = 'jet' # Colormap to be used. Defaults to 'jet'.
# settings.xlabel = 'pMHC (ng/well)' # Label to be displayed on x-axis of graphs.

# Fitting algorithm to be used. Changing this can help if the standard fitting doesn't look right. 
# settings.fitting_method = 'nelder' # 'leastsq' (default) or 'nelder'

# %%
handler = main(settings)
