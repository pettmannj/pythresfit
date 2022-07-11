from __future__ import annotations

import pandas as pd
from typing import List, Union, Sequence, Optional
from dataset import DataSet
from plotting import Plot

import os.path
from uuid import uuid4

#Class to save all results
class ExportHandler:
    _datasets: List[DataSet] # Contains all datasets (norm or not)
    _plots: List[Plot] # Contains all plots (norm or not)

    iD: str # Unique ID to identify these results later on
    _results_extension = '.csv'

    @property
    def datasets(self): return self._datasets

    @property
    def plots(self): return self._plots

    @datasets.setter
    def datasets(self, v):
        self._datasets = v
        self._check_names(self._datasets)

    @plots.setter
    def plots(self, v):
        self._plots = v
        self._check_names(self._plots)

    def _check_names(self, items: Union[Sequence[DataSet], Sequence[Plot]]) -> None:
        names = [item.name for item in items]
        
        if len(names) != len(set(names)):
            raise AttributeError(f'Found multiple datasets or plots with the same name in {self.__class__.__name__}.')
    
    def __init__(self, datasets: Optional[List[DataSet]] = None, plots: Optional[List[Plot]] = None) -> None:
        self.iD = str(uuid4())[:4]
        self.datasets = datasets if datasets else list()
        self.plots = plots if plots else list()

    def add_dataset(self, dataset: DataSet) -> None:
        self.datasets.append(dataset)
        self._check_names(self.datasets)

    def add_plot(self, plot: Plot) -> None:
        self.plots.append(plot)
        self._check_names(self.plots)

    def _save_plot(self, plot: Plot):
        path = plot.dataset.settings.result_path
        extension = plot.dataset.settings.figure_type
        resolution = plot.dataset.settings.figure_resolution
        name = plot.dataset.name

        name = f'{self.iD}-{name}'
        fpath = os.path.join(path, name + extension)
        plot.figure.savefig(fpath, dpi=resolution) # type: ignore

    def _save_results(self):
        results = [d.export_results() for d in self.datasets if d.settings.export_results]
        header = [d.name for d in self.datasets for _ in d.data.columns if d.settings.export_results]
 
        # Write results
        if results:
            df_results = pd.concat(results)
            df_results.index = header # type: ignore
            name = f'{self.iD}-results'
            fpath = self.datasets[0].settings.result_path + name + self._results_extension # Just save in path from first settings file.
            df_results.T.to_csv(fpath)

    def _save_settings(self):
        # All based on just the first settings file
        df_settings = self._datasets[0].settings.prepare_saving()
        fpath = self.datasets[0].settings.result_path + f'{self.iD}-settings' + self._results_extension
        df_settings.to_csv(fpath)

    def save(self):
        # Save plots
        for p in self.plots:
            if p.dataset.settings.export_figures: self._save_plot(p)

        # Save numeric results
        self._save_results()

        # Save settings
        if self._datasets[0].settings.export_figures or self._datasets[0].settings.export_results:
            self._save_settings()