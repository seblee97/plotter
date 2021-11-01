# Plotter

This package allows for simple visualisation of data collected (e.g. from machine learning experiments). It is primarily designed for complimentary use with [this data logger package](https://github.com/seblee97/data_logger) and [this runner package](https://github.com/seblee97/run_modes).

### Getting Started

This package is written in Python 3. Requirements are minimal:

- matplotlib
- pandas
- numpy

Installation can either be performed by cloning the repository and running ```pip install -e .``` from the package root, or via ```pip install -e git://github.com/seblee97/plotter.git#egg=plotter-seblee97```.

### Example Usage

The purpose of the package is to facilitate visualisation of ablations in machine learning experiments. The Plotter class in ```plotter.py``` is for creating plots of a range of scalar values logged during a specific experiment. Other functions in ```plot_functions.py``` allow for visualising and comparing different runs including consolidating seeds and showing variation across seeds. 