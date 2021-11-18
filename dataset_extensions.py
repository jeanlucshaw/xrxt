"""
Application specific extensions to the xarray Dataset class.
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

@xr.register_dataset_accessor('xrxt')
class DatasetExtension:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    # ==================
    # Ragged array tools
    # ==================
    def get_ragged(self, var_, size_var_, index=None, coord_index=False):
        """
        Getter function for variables organized in ragged form.

        Parameters
        ----------
        var_: str or list of str
            name of the variable(s) to extract.
        size_var_: str
            name of the ragged size variable.
        index: int, list of int, or None
            zero-indexed number of the data segment(s) in the sequence. The
            default is to return a list of all the data segments.
        coord_index: bool
            index is supplied as value(s) in the coordinate of `size_var_`.
            By default (False), index is assumed to be sequential.

        Returns
        -------
        xarray.Dataset, or list of xarray.Dataset:
            the requested data segment(s).

        """
        # Funnel all 

        # If many variables, ensure they have same coordinate
        if isinstance(var_, list):
            if len(self._obj[var_].dims) > 1:
                raise KeyError('Requested variables have different coordinates.')

        # Default is to make a list of all segments
        if index is None:
            index = range(self._obj[size_var_].size)

        # Ensure that index is an iterable type
        if isinstance(index, (int, float, str)):
            index = [index]
        elif not isinstance(index, (tuple, list, np.ndarray)):
            index = np.array(index)
            if index.size == 1:
                index = [index.tolist()]

        # Get variable and coordinate variable dimension names
        var_dim, = self._obj[var_].dims
        crd_dim, = self._obj[size_var_].dims

        # Initialize output
        segments = list()

        # Loop over requested indices
        for i_ in index:
            
            # Find segment start and end points
            if coord_index:
                # start_index = self._obj[size_var_].isel({crd_dim: slice(0, i_)}).sum()
                # end_index = self._obj[size_var_].isel({crd_dim: i_}) + start_index - 1
                start_index = self._obj[size_var_].sel({crd_dim: slice(None, i_)})[:-1].sum()
                end_index = self._obj[size_var_].sel({crd_dim: i_}) + start_index - 1
            else:
                start_index = self._obj[size_var_].isel({crd_dim: slice(0, i_)}).sum()
                end_index = self._obj[size_var_].isel({crd_dim: i_}) + start_index - 1

            # Isolate data segment
            segment = self._obj[var_].sel({var_dim: slice(start_index, end_index)})

            # Append to output
            segments.append(segment)

        # Unpack if single element list
        if len(segments) == 1:
            segments = segments[0]

        return segments

    # ==============
    # Plotting tools
    # ==============
    def plot_profile(self, variable, **kwargs):
        """
        Plot profile of a data variable against depth.

        The intended use case is to reduce the dataset
        to one dimension, for example via the `sel` or
        `mean` methods, and then call this extensions
        method.

        Parameters
        ----------
        variable: str
            name of the data variable to plot.
        **kwargs: dict
            parameter value pairs passed to `xarray.plot`.

        """
        # Handle defautls
        plot_defaults = dict(color='k', lw=0.5)
        plot_params = {**plot_defaults, **kwargs}

        # Unpack the dimension name
        y_axis, = self._obj[variable].dims

        # Plot
        self._obj[variable].plot(y=y_axis,
                                 yincrease=False,
                                 **plot_params)


    def plot_climatology(self, variable, spread, extremes=None, ax=None):
        """
        Plot climatology of a data variable against depth.

        The intended use case is to reduce the dataset
        to one dimension, for example via the `sel` or
        `mean` methods, and then call this extensions
        method.

        Parameters
        ----------
        variable: str
            name of the climate mean to plot.
        spread: str
            name of the variable to use as +- envelope.
        extremes: tuple of str
            names of the climate min and max variables. 
        ax: pyplot.Axes or None
            where to draw the climatology.

        Returns
        -------
        pyplot.Axes
            where the climatology is drawn.

        """
        # Create the axes if none supplied
        if ax is None:
            _, ax = plt.subplots()

        # Unpack the dimension name
        y_axis, = self._obj[variable].dims
        y_ = self._obj[y_axis].values

        # Envelope +- 1 spread
        x_min = (self._obj[variable] - self._obj[spread]).values
        x_max = (self._obj[variable] + self._obj[spread]).values
        ax.fill_betweenx(y_, x_min, x_max, color='powderblue', ls=[])

        # Envelope +- 0.5 spread
        x_min = (self._obj[variable] - 0.5 * self._obj[spread]).values
        x_max = (self._obj[variable] + 0.5 * self._obj[spread]).values
        ax.fill_betweenx(y_, x_min, x_max, color='skyblue', ls=[])

        # Extreme values
        if extremes is not None:
            ax.plot(self._obj[extremes[0]], y_, 'k:', lw=0.5)
            ax.plot(self._obj[extremes[1]], y_, 'k:', lw=0.5)

        # Plot mean
        self._obj[variable].plot(y=y_axis,
                                 color='k',
                                 lw=0.5,
                                 ls='--',
                                 yincrease=False,
                                 ax=ax)

        return ax
