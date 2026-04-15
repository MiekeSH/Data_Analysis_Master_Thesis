import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import warnings
from xarray.coding.times import SerializationWarning
import cftime
from scipy.optimize import curve_fit
import matplotlib.path as mpath
from matplotlib.colors import LogNorm, SymLogNorm
import regionmask



'''
Function to get the correct filepath for a specified data scenario, which is the string of the
directory where the data for that scenario is stored. Months = 1 gives monthly data, months = 12 
gives yearly data, if detrend = 'detrended' it gives the detrended arrays. 

'''
    

def filepath(data_scenario, months, detrend = 'sliced_', model = ''):
    if months == 1:
        timescale = ''
    elif months == 12:
        timescale = 'yearly_' 
    else:
        print('Give valid amount of months (1 fro monthly, 12 for yearly data)')
    
    if detrend == 'raw':      # raw data   (For Data_preprocessing)
        if model == 'CNRM':
            file_tas = [f for f in os.listdir(data_scenario) if f.startswith('tas_')][0]
            file_prsn = [f for f in os.listdir(data_scenario) if f.startswith('prsn_')][0]
            file_pr = [f for f in os.listdir(data_scenario) if f.startswith('pr_')][0]
            return data_scenario + '/' + file_tas, data_scenario + '/' + file_prsn, data_scenario + '/' + file_pr 
            
        else:
            file_tas = [f for f in os.listdir(data_scenario) if f.startswith('tas_')][0]
            file_prsn = [f for f in os.listdir(data_scenario) if f.startswith('prsn_')][0]
            file_pr = [f for f in os.listdir(data_scenario) if f.startswith('pr_')][0]
            file_sic = [f for f in os.listdir(data_scenario) if f.startswith('remapped_SIC')][0]
            return data_scenario + '/' + file_tas, data_scenario + '/' + file_prsn, data_scenario + '/' + file_pr, data_scenario + '/' + file_sic          

    else:
        if model == 'CNRM':
            file_tas = [f for f in os.listdir(data_scenario) if f.startswith(f'{timescale}{detrend}tas')][0]
            file_prsn = [f for f in os.listdir(data_scenario) if f.startswith(f'{timescale}{detrend}prsn')][0]
            file_pr = [f for f in os.listdir(data_scenario) if f.startswith(f'{timescale}{detrend}pr.')][0]
            file_snfr = [f for f in os.listdir(data_scenario) if f.startswith(f'{timescale}{detrend}snfr')][0]    
            return data_scenario + '/' + file_tas, data_scenario + '/' + file_prsn, data_scenario + '/' + file_pr, data_scenario + '/' + file_snfr

        else:
            file_tas = [f for f in os.listdir(data_scenario) if f.startswith(f'{timescale}{detrend}tas')][0]
            file_prsn = [f for f in os.listdir(data_scenario) if f.startswith(f'{timescale}{detrend}prsn')][0]
            file_pr = [f for f in os.listdir(data_scenario) if f.startswith(f'{timescale}{detrend}pr.')][0]
            file_snfr = [f for f in os.listdir(data_scenario) if f.startswith(f'{timescale}{detrend}snfr')][0]    
            file_sic = [f for f in os.listdir(data_scenario) if f.startswith(f'{timescale}{detrend}sic')][0]    
            return data_scenario + '/' + file_tas, data_scenario + '/' + file_prsn, data_scenario + '/' + file_pr, data_scenario + '/' + file_snfr, data_scenario + '/' + file_sic
    

'''
Function to create a dictionary of the dataset. Input is directory of the data, months and detrend as in function filepath
and model only needs specifying for CNRM in order to open the files. 
'''

warnings.simplefilter("ignore", SerializationWarning) # suppresses the warning that datetime does not work


def create_dictionary_data(directory, months, detrend = 'sliced_', model = ''):

    if model == 'CNRM':
        base_dirs = {
            'pi': directory + '/PI',
            '2K': directory + '/GWL2',
            '4K': directory + '/GWL4'
        }
    else:
        base_dirs = {
            'pi': directory + '/pi_control_smhi',
            '2K': directory + '/gwl2p0_knmi',
            '4K': directory + '/gwl4p0_dmi'
        }

    data = {}

    for experiment, base_dir in base_dirs.items():
        data[experiment] = {}
        if detrend == 'raw':
            if model == 'CNRM':
                variables = ['tas', 'prsn', 'pr']
            else:
                variables = ['tas', 'prsn', 'pr', 'siconc']
        
        else:
            if model == 'CNRM': 
                variables = ['tas', 'prsn', 'pr', 'snfr']
            else:
                variables = ['tas', 'prsn', 'pr', 'snfr', 'sic']
        if (model == 'UKESM') & (experiment == '2K') & (detrend == 'raw'):
            variables = ['tas', 'prsn', 'pr', 'siconca']
                
        for var, path in zip(variables, filepath(base_dir, months, detrend, model)):
            ds = xr.open_dataset(path)
            data[experiment][var] = ds[var]

    return data

def create_xarray_data(directory, months, detrend='sliced_', model=''):

    if model == 'CNRM':
        base_dirs = {
            'pi': directory + '/PI',
            '2K': directory + '/GWL2',
            '4K': directory + '/GWL4'
        }
    else:
        base_dirs = {
            'pi': directory + '/pi_control_smhi',
            '2K': directory + '/gwl2p0_knmi',
            '4K': directory + '/gwl4p0_dmi'
        }

    data = {}

    for experiment, base_dir in base_dirs.items():

        paths = filepath(base_dir, months, detrend, model)

        data[experiment] = xr.open_mfdataset(paths, combine="by_coords", parallel=True)
        
    return data


'''
Function for calculating the yearly mean from monthly means, can work with time arrays
of type cftime or datetime64
'''



def yearly_mean(data_array, snfr_threshold_snow = 0, snfr_threshold_pr = 0):

    if isinstance(data_array, (xr.DataArray, xr.Dataset)):      
           
        t0 = data_array.time.values[0]
        
        if isinstance(t0, cftime.Datetime360Day):
            yearly_array = data_array.resample(time="YS").mean(dim='time')
            print('360 calender, normal mean taken')
            
            years = yearly_array.time.dt.year.values
            yearly_array = yearly_array.assign_coords(time=years)

            return yearly_array
            
        else: 
            days = data_array.time.dt.days_in_month

            weights = days.groupby("time.year") / days.groupby("time.year").sum()
             
            weighted_data = data_array * weights
            
            yearly = weighted_data.groupby("time.year").sum(dim="time")

            yearly = yearly.rename({"year": "time"})

            return yearly

    elif isinstance(data_array, dict):

        if 'snfr'in data_array.keys():
            new_dict = {}

            for k, v in data_array.items():
                if k == 'snfr':
                    continue
                    
                new_dict[k] = yearly_mean(v, snfr_threshold_snow, snfr_threshold_pr)

            new_dict['snfr'] = snow_fraction(new_dict['prsn'], new_dict['pr'], snfr_threshold_snow, snfr_threshold_pr)

            return new_dict

        else:
            return {k: yearly_mean(v, snfr_threshold_snow, snfr_threshold_pr) for k, v in data_array.items()}

    # --- lists (your variability output) ---
    elif isinstance(data_array, list):
        return [yearly_mean(v, snfr_threshold_snow, snfr_threshold_pr) for v in data_array]

    else:
        print('Data is not a dictionary, list, xarray or xr dataset. Nothing was changed.')
        return data_array

        
'''
Function to calculate the snow fraction from arrays of the snowfall and the total precipitation.
Values for pr =< 0 are made nan as there is no snowfraction if there is no precipitation. 
Values for prsn < 0 are made 0 as this indicated model error. 
Where snowfall exceeds total precipitation they are made equal. 

'''
def snow_fraction(prsn, pr, snow_threshold = 0, pr_threshold = 0):
    pr = pr.where(pr > pr_threshold)
    prsn = prsn.where(pr > pr_threshold)

    prsn = prsn.where(prsn > snow_threshold, 0)

    difference = pr-prsn
    prsn_overshoot = difference.where(difference < 0, 0)
    # print(prsn_overshoot.min().values)
    prsn = prsn + prsn_overshoot # make it so snowfall cannot be larger than precipitation
    
    snow_fraction = prsn/pr
        
    return snow_fraction 



'''
Function to make a simple 2d plot (rectangular)
'''

def standard_2d_plot(array_2d, title = 'title', savename = 'quickplot.png', rotation = 0, max_scale = 'no_max', min_scale = 0, color = 'coolwarm'):
    
    # Create a figure and axis with a map projection
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})

    plot_kwargs = dict(
        ax = ax,
        transform = ccrs.PlateCarree(),
        cmap=color,
        cbar_kwargs={'label': title, 'orientation': 'horizontal', 'shrink': 0.6}
    ) 

    if max_scale != 'no_max':
        plot_kwargs.update(vmin=min_scale, vmax=max_scale)
        
    # ensure non-lazy data
    if hasattr(array_2d.data, "compute"):
        array_2d = array_2d.compute()

    valid = array_2d.where(~np.isnan(array_2d), drop=True)

    array_2d.plot(**plot_kwargs)


    lon = valid.lon
    lat = valid.lat

    ax.set_extent([lon.min()-2, lon.max()+ 2, lat.min() - 2, lat.max() + 2], crs = ccrs.PlateCarree())
    
    if rotation == 1:
        for cbar_ax in fig.axes:
            for label in cbar_ax.get_xticklabels():
                label.set_rotation(30)
                
    # Add coastlines and borders
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    plt.savefig('/nobackup/users/hartevel/data/Data_analysis/figures/' + savename)
    
    plt.show()


'''
Function to make a simple 2d plot (circular). It can also be used for subplots, then use ax and fig to specify. cbar_scale can be 'linear', 'log' or 'symlog'

'''
def standard_2d_plot_polar(array_2d, bar_label='title', savename='quickplot.png',
                           rotation=0, max_scale='no_max', min_scale='no_min',
                           color='coolwarm', cbar_scale="symlin", ax = 'undefined', fig = 'undefined', 
                           linthresh=1e-5, linscale=1.0, font = 14, min_lat = 60):

    if ax == 'undefined':
        # Create figure with POLAR projection
        proj = ccrs.NorthPolarStereo(central_longitude=rotation)
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": proj})

    # Circular clipping
    def set_circular_boundary(ax):
        theta = np.linspace(0, 2*np.pi, 200)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T * radius + center
        ax.set_boundary(mpath.Path(verts), transform=ax.transAxes)

    # Plot data
    plot_kwargs = dict(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=color,
        add_colorbar=False
    )

    if max_scale == "no_max":
        vmax = np.nanmax(np.abs(array_2d))
    else:
        vmax = max_scale

    if min_scale == "no_min":
        if cbar_scale == 'log':
            vmin = 1e-5
        else:
            vmin = np.nanmin(np.abs(array_2d))
    else:
        vmin = min_scale

    # --- COLORBAR SCALING OPTIONS ---
    if cbar_scale == "log":
        plot_kwargs["norm"] = LogNorm(vmin=vmin, vmax=vmax)

    elif cbar_scale == "symlog":
        plot_kwargs["norm"] = SymLogNorm(
            linthresh=linthresh,
            linscale=linscale,
            vmin=-vmax,
            vmax=vmax,
            base=10
        )

    elif cbar_scale == 'symlin': 
        plot_kwargs["vmin"] = -vmax #if np.nanmin(array_2d) < 0 else min_scale
        plot_kwargs["vmax"] = vmax

    else:  # linear
        plot_kwargs["vmin"] = vmin
        plot_kwargs["vmax"] = vmax

    mesh0 = array_2d.plot(**plot_kwargs)

    # Add features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.set_extent([-180, 180, min_lat, 90], crs=ccrs.PlateCarree())
    set_circular_boundary(ax)
    ax.set_title('', fontsize = 0.1)

    # Colorbar
    if cbar_scale != None:
        cbar0 = fig.colorbar(mesh0, ax=ax, fraction=0.035, pad=0.03)
        cbar0.set_label(bar_label, fontsize = font)

    # plt.savefig('/nobackup/users/hartevel/data/Data_analysis/figures/' + savename, bbox_inches="tight", dpi=300)

    if ax == 'undefined':
        plt.show()


'''
Funcition to select a specific area. Input can be a country name or Ocean or just specifying minimum and 
maximum latitudes and longtitudes. Combinations are also possible, but it is necessary to ensure all lists 
(for areas, min_lat, max_lat ect) are the same size and correctly aligned for the areas. So also if you don't 
need limits for lat and lon, you still need to put them in if looking at more than one area. 

'''
def select_area(data, areas = '', plot=0, min_lat = 0.5, max_lat = 90.5, min_lon = -1, max_lon = 360, mask = True):

    if isinstance(areas, str):
        areas = [areas]
        min_lat = [min_lat]
        max_lat = [max_lat]
        min_lon = [min_lon]
        max_lon = [max_lon]
        mask = [mask]

    if len(areas) != len(min_lat):
        print('Make lists for lats and lons the same length')
            

    regions = regionmask.defined_regions.natural_earth_v5_1_2.countries_110
    
        # --- xarray objects ---
    if isinstance(data, (xr.DataArray, xr.Dataset)):          

        masks = []
            
        for i, area in enumerate(areas):
            mask_slice = ((data.lat < max_lat[i]) & (data.lat > min_lat[i])) & ((data.lon < max_lon[i]) & (data.lon > min_lon[i])) & (mask[i])
            mask_all = regions.mask(data)
            
            if area in regions.names:
                area_id = regions.names.index(area)
                mask_area = (mask_all == area_id) & mask_slice & (mask[i])
                masks.append(mask_area)

            elif area.endswith('Ocean'):
                mask_ocean = mask_all.isnull() & mask_slice & (mask[i])
                masks.append(mask_ocean)

            else:
                masks.append(mask_slice)

        combined_mask = masks[0]
        
        for m in masks[1:]:
            combined_mask |= m


        if plot == 1: 
            # Choose a representative slice ONLY if time exists
            if 'time' in data.dims:
                ref = data.isel(time=0)
            else:
                ref = data

            standard_2d_plot_polar(ref.where(combined_mask))
            # standard_2d_plot(ref.where(combined_mask), max_scale = 0.1)


        return data.where(combined_mask)
    
    # --- dictionaries ---
    elif isinstance(data, dict):
        return {k: select_area(v, areas, plot, min_lat, max_lat, min_lon, max_lon, mask) for k, v in data.items()}

    # --- lists (your variability output) ---
    elif isinstance(data, list):
        return [select_area(v, areas, plot, min_lat, max_lat, min_lon, max_lon, mask) for v in data]

    else:
        print('Data is not a dictionary, list, xarray or xr dataset. Nothing was changed.')
        return data


def plot_bar_graph(data, errors, groups, members, stack_labels, title = '', savename = 'not', y_label = 'Contribution'):

    n_groups = len(groups)
    n_members = len(members)
    bar_width = 0.08
    
    group_x = np.arange(n_groups) * 0.55
    member_offsets = np.arange(n_members) * bar_width * 1.6
    x_positions = [
        group_x[g] + member_offsets[m]
        for g in range(n_groups)
        for m in range(n_members)
    ]
    
    # Colors (shared across all bars)
    colors = plt.cm.tab10.colors[:len(stack_labels)]
    
    fig, ax = plt.subplots(figsize = (4.2,4))
    
    pos_idx = 0
    for g in range(n_groups):
        for m, member in enumerate(members):
            pos_bottom = 0
            neg_bottom = 0

            values = np.array([data[member][g][s] for s in range(len(stack_labels))])
            mask = values > 0
            
            # Get indices of positive stacks
            pos_indices = np.where(mask)[0]
            
            for s, label in enumerate(stack_labels):
                value = data[member][g][s]
                yerror = errors[member][g]
            
                if value >= 0:
                    is_top = (s == pos_indices[-1]) if len(pos_indices) > 0 else False
            
                    ax.bar(
                        x_positions[pos_idx],
                        value,
                        bar_width,
                        bottom=pos_bottom,
                        color=colors[s],
                        label=label if (g == 0 and m == 0) else "",
                        yerr=yerror if is_top else None
                    )
            
                    pos_bottom += value
              
                else:
                    ax.bar(
                        x_positions[pos_idx],
                        value,
                        bar_width,
                        bottom=neg_bottom,
                        color=colors[s],
                        label=label if (g == 0 and m == 0) else ""
                    )
                    neg_bottom += value
    
            pos_idx += 1
    
    # Bottom x-ticks (members)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(members * n_groups, rotation=30, fontsize = 9)
    
    # Top x-ticks (groups)
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(group_x + bar_width / 2)
    ax_top.set_xticklabels(groups)
    ax_top.xaxis.set_ticks_position('bottom')
    ax_top.xaxis.set_label_position('bottom')
    ax_top.spines['bottom'].set_position(('outward', 40))
    
    # Styling

    if y_label == 'Contribution':
        ax.axhline(0, linewidth=1, c = 'black', linestyle = '--')
        ax.axhline(1, linewidth=1, c = 'black', linestyle = '--')
        ax.set_ylabel('Contribution')
    else:
        ax.set_ylabel(y_label)
        
    ax.legend()
    ax.set_title(title)
    plt.tight_layout()
    if savename != 'not':
        plt.savefig(savename , bbox_inches = 'tight')
    plt.show()