import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from matplotlib.colors import ListedColormap

def display_cross_spectra(asa_wvi_1p_dt, kmin=2*np.pi/1000, kmax=0.07, savepath=None):
    """
    Displays cross spectra (real and imaginary) from a given dataset.

    Args:
        asa_wvi_1p_dt (datatree.DataTree): datatree containing the cross spectra data obtained using the ASA_WVI_1P_Reader class. The given datatree must correspond to a specific imagette.
        kmin (float): minimum wavenumber to be plotted.
        kmax (float): maximum wavenumber to be plotted.
        savepath (str, defaults to None): path where to save the figure. If None is given, the figure is displayed and not saved.

    Returns:
        None
    """
    asa_cross_spectra = asa_wvi_1p_dt['cross_spectra']
    asa_cross_spectra = asa_cross_spectra.assign_coords(direction=('direction', np.radians(np.arange(0, 180, 10))))
    
    # get first and last wavelength values
    l0 = float(asa_wvi_1p_dt.attrs['first_wl_bin'][:-3])
    lm = float(asa_wvi_1p_dt.attrs['last_wl_bin'][:-3])
    N = int(asa_wvi_1p_dt.attrs['num_wl_bins'])
    wavelength = np.array([l0 / ((l0 / lm) ** (2 * m / (2*N-1))) for m in range(N-1)] + [lm])
    str_wavelength = np.array([f'{w:.1f}' for w in wavelength])
    
    asa_cross_spectra = asa_cross_spectra.assign_coords(wavenumber=('wavelength', 2*np.pi/wavelength))
    
    # rescale the real_spectra and imag_spectra
    real_spectra = asa_cross_spectra['real_spectra'].T / 255 * (asa_cross_spectra['max_real'] - asa_cross_spectra['min_real']) + asa_cross_spectra['min_real']
    imag_spectra = asa_cross_spectra['imag_spectra'].T / 255 * (asa_cross_spectra['max_imag'] - asa_cross_spectra['min_imag']) + asa_cross_spectra['min_imag']
    spectra = real_spectra + 1j*imag_spectra
    
    # concatenate half spectra and its conjugate
    sym_spectra = spectra.copy()
    sym_spectra = np.conj(sym_spectra.assign_coords(direction = np.pi + spectra.direction))
    full_spectra = xr.concat([spectra, sym_spectra], dim='direction')
    
    # create a figure with two polar subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': 'polar'})
    
    # create a brighter plasma colormap for real spectra
    plasma_bright = ListedColormap(np.clip(plt.cm.plasma(np.linspace(0.15, 1, 256)), 0.0, 1.0))
    
    # plot real spectra on the first subplot
    c0 = full_spectra.real.plot(ax=axes[0], cmap=plasma_bright, y='wavenumber',
                                vmin=full_spectra.real.min(), vmax=full_spectra.real.max(),
                                add_colorbar=False)
    axes[0].set_title("Real Spectra")
    axes[0].set(xlabel=None, ylabel=None)
    axes[0].set_theta_zero_location('N')  # set 0° to be at the top
    axes[0].set_xticks(np.arange(0, 2*np.pi, np.pi/4))
    axes[0].set_rticks([2*np.pi/w for w in [400,200,100]], ['400m','200m','100m'])
    axes[0].set_rlim(kmin, kmax)
    axes[0].grid(linestyle='--')
    fig.colorbar(c0, ax=axes[0], pad=0.1)
    
    # plot imaginary spectra on the second subplot
    c1 = full_spectra.imag.plot(ax=axes[1], cmap='PuOr', y='wavenumber',
                                vmin=full_spectra.imag.min(), vmax=full_spectra.imag.max(),
                                add_colorbar=False)
    axes[1].set_title("Imaginary Spectra")
    axes[1].set(xlabel=None, ylabel=None)
    axes[1].set_theta_zero_location('N')  # set 0° to be at the top
    axes[1].set_xticks(np.arange(0, 2*np.pi, np.pi/4))
    axes[1].set_rticks([2*np.pi/w for w in [400,200,100]], ['400m','200m','100m'])
    axes[1].set_rlim(kmin, kmax)
    axes[1].grid(linestyle='--')
    fig.colorbar(c1, ax=axes[1],  pad=0.1)

    if savepath:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()


def display_digital_numbers(asa_wvi_1p_dt, savepath=None):
    """
    Displays digital numbers from a given dataset.

    Args:
        asa_wvi_1p_dt (datatree.DataTree): datatree containing the imagette data obtained using the ASA_WVI_1P_Reader class. The given datatree must correspond to a specific imagette.
        savepath (str, defaults to None): path where to save the figure. If None is given, the figure is displayed and not saved.

    Returns:
        None
    """
    DN = np.log10(np.abs(asa_wvi_1p_dt['imagette']['measurement_real'] + 1j*asa_wvi_1p_dt['imagette']['measurement_imag'])**2)
    DN = DN.assign_coords(range_ = DN['sample']*asa_wvi_1p_dt['processing_parameters'].ds['ground_res'])
    DN = DN.assign_coords(azimuth_ = DN['line']*asa_wvi_1p_dt['processing_parameters'].ds['imagette_az_res'])

    c = DN.plot(cmap='gray', vmin=0, x='range_',y='azimuth_', add_colorbar=False)
    plt.colorbar(c, label=r'$log_{10}(|DN|^2)$')
    plt.xlabel('ground range (m)'), plt.ylabel('azimuth (m)')
    plt.axis('scaled')

    if savepath:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()


def display_map(reader, i, dx=10, savepath=None):
    """
    Displays a map with two subplots: a global view and a zoomed-in view around a specified point of interest.

    Args:
        reader (ASA_WVI_1P_Reader): result of the reader class of ASA_WVI_1P data.
        i (int): Index of the considered imagette.
        dx (float): delta of degrees to display around the imagette center for the zoom.
        savepath (str, defaults to None): path where to save the figure. If None is given, the figure is displayed and not saved.

    Returns:
        None
    """
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), subplot_kw={'projection': proj}, width_ratios=(3, 1))

    lon, lat = reader.geolocation_ads_df['lon'].iloc[i], reader.geolocation_ads_df['lat'].iloc[i]
    
    axes[0].coastlines(resolution='10m')
    axes[0].stock_img()
    axes[0].scatter(reader.geolocation_ads_df['lon'], reader.geolocation_ads_df['lat'], marker='.', color='orange', s=3)
    axes[0].scatter(lon, lat, marker='x', color='r', s=100)
    axes[0].set_xlim(-180, 180), axes[0].set_ylim(-90, 90)
    axes[0].gridlines(draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    
    axes[1].coastlines(resolution='10m')
    axes[1].stock_img()
    axes[1].scatter(reader.geolocation_ads_df['lon'], reader.geolocation_ads_df['lat'], marker='.', color='orange', s=5)
    axes[1].scatter(lon, lat, marker='x', color='r', s=100)
    axes[1].set_xlim(lon-dx, lon+dx), axes[1].set_ylim(lat-dx, lat+dx)
    axes[1].gridlines(draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    
    plt.suptitle(f"{reader.mph['PRODUCT']} - Imagette n°{i} \nDate: {reader.geolocation_ads_df['date'].iloc[i]}", y=0.9)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()