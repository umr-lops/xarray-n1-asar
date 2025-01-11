import struct
import numpy as np
import xarray as xr

from n1_asar.asa_wv_reader import ASA_WV_Reader

class ASA_WVW_2P_Reader(ASA_WV_Reader):
    """
    Reader class for ASA_WVW_2P product files.

    Args:
        path (str): Path to the ASA_WVW_2P product file. The path can optionally include an index in the format ':WV_XXX', where XXX can range from 000 to 400.

    Attributes:
        file (file object): The file object for the product file.
        mph (dict): Main Product Header.
        sph (dict): Specific Product Header.
        dsd_df (pandas.DataFrame): The Data Set Descriptor DataFrame.
        geolocation_ads_df (pandas.DataFrame): The geolocation ADS DataFrame.
        datatree (xr.DataTree or None): The DataTree object corresponding to the imagette index, if provided.
        
    Example:
        reader = ASA_WVW_2P_Reader('ASA_WVW_2PPIFR20110110_001927_000000593098_00203_46338_0844.N1:WV_000')
    """ 

    def get_datatree(self, idx):
        """
        Constructs a datatree corresponding to the given imagette index.

        Args:
            idx (int): Imagette index.

        Returns:
            datatree.DataTree: A DataTree object containing the following datasets:
                - /imagette: Imagette dataset.
                - /cross_spectra: Cross spectra dataset.
                - /geolocation_ads: Geolocation ADS dataset.
                - /sq_ads: SQ ADS dataset.
                - /processing_parameters: Processing parameters dataset.
        """
        ocean_wave_spectra_ds = self.read_ocean_wave_spectra_mds(idx)
        sq_ads_ds = self.read_sq_ads(idx)
        processing_parameters_ds = self.read_processing_parameters_ads(idx)
        geolocation_ads_ds = self.get_geolocation_ads_from_idx(idx)
        
        final_dict = {
        "/ocean_wave_spectra": ocean_wave_spectra_ds,
        "/geolocation_ads": geolocation_ads_ds,
        "/sq_ads": sq_ads_ds,
        "/processing_parameters": processing_parameters_ds
        }

        datatree = xr.DataTree.from_dict(final_dict)
        self.set_attributes(datatree)
        datatree.attrs['platform'] = 'ENVISAT'
        
        return datatree

    
    def read_ocean_wave_spectra_mds(self, idx):
        """
        """   
        row = self.dsd_df[self.dsd_df['DS_NAME'].str.contains('OCEAN WAVE SPECTRA MDS')]
    
        offset = int(row.DS_OFFSET.item()[:-7])
        size = int(row.DS_SIZE.item()[:-7])
        num_dsr = int(row.NUM_DSR.item())
        dsr_size = int(row.DSR_SIZE.item()[:-7])
        
        self.file.seek(offset + idx*dsr_size)
        raw_ocean_wave_spectra_mds = self.file.read(dsr_size)
        data = struct.unpack((">3i ? 2f 4x 9f 56x 2f 8x 6f H 3f H 24x 432B 432B"), raw_ocean_wave_spectra_mds)
        real_spectra = np.array(data[28: 460]).reshape((18,24))
        imag_spectra = np.array(data[460:]).reshape((18,24))
        
        ocean_wave_spectra_ds = xr.Dataset(
            data_vars={
                    'quality_flag': ([], data[3], 
                                    {'units': 'flag',
                                     'long_name': 'Quality Indicator',
                                     'description': '-1 for blank MDSR, 0 otherwise'
                                    }),
                    'range_spectral_res': ([], data[4],
                                           {'units': '',
                                           'long_name': 'Range bin size of the cartesian cross spectrum.'}),
                    'az_spectral_res': ([], data[5],
                                        {'units': '',
                                        'long_name': 'Azimuth bin size of the cartesian cross spectrum.'}),
                    'spec_tot_energy': ([], data[6],
                                        {'units': '',
                                        'long_name': 'Spectrum total energy.'}),
                    'spec_max_energy': ([], data[7],
                                        {'units': '',
                                        'long_name': 'Spectrum max energy.'}),
                    'spec_max_dir': ([], data[8],
                                     {'units': 'deg',
                                     'long_name': 'Direction of spectrum max.',
                                     'description': 'Direction is clockwise from north in the direction the wave propagates.'}),
                    'spec_max_wl': ([], data[9],
                                        {'units': 'm',
                                        'long_name': 'Wavelength of spectrum max.',
                                        'description': 'On higher resolution grid.'}),
                    'az_image_shift_var': ([], data[10],
                            {'units': 'm^2',
                            'long_name': 'Variance of the azimuth image shift caused by the orbital velocity.'}),
                    'az_cutoff': ([], data[11],
                            {'units': 'm',
                            'long_name': 'Azimuthal Clutter Cut-off wavelength.'}),
                    'nonlinear_spectral_width': ([], data[12],
                            {'units': 'm',
                            'long_name': 'Spectral width of the non-linear part of the cross spectra.'}),
                    'image_intensity': ([], data[13],
                                {'units': '',
                                'long_name': 'Image intensity'}),
                    'image_variance': ([], data[14],
                                {'units': '',
                                'long_name': 'Normalised image variance'}),
                    'min_spectrum': ([], data[15],
                                {'units': 'm^4',
                                'long_name': 'Min value of ocean wave spectrum.'}),
                    'max_spectrum': ([], data[16],
                                    {'units': 'm^4',
                                    'long_name': 'Max value of ocean wave spectrum.'}),
                    'wind_speed': ([], data[17],
                                    {'units': 'm/s',
                                    'long_name': 'Wind speed used in the wave spectra retrieval.'}),
                    'wind_direction': ([], data[18],
                                        {'units': 'deg',
                                        'long_name': 'Wind direction used in the wave spectra retrieval.',
                                        'description': 'Clockwise from north where the wind from if confidence 0, relative to range otherwise'}),
                    'norm_inv_wave_age': ([], data[19],
                                        {'units': '',
                                        'long_name': 'Normalised inverse wave age.'}),
                    'SAR_wave_height': ([], data[20],
                                        {'units': 'm',
                                        'long_name': 'SAR swell wave height'}),
                    'SAR_az_shift_var': ([], data[21],
                                        {'units': 'm^2',
                                        'long_name': 'Variance of the azimuth shift computed from the SAR swell wave spectra.'}),
                    'backscatter': ([], data[22],
                                            {'units': 'dB',
                                            'long_name': 'Radar backscatter cross section.'}),
                    'confidence_swell': ([], data[23],
                                            {'units': '',
                                            'long_name': 'Confidence measure of the swell inversion.',
                                            'description': ('0 = inversion succesful (a unique spectrum in terms of propagation direction);\n'
                                                            '1 = inversion not succesful (symmetric spectrum).')}),
                    'signal_to_noise': ([], data[24],
                                            {'units': '',
                                            'long_name': 'Average signal-to-noise ratio.'}),
                    'radar_vel_corr': ([], data[25],
                                {'units': 'm/s',
                                'long_name': 'Radar velocity off-set correction'}),
                    'cmod_cal_const': ([], data[26],
                                {'units': '',
                                'long_name': 'Geophysical calibration constant - CMOD'}),
                    'confidence_wind': ([], data[27],
                                {'units': 'Confidence measure of the wind retrieval',
                                'long_name': ('0 = external wind direction used during inversion;'
                                              '1 = external wind direction not used during inversion.')}),
            },
            coords={
                    'zero_doppler_time': (['zero_doppler_time'], np.array(self.convert_mjd(data[:3])).reshape((1,)),
                            {
                            'long_name': 'First zero doppler azimuth time of the wave cell.',
                            'description': 'Time of first range line in the SLC imagette MDS described by this data set.'})
                })
    
        return ocean_wave_spectra_ds