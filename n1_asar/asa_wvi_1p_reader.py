import struct
import numpy as np
import xarray as xr
import datatree as dtt

from asa_reader import ASA_WV_Reader

class ASA_WVI_1P_Reader(ASA_WV_Reader):
    """
    Reader class for ASA_WVI_1P product files.

    Args:
        path (str): Path to the ASA_WVI_1P product file. The path can optionally include an index in the format ':WV_XXX', where XXX can range from 000 to 400.

    Attributes:
        file (file object): The file object for the product file.
        mph (dict): Main Product Header.
        sph (dict): Specific Product Header.
        dsd_df (pandas.DataFrame): The Data Set Descriptor DataFrame.
        geolocation_ads_df (pandas.DataFrame): The geolocation ADS DataFrame.
        datatree (dtt.DataTree or None): The DataTree object corresponding to the imagette index, if provided.
        
    Example:
        reader = ASA_WVI_1P_Reader('ASA_WVI_1PNPDK20040728_185756_000000902029_00027_12606_1040.N1:WV_001')
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
        imagette_ds = self.read_imagette_mds(idx)
        cross_spectra_ds = self.read_cross_spectra_mds(idx)
        sq_ads_ds = self.read_sq_ads(idx)
        processing_parameters_ds = self.read_processing_parameters_ads(idx)
        geolocation_ads_ds = self.get_geolocation_ads_from_idx(idx)
        
        final_dict = {
        "/imagette": imagette_ds,
        "/cross_spectra": cross_spectra_ds,
        "/geolocation_ads": geolocation_ads_ds,
        "/sq_ads": sq_ads_ds,
        "/processing_parameters": processing_parameters_ds
        }

        datatree = dtt.DataTree.from_dict(final_dict)
        self.set_attributes(datatree)
        datatree.attrs['platform'] = 'ENVISAT'
        
        return datatree


    def set_attributes(self, datatree):
        """
        Sets attributes on the given DataTree object from SPH (see read_sph) and MPH (see read_mph).

        Args:
            datatree (datatree.DataTree): The Imagette DataTree object to which attributes are added.
        """
        for key in list(self.mph.keys())[:-5]:
            datatree.attrs[key.lower()] = self.mph[key]

        for key in list(self.sph.keys()):
            datatree.attrs[key.lower()] = self.sph[key]

    
    def read_line(self, line, fmt):
        """
        Reads and parses a line of imagette data using a specified format.

        Args:
            line (bytes): The line of binary data to be read.
            fmt (str): The format string for struct.unpack to interpret the binary data.

        Returns:
            tuple: A tuple containing the following elements:
                - datetime.datetime: The converted date from the MJD tuple.
                - int: Quality indicator.
                - int: Range line number.
                - np.ndarray: SAR data (complex numbers).
        """
        data = struct.unpack(fmt, line)
        date = self.convert_mjd(data[0:3])
        quality_indicator = data[3]
        range_line_number = data[4]
        
        sar_data = np.array([complex(data[5:][i], data[5:][i + 1]) for i in range(0, len(data[5:]), 2)])
        return date, quality_indicator, range_line_number, sar_data

    
    def read_imagette_mds(self, idx):
        """
        Reads the SLC Imagette Measurement Data Set (MDS) for the given index. 

        Args:
            idx (int): Imagette index.

        Returns:
            xr.Dataset: An xarray Dataset containing the imagette data.
        """
        row = self.dsd_df[self.dsd_df['DS_NAME'].str.contains('SLC IMAGETTE MDS %03d' % idx)]

        offset = int(row.DS_OFFSET.item()[:-7])
        size = int(row.DS_SIZE.item()[:-7])
        num_dsr = int(row.NUM_DSR.item())
        dsr_size = int(row.DSR_SIZE.item()[:-7])
        
        self.file.seek(offset)
        imagette_mds = self.file.read(size)
        
        header_length = 17
        fmt = '>iii b L {}h'.format((dsr_size-header_length)//2)
        measurement = np.zeros((num_dsr, (dsr_size-header_length)//4), dtype=np.complex128)
    
        dates, quality_indicators, range_lines_numbers = [], [], []
        for i in range(num_dsr):
            line = imagette_mds[i*dsr_size:(i+1)*dsr_size]
            d, qi, rln, sar_data = self.read_line(line, fmt)
            dates.append(d), quality_indicators.append(qi), range_lines_numbers.append(rln)
            measurement[i] = sar_data

        imagette_ds = xr.Dataset(
            data_vars={
                'measurement_real':(['line','sample'], measurement.real,
                                    {'units': '',
                                     'long_name': 'SAR processed data (real part)',
                                     'description': 'Real samples (detected products)' 
                                }),
                'measurement_imag':(['line','sample'], measurement.imag,
                                    {'units': '',
                                     'long_name': 'SAR processed data (imaginary part)',
                                     'description': 'Complex samples (detected products)' 
                                }),
                'zero_doppler_time':(['line'], dates,
                                    {'long_name': 'Zero Doppler Time in azimuth',
                                     'description': '' 
                                }),
                'quality_indicator':(['line'], quality_indicators,
                                    {'units': '',
                                     'long_name': 'Quality Indicator',
                                     'description': ('For non-WSS products, this field is a signed character, where:\n'
                                                     'Set to -1 if all entries in MDSR are zero\n'
                                                     'Set to 0 if the MDSR contains imagery.')
                                }),
        },
            coords={
                'line':(['line'], range_lines_numbers,
                        {'units': '',
                         'long_name': 'Range line number',
                         'description': 'Numbered sequentially, for each product (or slice) first range line in MDS is 1.' 
                        }),
        }
        )
        
        return imagette_ds


    def read_cross_spectra_mds(self, idx):
        """
        Reads the Cross Spectra Measurement Data Set (MDS) for the given index. 

        Args:
            idx (int): Imagette index.

        Returns:
            xr.Dataset: An xarray Dataset containing the cross spectra data. 
        """        
        row = self.dsd_df[self.dsd_df['DS_NAME'].str.contains('CROSS SPECTRA MDS')]

        offset = int(row.DS_OFFSET.item()[:-7])
        size = int(row.DS_SIZE.item()[:-7])
        num_dsr = int(row.NUM_DSR.item())
        dsr_size = int(row.DSR_SIZE.item()[:-7])
        
        self.file.seek(offset + idx*dsr_size)
        raw_cross_spectra_mds = self.file.read(dsr_size)

        data = struct.unpack(">3i ? 30f 64x 432B 432B", raw_cross_spectra_mds)        
    
        real_spectra = np.array(data[34: 466]).reshape((18,24))
        imag_spectra = np.array(data[466:]).reshape((18,24))
        
        cross_spectra_ds = xr.Dataset(
            data_vars={
                'real_spectra': (['direction', 'wavelength'], real_spectra,
                                 {'units': '',
                                 'long_name': 'Real part of cross spectra polar grid.',
                                 'description': ('Number of bins in wavelength and direction defined in SPH (nominally 24 by 36). '
                                                 'However, only 0 to 180 degree of the spectum need to be supplied (24 by 18). '
                                                 'Arranged as: 24 wavelength values for [-5, 5] deg. sector, 24 values for [5, 15] deg. sector, ..., '
                                                 '24 values for [165, 175] deg. sector, in the counter-clockwise direction. '
                                                 'The 24 values for each sector are given in order from longest to shortest wavelength.')
                                 }),
                'imag_spectra': (['direction', 'wavelength'], imag_spectra,
                                 {'units': '',
                                 'long_name': 'Imaginary part of cross spectra polar grid.',
                                 'description': ('Number of bins in wavelength and direction defined in SPH (nominally 24 by 36). '
                                                 'However, only 0 to 180 degree of the spectum need to be supplied (24 by 18). '
                                                 'Arranged as: 24 wavelength values for [-5, 5] deg. sector, 24 values for [5, 15] deg. sector, ..., '
                                                 '24 values for [165, 175] deg. sector, in the counter-clockwise direction. '
                                                 'The 24 values for each sector are given in order from longest to shortest wavelength.')
                                 }),
                'quality_flag': ([], data[3],
                                 {'units': '',
                                 'long_name': 'Complex part of cross spectra polar grid.',
                                 }),
                'range_spectral_res': ([], data[4],
                                       {'units': '',
                                       'long_name': 'Range bin size of the cartesian cross spectrum.'}),
                'az_spectral_res': ([], data[5],
                                    {'units': '',
                                    'long_name': 'Azimuth bin size of the cartesian cross spectrum.'}),
                # 'az_resampling_factor': data[6], exists in product specification but not in product handbook
                'spec_tot_energy': ([], data[7],
                                    {'units': '',
                                    'long_name': 'Spectrum total energy.'}),
                'spec_max_energy': ([], data[8],
                                    {'units': '',
                                    'long_name': 'Spectrum max energy.'}),
                'spec_max_dir': ([], data[9],
                                 {'units': 'deg',
                                 'long_name': 'Direction of spectrum max.',
                                 'description': 'On higher resolution grid. Direction is counter-clockwise from satellite track heading.'}),
                'spec_max_wl': ([], data[10],
                                {'units': 'm',
                                'long_name': 'Wavelength of spectrum max.',
                                'description': 'On higher resolution grid.'}),
                'clutter_noise': ([], data[11],
                                  {'units': '',
                                  'long_name': 'Clutter noise.'}),
                'az_cutoff': ([], data[12],
                              {'units': 'm',
                              'long_name': 'Azimuthal clutter cut-off length.'}),
                'num_iterations': ([], data[13],
                                   {'units': '',
                                   'long_name': 'Number of iterations to compute azimuthal clutter cut-off.'}),
                'range_offset': ([], data[14],
                                 {'units': 'm',
                                 'long_name': 'Range offset of peak of cross covariance function.'}),
                'az_offset': ([], data[15],
                              {'units': 'm',
                              'long_name': 'Azimuth offset of peak of cross covariance function.'}),
                'cc_range_res': ([], data[16],
                                 {'units': 'm',
                                 'long_name': 'Range bin size of cross covariance function.'}),
                'cc_azimuth_res': ([], data[17],
                                   {'units': 'm',
                                   'long_name': 'Azimuth bin size of cross covariance function.'}),
                'sublook_means': (['dim_0'], np.array(data[18:20]).reshape((2,)),
                                  {'units': '',
                                  'long_name': 'First and last sub-look image means.'}),
                'sublook_variance': (['dim_0'], np.array(data[20:22]).reshape((2,)),
                                     {'units': '',
                                     'long_name': 'First and last sub-look image variance.'}),
                'sublook_skewness': (['dim_0'], np.array(data[22:24]).reshape((2,)),
                                     {'units': '',
                                     'long_name': 'First and last sub-look image skewness.'}),
                'sublook_kurtosis': (['dim_0'], np.array(data[24:26]).reshape((2,)),
                                     {'units': '',
                                     'long_name': 'First and last sub-look image kurtosis.'}),
                'range_sublook_detrend_coeff': (['dim_0'], np.array(data[26:28]).reshape((2,)),
                                                {'units': '',
                                                'long_name': 'First and last sub-look de-trend coefficient in range.'}),
                'az_sublook_detrend_coeff': (['dim_0'], np.array(data[28:30]).reshape((2,)),
                                             {'units': '',
                                             'long_name': 'First and last sub-look de-trend coefficient in azimuth.'}),
                'min_imag': ([], data[30],
                             {'units': '',
                             'long_name': 'Min value of imaginary part of cross-spectrum.'}),
                'max_imag': ([], data[31],
                             {'units': '',
                             'long_name': 'Max value of imaginary part of cross-spectrum.'}),
                'min_real': ([], data[32],
                             {'units': '',
                             'long_name': 'Min value of real part of cross-spectrum.'}),
                'max_real': ([], data[33],
                             {'units': '',
                            'long_name': 'Max value of real part of cross-spectrum.'}),
            },
            coords={
                'zero_doppler_time': (['zero_doppler_time'], np.array(self.convert_mjd(data[:3])).reshape((1,)),
                        {
                        'long_name': 'First zero doppler azimuth time of the wave cell.',
                        'description': 'Time of first range line in the SLC imagette MDS described by this data set.'})
            }
        )

        return cross_spectra_ds