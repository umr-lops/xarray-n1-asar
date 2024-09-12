"""
This reader was developped using the ASAR-Product-Handbook and ENVISAT ASAR products specification documents.
"""

import struct
import numpy as np
import xarray as xr
import pandas as pd
import datatree as dtt

from datetime import datetime, timedelta


class ASA_WV_Reader:
    """
    Abstract class for reading ASA_WV product files.

    Args:
        path (str): Path to the ASA_WV product file. The path can optionally include an index in the format ':WV_XXX', where XXX can range from 000 to 400.

    Attributes:
        file (file object): The file object for the product file.
        mph (dict): Main Product Header.
        sph (dict): Specific Product Header.
        dsd_df (pandas.DataFrame): The Data Set Descriptor DataFrame.
        geolocation_ads_df (pandas.DataFrame): The geolocation ADS DataFrame.
        datatree (dtt.DataTree or None): The DataTree object corresponding to the imagette index, if provided.

    """
    def __init__(self, path):
        
        if ':WV_' in path:
            path, tail = path.rsplit(':WV_', maxsplit=1)
        else:
            path, tail = path, None
    
        self.file = open(path, 'rb')
        
        self.mph = self.read_mph()
        self.sph = self.read_sph()    
        self.dsd_df = self.get_dsd_dataframe()
        self.geolocation_ads_df = self.read_geolocation_ads()

        if tail:
            self.datatree = self.get_datatree(int(tail))

    
    def get_datatree(self, idx):
        pass

        
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

        
    def read_mph(self):
        """
        Reads the Main Product Header (MPH).

        Returns:
            dict: A dictionary containing all the fields available in the MPH.
        """
        self.file.seek(0)
        raw_mph = self.file.read(1247)
        
        mph_dict = {
            'PRODUCT': raw_mph[9:71].decode('utf-8'),
            'PROC_STAGE': raw_mph[84:85].decode('utf-8'),
            'REF_DOC': raw_mph[95:118].decode('utf-8'),
            'ACQUISITION_STATION': raw_mph[182:202].decode('utf-8'),
            'PROC_CENTER': raw_mph[217:223].decode('utf-8'),
            'PROC_TIME': raw_mph[236:263].decode('utf-8'),
            'SOFTWARE_VERSION': raw_mph[279:293].decode('utf-8'),
            'SENSING_START': raw_mph[351:378].decode('utf-8'),
            'SENSING_STOP': raw_mph[394:421].decode('utf-8'),
            'PHASE': raw_mph[470:471].decode('utf-8'),
            'CYCLE': raw_mph[478:482].decode('utf-8'),
            'REL_ORBIT': raw_mph[493:499].decode('utf-8'),
            'ABS_ORBIT': raw_mph[510:516].decode('utf-8'),
            'STATE_VECTOR_TIME': raw_mph[536:563].decode('utf-8'),
            'DELTA_UT1': raw_mph[575:586].decode('utf-8'),
            'X_POSITION': raw_mph[598:613].decode('utf-8'),
            'Y_POSITION': raw_mph[625:640].decode('utf-8'),
            'Z_POSITION': raw_mph[652:667].decode('utf-8'),
            'X_VELOCITY': raw_mph[679:696].decode('utf-8'),
            'Y_VELOCITY': raw_mph[708:725].decode('utf-8'),
            'Z_VELOCITY': raw_mph[737:754].decode('utf-8'),
            'VECTOR_SOURCE': raw_mph[770:772].decode('utf-8'),
            'UTC_SBT_TIME': raw_mph[829:856].decode('utf-8'),
            'SAT_BINARY_TIME': raw_mph[874:885].decode('utf-8'),
            'CLOCK_STEP': raw_mph[897:912].decode('utf-8'),
            'LEAP_UTC': raw_mph[956:983].decode('utf-8'),
            'LEAP_SIGN': raw_mph[995:999].decode('utf-8'),
            'LEAP_ERR': raw_mph[1009:1010].decode('utf-8'),
            'PRODUCT_ERR': raw_mph[1064:1065].decode('utf-8'),
            'TOT_SIZE': raw_mph[1075:1103].decode('utf-8'),
            'SPH_SIZE': raw_mph[1113:1131].decode('utf-8'),
            'NUM_DSD': raw_mph[1140:1151].decode('utf-8'),
            'DSD_SIZE': raw_mph[1161:1179].decode('utf-8'),
            'NUM_DATA_SETS': raw_mph[1194:1205].decode('utf-8')
        }
        
        return mph_dict


    def read_sph(self):
        """
        Reads the Specific Product Header (SPH).

        Returns:
            dict: A dictionary containing all the fields available in the SPH.
        """        
        self.file.seek(1247)
        raw_sph = self.file.read(901)

        sph_dict = {
            'SPH_DESCRIPTOR': raw_sph[16:44].decode('utf-8'),
            'FIRST_CELL_TIME': raw_sph[63:90].decode('utf-8'),
            'LAST_CELL_TIME': raw_sph[108:135].decode('utf-8'),
            'SWATH_1': raw_sph[197:200].decode('utf-8'),
            'SWATH_2': raw_sph[211:214].decode('utf-8'),
            'PASS': raw_sph[222:232].decode('utf-8'),
            'TX_RX_POLAR': raw_sph[247:250].decode('utf-8'),
            'COMPRESSION': raw_sph[265:270].decode('utf-8'),
            'NUM_DIR_BINS': raw_sph[336:340].decode('utf-8'),
            'NUM_WL_BINS': raw_sph[353:357].decode('utf-8'),
            'FIRST_DIR_BIN': raw_sph[372:396].decode('utf-8'),
            'DIR_BIN_STEP': raw_sph[410:434].decode('utf-8'),
            'FIRST_WL_BIN': raw_sph[448:466].decode('utf-8'),
            'LAST_WL_BIN': raw_sph[479:497].decode('utf-8'),
            'LOOK_SEP': raw_sph[558:576].decode('utf-8'),
            'LOOK_BW': raw_sph[585:604].decode('utf-8'),
            'FILTER_ORDER': raw_sph[618:622].decode('utf-8'),
            'TREND_REMOVAL': raw_sph[637:638].decode('utf-8'),
            'ANTENNA_CORR': raw_sph[652:653].decode('utf-8'),
            'SR_GR': raw_sph[660:661].decode('utf-8'),
            'CC_WINDOW': raw_sph[672:673].decode('utf-8'),
            'NUM_LOOK_PAIRS': raw_sph[719:723].decode('utf-8'),
            'CC_RANGE_BINS': raw_sph[738:749].decode('utf-8'),
            'CC_AZIMUTH_BINS': raw_sph[766:777].decode('utf-8'),
            'CC_HALF_WIDTH': raw_sph[792:810].decode('utf-8'),
            'IMAGETTES_FAILED': raw_sph[828:832].decode('utf-8'),
            'SPECTRA_FAILED': raw_sph[848:852].decode('utf-8'),
            'IMAGETTES_MADE': raw_sph[868:872].decode('utf-8'),
            'SPECTRA_MADE': raw_sph[886:890].decode('utf-8'),
        }
        
        return sph_dict

        
    def read_dsd(self, raw_dsd):
        """
        Reads the Data Set Descriptor (DSD) corresponding to the raw data given as input.

        Args:
            raw_dsd (bytes): raw DSD to convert.
            
        Returns:
            dict: A dictionary containing all the fields available in the given DSD.
        """        
        dsd_dict = {
            'DS_NAME': raw_dsd[9:37].decode('utf-8'),
            'DS_TYPE': raw_dsd[47:48].decode('utf-8'),
            'FILENAME': raw_dsd[59:121].decode('utf-8'),
            'DS_OFFSET': raw_dsd[133:161].decode('utf-8'),
            'DS_SIZE': raw_dsd[170:198].decode('utf-8'),
            'NUM_DSR': raw_dsd[207:218].decode('utf-8'),
            'DSR_SIZE': raw_dsd[228:246].decode('utf-8'),
        }
        
        return dsd_dict
    
    
    def get_dsd_dataframe(self):
        """
        Reads all the Data Set Descriptor (DSD) that are present in the given file and set them in a dataframe.
            
        Returns:
            pandas.DataFrame: A dataframe containing all data from the DSD.
        """     
        self.file.seek(2148) # 2148 = 1247 (mph) + 901 (sph)
        
        dsd_size, num_dsd = int(self.mph['DSD_SIZE'][:-7]), int(self.mph['NUM_DSD'])        
        raw_dsd = self.file.read(dsd_size*num_dsd)
        
        dsd_list = []
        for i in range(num_dsd):
            dsd = self.read_dsd(raw_dsd[i*dsd_size:(i+1)*dsd_size])
            dsd_list.append(dsd)

        return pd.DataFrame(dsd_list)


    def get_geolocation_ads_from_idx(self, idx):
        """
        Constructs a dataset containing the geolocation Annotation Data Set corresponding to the given imagette index.

        Args:
            idx (int): Imagette index.

        Returns:
            xarray.Dataset: Dataset containing all the data pertaining to the geolocation ADS associated with the given imagette index.
        """
        row = self.geolocation_ads_df.iloc[idx]

        geolocation_ads_ds = xr.Dataset(
            data_vars={
                'attach_flag': ([], row['attach_flag'], 
                                {'units': 'flag',
                                 'long_name': 'Attachment Flag',
                                 'description': ('Set to 1 if unable to compute the cross spectra for a given SLC imagette '
                                                 '(i.e. no Cross Spectra MDSR corresponding to this ADSR), set to 0 otherwise.')
                                }),
                'center_lat': ([], row['lat'], 
                {'units': 'deg',
                 'long_name': 'Geodetic latitude of center point (positive north)',
                 'description': ('This is the center point of the wave cell. It is calculated after the cross spectra '
                                 'processing, and thus may differ from the center sample latitude of the SLC '
                                 'imagette if slant range to ground range conversion was performed during the '
                                 'cross spectra calculation.')
                }),
                'center_long': ([], row['lon'], 
                {'units': 'deg',
                 'long_name': 'Geodetic longitude of center point (positive east)',
                 'description': ('This is the center point of the wave cell. It is calculated after the cross spectra '
                                 'processing, and thus may differ from the center sample latitude of the SLC '
                                 'imagette if slant range to ground range conversion was performed during the '
                                 'cross spectra calculation.')
                }),
                'heading': ([], row['heading'], 
                {'units': 'deg',
                 'long_name': 'Subsatellite Track Heading (relative to north) of center point',
                 'description': ''
                })
            },
            coords={
                'zero_doppler_time': (['time'],  np.array(row['date']).reshape((1,)),
                         {
                          'long_name': ('First zero doppler azimuth time of the wave cell. '
                                        'Time of first range line in the SLC imagette MDS described by this data set.')
                         })
            }

        )

        return geolocation_ads_ds
    
    
    def read_geolocation_ads(self):
        """
        Reads all the geolocation Annotation Data Set (ADS) that are present in the given file and set them in a dataframe.
            
        Returns:
            pandas.DataFrame: A dataframe containing all data from the geolocation ADS.
        """    
        row = self.dsd_df[self.dsd_df['DS_NAME'].str.contains('GEOLOCATION ADS')]
        
        offset = int(row.DS_OFFSET.item()[:-7])
        size = int(row.DS_SIZE.item()[:-7])
        num_dsr = int(row.NUM_DSR.item())
        dsr_size = int(row.DSR_SIZE.item()[:-7])
        
        self.file.seek(offset)
        raw_geolocation_ads = self.file.read(num_dsr * dsr_size)
        
        geolocation_ads_struct = ">iii ? l l f"
    
        results = []
        for i in range(num_dsr):
            data = struct.unpack(geolocation_ads_struct, raw_geolocation_ads[i*dsr_size:(i+1)*dsr_size])
            results.append({'date': self.convert_mjd(data[:3]),
                            'attach_flag': data[3],
                            'lat': data[4]*1e-6,
                            'lon': data[5]*1e-6,
                            'heading': data[6]})
            
        return pd.DataFrame(results)

    
    def convert_mjd(self, mjd_tuple):
        """
        Converts a Modified Julian Date (MJD) tuple to a datetime object.

        Args:
            mjd_tuple (tuple): A tuple containing three elements:
                - mjd_tuple[0] (int): The number of days since the MJD epoch (2000-01-01).
                - mjd_tuple[1] (int): The number of seconds past the day in mjd_tuple[0].
                - mjd_tuple[2] (int): The number of microseconds past the second in mjd_tuple[1].

        Returns:
            datetime.datetime: A datetime object representing the converted MJD date.
        """
        date = datetime(2000, 1, 1) + timedelta(days=mjd_tuple[0], seconds=mjd_tuple[1], microseconds=mjd_tuple[2])
        return date
        

    def read_sq_ads(self, idx):
        """
        Reads the Summary Quality Annotation Data Set (SQ ADS) for the given index.
        The SQ ADS contains a summary of parameters used to establish the quality of the product.

        Args:
            idx (int): Imagette index.

        Returns:
            xr.Dataset: An xarray Dataset containing the SQ ADS data.
        """
        row = self.dsd_df[self.dsd_df['DS_NAME'].str.contains('SQ ADS')]

        offset = int(row.DS_OFFSET.item()[:-7])
        size = int(row.DS_SIZE.item()[:-7])
        num_dsr = int(row.NUM_DSR.item())
        dsr_size = int(row.DSR_SIZE.item()[:-7])
        
        self.file.seek(offset + idx*dsr_size)
        raw_sq_ads = self.file.read(dsr_size)

        data = struct.unpack(">3i 12? 7x 15f L 15x 10f L 16x 6? 4x 4f L 2f 12x 5f 12x", raw_sq_ads)
        
        sq_ads_ds = xr.Dataset(
            data_vars={
                'attach_flag': ([], data[3], 
                                {'units': 'flag',
                                 'long_name': 'Attachment Flag',
                                 'description': ('Set to 1 if all MDSRs corresponding to this ADSR are blank, set to zero otherwise. '
                                                 'This flag will always be zero because his ADSR is updated once per slice or scene. '
                                                 'Therefore, if there are no MDSRs, this ADSR is not produced at all.')
                                }),
                'input_mean_flag': ([], data[4], 
                                    {'units': 'flag',
                                     'long_name': 'Input data mean outside nominal range flag',
                                     'description': ('0 = mean of I and Q input values are both within specified range from expected mean.\n'
                                                     'For expected mean of x, the measured mean must fall between x-threshold to x+threshold.\n'
                                                     '1 = otherwise.')
                                    }),
                'input_std_dev_flag': ([], data[5], 
                                       {'units': 'flag',
                                        'long_name': 'Input standard deviation outside nominal range flag',
                                        'description': ('0 = mean of I and Q input values are both within specified range of expected standard deviation.\n'
                                                        'For expected std. dev. of x, the measured std. dev. must fall between x-threshold to x+threshold.\n'
                                                        '1 = otherwise.')
                                       }),
                'input_gaps_flag': ([], data[6], 
                                    {'units': 'flag',
                                     'long_name': 'Significant gaps in the input data flag',
                                     'description': ('0 = number of input gaps <= threshold value\n'
                                                     '1 = number of input data gaps > threshold value.')
                                    }),
                'input_missing_lines_flag': ([], data[7], 
                                             {'units': 'flag',
                                              'long_name': 'Missing lines significant flag',
                                              'description': ('0 = percentage of missing lines <= threshold value\n'
                                                              '1 = percentage of missing lines > threshold value\n'
                                                              'The number of missing lines is the number of lines missing from the input data excluding data gaps.')}),
                'dop_cen_flag': ([], data[8], 
                                 {'units': 'flag',
                                  'long_name': 'Doppler centroid uncertain flag',
                                  'description': ('0 = confidence measure >= specified value\n'
                                                  '1 = confidence measure < specified value\n'
                                                  '(note: if more than one Doppler centroid estimation is performed in a slice '
                                                  'the flag is set if any confidence measure is less than the threshold).')
                                 }),
                'dop_amb_flag': ([], data[9], 
                                 {'units': 'flag',
                                  'long_name': 'Doppler ambiguity estimate uncertain flag',
                                  'description': ('0 = confidence measure >= specified value\n'
                                                  '1 = confidence measure < specified value')
                                 }),
                'output_mean_flag': ([], data[10], 
                                     {'units': 'flag',
                                      'long_name': 'Output data mean outside nominal range flag',
                                      'description': ('0 = mean of I and Q output values for SLC image or mean of detected pixels for a detected product, '
                                                      'are both within specified range from expected mean. '
                                                      'For expected mean of x, the measured mean must fall between x-threshold to x+threshold.\n'
                                                      '1 = otherwise')
                                     }),
                'output_std_dev_flag': ([], data[11], 
                                        {'units': 'flag',
                                         'long_name': 'Output data standard deviation outside nominal range flag',
                                         'description': ('0 = std. dev. of I and Q output values for SLC image or std. dev. of detected pixels for a detected product, '
                                                         'are both within specified range from expected std. dev. '
                                                         'For expected std. dev. of x, the measured std. dev. must fall between x-threshold to x+threshold.\n'
                                                         '1 = otherwise')
                                        }),
                'chirp_flag': ([], data[12], 
                               {'units': 'flag',
                                'long_name': 'Chirp reconstruction failed or is of low quality flag',
                                'description': ('0 = able to reconstruct all chirps or chirp reconstruction not requested (nominal chirp used) '
                                                'AND all quality measures were acceptable.\n'
                                                '1 = unable to reconstruct a chirp during processing and chirp reconstruction was requested '
                                                'or the quality is below the acceptable levels.'
                                                'If this is the case PF-ASAR uses the nominal range pulse for processing '
                                                'and a nominal elevation beam scaling factor.')
                                                
                                }),
                'missing_data_sets_flag': ([], data[13], 
                                           {'units': 'flag',
                                            'long_name': 'Data sets missing flag',
                                            'description': ('0 = all data sets which are supposed to be in the product are present\n'
                                                            '1 = any data sets (including ADSs) are missing from the product which '
                                                            'are supposed to be included under normal circumstances. '
                                                            ' Which data sets are missing can be determined by an examination of the DSDs in the SPH.')
                                           }),
                'invalid_downlink_flag': ([], data[14], 
                                          {'units': 'flag',
                                           'long_name': 'Invalid downlink parameters flag',
                                           'description': ('0 = all parameters read from the downlinked data were valid'
                                                           '1 = displayed if any downlink parameter is out of range and '
                                                           'therefore a default value has been used during processing.')
                                          }),
                'thresh_chirp_broadening': ([], data[15], 
                                            {'units': '%',
                                             'long_name': 'Threshold for setting the chirp quality flag',
                                             'description': 'Maximum percentage broadening permitted in cross-correlation pulse width compared to theoretical width.'
                                            }),
                'thresh_chirp_sidelobe': ([], data[16], 
                                          {'units': 'dB',
                                           'long_name': 'Threshold for setting the chirp quality flag',
                                           'description': 'First sidelobe of the chirp cross correlation function.'
                                          }),
                'thresh_chirp_islr': ([], data[17], 
                                      {'units': 'dB',
                                       'long_name': 'Threshold for setting the chirp quality flag',
                                       'description': 'ISLR of the chirp cross correlation function.'
                                      }),
                'thresh_input_mean': ([], data[18], 
                                      {'units': '',
                                       'long_name': 'Threshold for setting the mean of input data quality flag',
                                       'description': 'For an expected mean value of x, this is the value T, such that the measured mean must fall between the x-T and x+T.'
                                    }),
                'exp_input_mean': ([], data[19], 
                                   {'units': '',
                                    'long_name': 'Expected mean input value for this product for both I and Q.'
                                   }),
                'thresh_input_std_dev': ([], data[20], 
                                         {'units': '',
                                          'long_name': 'Threshold for setting the standard deviation of input data quality flag',
                                          'description': ('For an expected std. dev. value of y, this is the value D, '
                                                          'such that the measured std. dev. must fall between the y-D and y+D.')
                                         }),
                'exp_input_std_dev': ([], data[21], 
                                      {'units': '',
                                       'long_name': 'Expected input std. dev. for this product for both I and Q.'
                                      }),
                'thresh_dop_cen': ([], data[22], 
                                   {'units': '',
                                    'long_name': 'Threshold for setting the Doppler Centroid quality flag',
                                    'description': 'Threshold for Doppler Centroid confidence'
                                   }),
                'thresh_dop_amb': ([], data[23], 
                                   {'units': '',
                                    'long_name': 'Threshold for setting the Doppler Centroid ambiguity quality flag',
                                    'description': 'Threshold for setting the Doppler Centroid ambiguity confidence flag'
                                   }),
                'thresh_output_mean': ([], data[24], 
                                       {'units': '',
                                        'long_name': 'Threshold for setting the mean of output data quality flag',
                                        'description': ('For an expected mean value of x, this is the value T, '
                                                        'such that the measured mean must fall between the x-T and x+T.')
                                       }),
                'exp_output_mean': ([], data[25], 
                                    {'units': '',
                                     'long_name': 'Expected mean output value for this product',
                                     'description': 'For an SLC product this is the expected mean of both the I and Q values.'
                                    }),
                'thresh_output_std_dev': ([], data[26], 
                                          {'units': '',
                                           'long_name': 'Threshold for setting the standard deviation of output data quality flag',
                                           'description': ('For an expected std. dev. value of y, this is the value D, '
                                                           'such that the measured std. dev. must fall between the y-D and y+D.')
                                          }),
                'exp_output_std_dev': ([], data[27], 
                                       {'units': '',
                                        'long_name': 'Expected output std. dev. for this product',
                                        'description': ' For an SLC product this is the expected output std. dev. for both I and Q values'
                                       }),
                'thresh_input_missing_lines': ([], data[28], 
                                               {'units': '%',
                                                'long_name': 'Threshold for setting the missing lines quality flag',
                                                'description': 'Maximum percentage of missing lines to total lines.'
                                               }),
                'thresh_input_gaps': ([], data[29], 
                                      {'units': '',
                                       'long_name': 'Threshold for setting the missing gaps quality flag',
                                       'description': 'Maximum number of missing gaps allowed.'}),
                'lines_per_gap': ([], data[30], 
                                  {'units': 'lines',
                                   'long_name': 'Number of missing lines which constitute a gap',
                                   'description': ''
                                  }),
                'input_mean': (['dim_0'], np.array(data[31:33]).reshape((2,)), 
                               {'units': '',
                                'long_name': 'Input data mean',
                                'description': '(i channel, then q channel)'
                               }),
                'input_std_dev': (['dim_0'], np.array(data[33:35]).reshape((2,)), 
                                  {'units': '',
                                   'long_name': 'Input data standard deviation ',
                                   'description': '(i channel, then q channel)'
                                  }),
                'num_gaps': ([], data[35], 
                             {'units': '',
                              'long_name': 'Number of gaps',
                              'description': '(composed of a predetermined number of consecutive missing lines)'
                             }),
                'num_missing_lines': ([], data[36], 
                                      {'units': '',
                                       'long_name': 'Number of missing lines',
                                       'description': '(excluding gaps)'
                                      }),
                'output_mean': (['dim_0'], np.array(data[36:38]).reshape((2,)), 
                                {'units': '',
                                 'long_name': 'Output data mean',
                                 'description': ('For SLC products, first value is for the i channel, '
                                                 'second is for the q channel. For detected products, '
                                                 'second value is set to zero')
                                }),
                'output_std_dev': (['dim_0'], np.array(data[38:40]).reshape((2,)), 
                                   {'units': '',
                                    'long_name': 'Output data standard deviation',
                                    'description': ('For SLC products, first value is for the i channel, '
                                                    'second is for the q channel. For detected products, ' 
                                                    'second value is set to zero')
                                   }),
                'tot_errors': ([], data[41], 
                               {'units': '',
                                'long_name': 'Total number of errors detected in ISP headers',
                                'description': ''
                               }),
                'land_flag': ([], data[42], 
                              {'units': 'flag',
                               'long_name': 'Land Flag',
                               'description': '0 = no land in imagette \n1 = land in imagette'
                              }),
                'look_conf_flag': ([], data[43], 
                                   {'units': 'flag',
                                    'long_name': 'Look image statistics confidence parameter flag',
                                    'description': ('1 = The ratio of the standard deviation to the mean of the first look image '
                                                    'is outside the range given by a minimum and a maximum threshold.\n'
                                                    '0 = otherwise')
                                   }),
                'inter_look_conf_flag': ([], data[44], 
                                         {'units': 'flag',
                                          'long_name': 'Inter-look confidence statistics confidence parameter flag',
                                          'description': ('1 = The normalised deviation of the two inter-look sub-images is greater '
                                                          'than a maximum threshold.\n'
                                                          '0 = otherwise')
                                         }),
                'az_cutoff_flag': ([], data[45], 
                                   {'units': 'flag',
                                    'long_name': 'Azimuth cut-off convergence measure flag',
                                    'description': ('1 = The normalised RMS error between the fitted co-variance profile is '
                                                    'greater than a maximum threshold.\n'
                                                    '0 = otherwise')
                                   }),
                'az_cutoff_iteration_flag': ([], data[46], 
                                             {'units': 'flag',
                                              'long_name': 'Azimuth cut-off Iteration count overflow flag',
                                              'description': ('1 = The Azimuth cut-off fit did not converge within a minimum number of iterations.\n'
                                                              '0 = otherwise')
                                             }),
                'phase_flag': ([], data[47], 
                               {'units': 'flag',
                                'long_name': 'Phase information confidence measure flag',
                                'description': ('1 = The imaginary spectral peak is less than a minimum threshold, '
                                                'or the zero lag shift is greater than a minimum threshold.'
                                                '0 = otherwise')
                               }),
                'look_conf_thresh': (['dim_0'], np.array(data[48:50]).reshape((2,)), 
                                     {'units': '',
                                      'long_name': 'Look image statistics confidence parameter thresholds (minimum and maximum)',
                                      'description': ''}),
                'inter_look_conf_thresh': ([], data[50], 
                                           {'units': '',
                                            'long_name': 'Inter-look confidence statistics confidence parameter threshold',
                                            'description': ''}),
                'az_cutoff_thresh': ([], data[51], 
                                     {'units': '',
                                      'long_name': 'Azimuth cut-off convergence measure threshold',
                                      'description': ''}),
                'az_cutoff_iteration_thresh': ([], data[52], 
                                               {'units': '',
                                                'long_name': 'Azimuth cut-off Iteration count overflow threshold',
                                                'description': ''}),
                'phase_peak_thresh': ([], data[53], 
                                      {'units': '',
                                       'long_name': 'Phase information confidence measure threshold for the spectral peak',
                                       'description': ''}),
                'phase_cross_thresh': ([], data[54], 
                                       {'units': '',
                                        'long_name': 'Phase information confidence measure threshold for cross covariance peak offset',
                                        'description': ''}),
                'look_conf': ([], data[55], 
                              {'units': '',
                               'long_name': 'Look image statistics confidence parameter',
                               'description': 'The ratio of the standard deviation to the mean of the first look image.'}),
                'inter_look_conf': ([], data[56], 
                                    {'units': '',
                                     'long_name': 'Inter-look confidence statistics confidence parameter',
                                     'description': 'The normalised deviation of the two inter-look sub-images.'}),
                'az_cutoff': ([], data[57], 
                              {'units': '',
                               'long_name': 'Azimuth cut-off convergence measure',
                               'description': 'The normalised RMS error between the fitted co-variance profile.'}),
                'phase_peak_conf': ([], data[58], 
                                    {'units': '',
                                     'long_name': 'Phase information confidence measure for the spectral peak',
                                     'description': 'The imaginary spectral peak.'}),
                'phase_cross_conf': ([], data[59], 
                                     {'units': '',
                                      'long_name': 'Phase information confidence measure for cross covariance peak offset',
                                      'description': ''})
            },
            coords={
                'time': (['time'], np.array(self.convert_mjd(data[:3])).reshape((1,)),
                         {
                          'description': 'First zero doppler azimuth time of the wave cell. Time of first range line in the SLC imagette MDS described by this data set.'
                         })
            }
        )
        
        return sq_ads_ds


    def read_processing_parameters_ads(self, idx):
        """
        Reads the Processing Parameters Annotation Data Set (ADS) for the given index. 
        
        Args:
            idx (int): Imagette index.

        Returns:
            xr.Dataset: An xarray Dataset containing processing parameters.
        """   
        row = self.dsd_df[self.dsd_df['DS_NAME'].str.contains('PROCESSING PARAMS ADS')]

        offset = int(row.DS_OFFSET.item()[:-7])
        size = int(row.DS_SIZE.item()[:-7])
        num_dsr = int(row.NUM_DSR.item())
        dsr_size = int(row.DSR_SIZE.item()[:-7])
        
        self.file.seek(offset + idx*dsr_size)
        raw_processing_parameters_ads = self.file.read(dsr_size)
        data = struct.unpack((">3i ? 3i 12c f 3c 3f 2L 5c L f 43x 15? 6x 4L 14f 4? 4f 92x 32x 2L 3i 20x 60h 60x"
                              "10L 26x 10f 5L 40f 5h 5f 5L 62x L 3f h 7c f 10f 8f 128x 60x L h 2f 7c 6f 68x 2f 8x"
                              "5f 5L 64x 12x 4f 16x f 48x 28c 64x 4L 4f 5L 3i 16x 3i 6l 3i 6l 3i 6l 3i 6l 3i 6l"
                              "64x 7f c 13x 6f c f 7c 4x 352f 16x 3i 3L 6f 6l 3i L 3L 6f 6l 3i L 3L 6f 6l 12f H 3f 12x 33f 14x"),
        raw_processing_parameters_ads)
        
        processing_parameters_dict = {
            "/": xr.Dataset({
                'first_zero_doppler_time': ([], self.convert_mjd(data[:3]),
                                            {'long_name': 'First Zero Doppler Azimuth time of MDS which this data set describes', 
                                             'description': 'Time of first range line in the MDS described by this data set.'
                                            }),
                'attach_flag': ([], data[3],
                                {'units': 'flag',
                                 'long_name': 'Attachment flag', 
                                 'description': 'Always set to zero for this ADSR.'}),
                'last_zero_doppler_time': ([],self.convert_mjd(data[4:7]),
                                           {'long_name': 'Last Zero Doppler Azimuth time of MDS which this data set describes', 
                                           'description': 'Time of last range line in the MDS described by this data set.'
                                           }),
                'work_order_id': ([], b''.join(data[7:19]).decode('utf-8'),
                                  {'units': '', 
                                   'long_name': 'Work Order ID',
                                   'description': 'Left-justified.'
                                  }),
                'time_diff': ([], data[19],
                              {'units': 's',
                               'long_name': ('Time difference between sensing time of first input line and '
                                             'zero Doppler time of first output image line (tdelta).'),
                               'description': ''
                              }),
                'swath_num': ([], b''.join(data[20:23]).decode('utf-8'),
                              {'units': '',
                               'long_name': 'Swath number',
                               'description': 'IS1, IS2, IS3, IS4, IS5, IS6 or IS7.'
                              }),
                'range_spacing': ([], data[23],
                                  {'units': 'm',
                                   'long_name': 'Range sample spacing',
                                   'description': ''
                                  }),
                'azimuth_spacing':([], data[24],
                                  {'units': 'm',
                                  'long_name': 'Azimuth sample spacing at image center',
                                   'description': ''
                                  }),
                'line_time_interval':([], data[25],
                                      {'units': 's',
                                       'long_name': 'Azimuth sample spacing in time (Line Time Interval) ',
                                       'description': ''
                                      }),
                'num_output_lines':([], data[26],
                                    {'units': 'lines',
                                     'long_name': 'Number of output range lines in the image described by this ADSR',
                                     'description': 'For WSS products, this number will vary for each sub swath.'
                                    }),
                'num_samples_per_line':([], data[27],
                                        {'units': 'samples',
                                         'long_name': ('Number of samples per output range line (includes zero filled samples) '
                                                       'in the image described by this ADSR'),
                                        'description': ''
                                        }),
                'data_type': ([], b''.join(data[28:33]).decode('utf-8'),
                              {'units': '',
                               'long_name': 'Output data type',
                               'description': 'SWORD, UWORD or UBYTE.'
                              }),
                'num_range_lines_per_burst': ([], data[33],
                                              {'units': 'lines',
                                               'long_name': 'Number of output range lines per burst',
                                               'description': 'Not used for single-beam products.'
                                              }),
                'time_diff_zero_doppler': ([], data[34],
                                           {'units': '',
                                            'long_name': 'Time difference between zero Doppler time and acquisition time of output image lines',
                                            'description': ''
                                           }),
                'data_analysis_flag': ([], data[35],
                       {'units': 'flag',
                        'long_name': 'Raw Data Analysis used for Raw Data Correction',
                        'description': ('0 = correction done using default parameters\n'
                                        '1 = correction done using raw data analysis results.')
                       }),
                'ant_elev_corr_flag': ([], data[36],
                                       {'units': 'flag',
                                        'long_name': 'Antenna Elevation Pattern Correction Applied',
                                        'description': ('0 = no correction applied\n'
                                                        '1 = correction applied')
                                       }),
                'chirp_extract_flag': ([], data[37],
                                       {'units': '',
                                        'long_name': 'Reconstructed Chirp to be used (if reconstruction successful)',
                                        'description': ('0 = nominal chirp replica to be used\n'
                                                        '1 = reconstructed chirp to be used')
                                       }),
                'srgr_flag': ([], data[38],
                              {'units': 'flag',
                               'long_name': 'Slant Range to Ground Range Conversion Applied',
                               'description': ('0 = no conversion applied\n'
                                               '1 = conversion applied')
                              }),
                'dop_cen_flag': ([], data[39],
                                 {'units': 'flag',
                                  'long_name': 'Doppler Centroid Estimation Performed',
                                  'description': ('0 = no estimation done'
                                                  '1 = estimation done')
                                 }),
                'dop_amb_flag': ([], data[40],
                                 {'units': 'flag',
                                  'long_name': 'Doppler Ambiguity Estimation Performed',
                                  'description': ('0 = no estimate done\n'
                                                  '1 = estimate done')
                                 }),
                'range_spread_comp_flag': ([], data[41],
                                           {'units': 'flag',
                                            'long_name': 'Range-spreading loss compensation Applied',
                                            'description': ('0 = no compensation applied\n'
                                                            '1 = compensation applied')
                                           }),
                'detected_flag': ([], data[42],
                                  {'units': 'flag',
                                   'long_name': 'Detection Applied',
                                   'description': ('0 = output product is complex'
                                                   '1 = output product was detected')
                                  }),
                'look_sum_flag': ([], data[43],
                                  {'units': 'flag',
                                   'long_name': 'Look Summation Performed',
                                   'description': ('0 = product is single look\n'
                                                   '1 = product is multi-looked')
                                  }),
                'rms_equal_flag': ([], data[44],
                                   {'units': 'flag',
                                    'long_name': 'RMS Equalization Performed',
                                    'description': ('0 = RMS equalization was not performed during FBAQ decoding\n'
                                                    '1 = RMS equalization was performed during FBAQ decoding')
                                   }),
                'ant_scal_flag': ([], data[45],
                                  {'units': '',
                                   'long_name': 'Antenna Elevation Gain Scaling Factor Applied',
                                   'description': ('0 = no scaling factor applied\n'
                                                   '1 = scaling factor applied')
                                  }),
                'vga_com_echo_flag': ([], data[46],
                                      {'units': '',
                                       'long_name': 'Receive Gain Droop Compensation Applied to Echo Data',
                                       'description': ('0 = no compensation applied\n'
                                                       '1 = compensation applied')
                                      }),
                'vga_com_cal_flag': ([], data[47],
                                     {'units': 'flag',
                                      'long_name': 'Receive Gain Droop Compensation Applied to Calibration Pulse P2',
                                      'description': ('0 = no compensation applied\n'
                                                      '1 = compensation applied')
                                     }),
                'vga_com_nom_time_flag': ([], data[48],
                                          {'units': 'flag',
                                           'long_name': 'Receive Gain Droop Compensation for Calibration Pulse P2 Order Zero: Nominal Time Delay Applied',
                                           'description': ('0 = do not use nominal time delay '
                                                           '(compensation depends on P2 time delay with respect to the end of the echo window)\n'
                                                           '1 = use nominal time delay (compensation is constant)')
                                          }),
                'gm_range_comp_inverse_filter_flag': ([], data[49],
                                                      {'units': 'flag',
                                                       'long_name': 'Inverse Filter used for range compression (GM Mode only)',
                                                       'description': ('0 = matched filter used for range compression\n'
                                                                       '1 = inverse filter used for range compression')
                                                      }),
                
                'first_proc_range_samp': ([], data[221],
                                          {'units': 'samples',
                                           'long_name': 'First processed input range sample, first sample is 1',
                                           'description': ''
                                          }),
                'range_ref': ([], data[222],
                              {'units': 'm',
                               'long_name': 'Range spreading loss reference range',
                               'description': ''
                              }),
                'range_samp_rate': ([], data[223],
                                    {'units': 'Hz',
                                     'long_name': 'Range sampling rate',
                                     'description': ''
                                    }),
                'radar_freq': ([], data[224],
                               {'units': 'Hz',
                                'long_name': 'Radar Frequency',
                                'description': ''
                               }),
                'num_looks_range': ([], data[225],
                                    {'units': 'looks',
                                     'long_name': 'Number of range looks',
                                     'description': ''
                                    }),
                'filter_range': ([], b''.join(data[226:233]).decode('utf-8'),
                                 {'units': '',
                                  'long_name': 'Matched filter window type',
                                  'description': 'HAMMING or KAISER  or NONE   '
                                 }),
                'filter_coef_range': ([], data[233],
                                      {'units': '',
                                       'long_name': 'Window coefficient for range-matched filter',
                                       'description': ''
                                      }),

                
                'num_lines_proc': ([], data[252],
                                   {'units': 'lines',
                                    'long_name': 'Number of input lines processed',
                                    'description': ''
                                   }),
                'num_look_az': ([], data[253],
                                {'units': 'looks',
                                 'long_name': 'Number of Azimuth Looks',
                                 'description': ''
                                }),
                'look_bw_az': ([], data[254],
                               {'units': 'Hz',
                                'long_name': 'Azimuth Look Bandwidth',
                                'description': '(null to null)'
                               }),
                'to_bw_az': ([], data[255],
                             {'units': 'Hz',
                              'long_name': 'Processed Azimuth bandwidth',
                              'description': '(null to null)'
                             }),
                'filter_az': ([], b''.join(data[256:263]).decode('utf-8'),
                              {'units': '',
                               'long_name': 'Matched filter window type:',
                               'description': 'HAMMING or KAISER  or NONE   '
                              }),
                'filter_coef_az': ([], data[263],
                                   {'units': '',
                                    'long_name': 'Window coefficient for azimuth-matched filter',
                                    'description': ''
                                   }),
                'az_fm_rate': (['dim_0'], np.array(data[264:267]),
                               {'units': 'Hz/s, Hz/s2, Hz/s3',
                                'long_name': '3 co-efficients for Azimuth FM rate',
                                'description': ('Azimuth FM rate = C_0 + C_1(t_SR-t_0) + C2(t_SR - t_0)^2\n'
                                                't_SR = 2 way slant range time')
                               }),
                'ax_fm_origin': ([], data[267],
                                 {'units': '',
                                  'long_name': '2 way slant range time origin (t_0) for Azimuth FM rate calculation',
                                  'description': 'ns'
                                 }),
                'dop_amb_conf': ([], data[268],
                                 {'units': '',
                                  'long_name': 'Doppler Centroid Ambiguity Confidence Measure',
                                  'description': 'Value between 0 and 1, 0 = poorest confidence, 1 = highest confidence'
                                 }),

                'avg_scene_height_ellipsoid': ([], data[285],
                                               {'units': 'm',
                                                'long_name': 'Average scene height above ellipsoid used for processing',
                                                'description': ''
                                               }),
                'echo_comp': ([], b''.join(data[286:290]).decode('utf-8'),
                              {'units': '',
                               'long_name': 'Compression Method used for echo samples',
                               'description': 'FBAQ, S&M , NONE'
                              }),
                'echo_comp_ratio': ([], b''.join(data[290:293]).decode('utf-8'),
                                    {'units': '',
                                     'long_name': 'Compression Ratio for echo samples',
                                     'description': '8/4, 8/3, 8/2, or 8/8'
                                    }),
                'init_cal_comp': ([], b''.join(data[293:297]).decode('utf-8'),
                                  {'units': '',
                                   'long_name': 'Compression Method used for initial calibration samples',
                                   'description': 'FBAQ, S&M , NONE'
                                  }),
                'init_cal_ratio': ([], b''.join(data[297:300]).decode('utf-8'),
                                   {'units': '',
                                    'long_name': 'Compression Ratio for initial calibration samples',
                                    'description': '8/4, 8/3, 8/2, or 8/8'
                                   }),
                'per_cal_comp': ([], b''.join(data[300:304]).decode('utf-8'),
                                 {'units': '',
                                  'long_name': 'Compression Method used for periodic calibration samples',
                                  'description': 'FBAQ, S&M , NONE'
                                 }),
                'per_cal_ratio': ([], b''.join(data[304:307]).decode('utf-8'),
                                  {'units': '',
                                   'long_name': 'Compression Ratio for periodic calibration samples',
                                   'description': '8/4, 8/3, 8/2, or 8/8'
                                  }),
                'noise_comp': ([], b''.join(data[307:311]).decode('utf-8'),
                               {'units': '',
                                'long_name': 'Compression Method used for noise samples',
                                'description': 'FBAQ, S&M , NONE'
                               }),
                'noise_comp_ratio': ([], b''.join(data[311:314]).decode('utf-8'),
                                     {'units': '',
                                      'long_name': 'Compression Ratio for noise samples',
                                      'description': '8/4, 8/3, 8/2, or 8/8'
                                     }),
                'beam_overlap': (['dim_1'], np.array(data[314:318]),
                                 {'units': '',
                                  'long_name': 'Number of slant range samples in beam merging',
                                  'description': ('One value per merge region (1-2, 2-3, 3-4, 4-5).\n'
                                                  'This parameter is equivalent to N in the following beam merging formula: \n'
                                                  'xmerged(n) = (1 - (n/N)^P * xnear(n) + ((n/N)^P * xfar(n)\n'
                                                  'These fields are set to zero for single beam and WSS products.')
                                 }),
                'beam_param': (['dim_1'], np.array(data[318:322]),
                               {'units': '',
                                'long_name': 'Beam merge algorithm parameter used for beam merging',
                                'description': ('One value per merge region (1-2, 2-3, 3-4, 4-5).\n'
                                                'This parameter is equivalent to P in the above beam merging formula, '
                                                'and different values have the following affect: \n'
                                                'P = 1, linear weighting of the two beams (near and far) '
                                                'P = -1, (which represents infinity in the beam merging formula) '
                                                'only near beam contributes to the merged one\n'
                                                'P = 0, only far beam contributes to the merged one\n'
                                                'P > 1, near beam is favoured\n'
                                                '0 < P < 1, far beam is favoured\n'
                                                'These fields are set to zero for single beam and WSS products.')
                               }),
                'lines_per_burst': (['dim_2'], np.array(data[322:327]),
                                    {'units': 'lines',
                                     'long_name': 'Number of raw data lines per burst for this image',
                                     'description': '5 values for beams SS1 to SS5 in WS and GM modes. Two values for AP mode, all others set to zero.'
                                    }),
                'time_first_SS1_echo': ([], self.convert_mjd(data[327:330]),
                                        {'long_name': 'Time of first SS1 Echo Source Packet',
                                         'description': ''
                                        }),
        
                'slant_range_time': ([], data[375],
                                     {'units': 'ns',
                                      'long_name': '2-way slant range time origin (t_0)',
                                      'description': ''
                                     }),
                'dop_coef': (['dim_2'], np.array(data[376:381]),
                             {'units': 'Hz, Hz/s, Hz/s2, Hz/s3, Hz/s4',
                              'long_name': 'Doppler centroid coefficients as a function of slant range time: D0, D1, D2, D3, and D4.',
                              'description': 'Where Doppler Centroid = D0 + D1(tSR-t0) + D2(tSR-t0)^2 + D3(tSR-t0)^3 + D4(tSR-t0)^4'
                             }),
                'dop_conf': ([], data[381],
                             {'units': '',
                              'long_name': 'Doppler Centroid Confidence Measure',
                              'description': 'Value between 0 and 1, 0 = poorest confidence, 1 = highest confidence.'
                             }),
                'dop_conf_below_thresh': ([], data[382],
                                          {'units': '',
                                           'long_name': 'Doppler Confidence Below Threshold Flag',
                                           'description': ('0 = confidence above threshold, Doppler Centroid calculated from data\n'
                                                           '1 = confidence below threshold, Doppler Centroid calculated from orbit parameters')
                                          }),
                'chirp_width': ([], data[383],
                                {'units': 'samples',
                                 'long_name': '3-dB pulse width of chirp replica cross-correlation function between reconstructed chirp and nominal chirp',
                                 'description': ''
                                }),
                'chirp_sidelobe': ([], data[384],
                                   {'units': 'dB',
                                    'long_name': '',
                                    'description': ''
                                   }),
                'chirp_islr': ([], data[385],
                                {'units': 'dB',
                                 'long_name': 'ISLR of chirp replica cross-correlation function between reconstructed chirp and nominal chirp',
                                 'description': ''
                                }),
                'chirp_peak_loc': ([], data[386],
                                {'units': 'samples',
                                 'long_name': 'Peak location of cross-correlation function between reconstructed chirp and nominal chirp',
                                 'description': ''
                                }),
                'chirp_power': ([], data[387],
                                {'units': 'dB',
                                 'long_name': 'Reconstructed chirp power',
                                 'description': ''
                                }),
                'eq_chirp_power': ([], data[388],
                                   {'units': 'dB',
                                    'long_name': 'Equivalent chirp power',
                                    'description': ''
                                   }),
                'rec_chirp_exceeds_qua_thres': ([], data[389],
                                {'units': '',
                                 'long_name': 'Reconstructed chirp exceeds quality thresholds',
                                 'description': ('0 = reconstructed chirp does not meet quality thresholds, chirp is invalid\n'
                                                 '1 = reconstructed chirp does meet quality thresholds')
                                }),
                'ref_chirp_power': ([], data[390],
                                {'units': 'dB',
                                 'long_name': 'Reference chirp power',
                                 'description': ''
                                }),
                'norm_source': ([], b''.join(data[391:398]).decode('utf-8'),
                                {'units': '',
                                 'long_name': 'Normalisation source',
                                 'description': 'REPLICA or REF     EQV     or NONE    (if normalisation not applied)'
                                }),
                
                'swst_offset': ([], data[806],
                                {'units': 'ns',
                                 'long_name': 'Wave cell SWST offset',
                                 'description': 'From center of the sub-swath to start of imagette. 208 ns increments.'
                                }),
                'ground_range_bias': ([], data[807],
                                {'units': 'km',
                                 'long_name': 'Wave cell Ground range bias',
                                 'description': 'From centre of the Sub-Swath to the centre of the imagette.'
                                }),
                'elev_angle_bias': ([], data[808],
                                {'units': 'deg',
                                 'long_name': 'Wave cell Elevation angle bias',
                                 'description': 'From centre of the Sub-Swath elevation to the centre of the imagette.'
                                }),
                'imagette_range_len': ([], data[809],
                                {'units': 'm',
                                 'long_name': 'Imagette length in range',
                                 'description': ''
                                }),
                'imagette_az_len': ([], data[810],
                                {'units': 'm',
                                 'long_name': 'Imagette length in azimuth',
                                 'description': ''
                                }),
                'imagette_range_res': ([], data[811],
                                {'units': 'm',
                                 'long_name': 'Nominal Imagette resolution in slant range',
                                 'description': ''
                                }),
                'ground_res': ([], data[812],
                                {'units': 'm',
                                 'long_name': 'Nominal resolution in ground range',
                                 'description': ''
                                }),
                'imagette_az_res': ([], data[813],
                                {'units': 'm',
                                 'long_name': 'Nominal Imagette resolution in azimuth',
                                 'description': ''
                                }),
                'platform_alt': ([], data[814],
                                {'units': 'm',
                                 'long_name': 'Altitude (platform to ellipsoid) in metres',
                                 'description': 'Centre of wave cell.'
                                }),
                'ground_vel': ([], data[815],
                                {'units': 'm/s',
                                 'long_name': 'Ground Velocity',
                                 'description': 'w.r.t moving earth.'
                                }),
                'slant_range': ([], data[816],
                                {'units': 'm',
                                 'long_name': 'Range to centre of imagette',
                                 'description': 'From platform to target.'
                                }),
                'cw_drift': ([], data[817],
                                {'units': '',
                                 'long_name': 'CW signal drift',
                                 'description': ''
                                }),
                'wave_subcycle': ([], data[818],
                                {'units': '',
                                 'long_name': 'Wave sub-cycle',
                                 'description': '(1 or 2) of this wave cell '
                                }),
                'earth_radius': ([], data[819],
                                {'units': 'm',
                                 'long_name': 'Earth Radius at imagette center sample ',
                                 'description': ''
                                }),
                'sat_height': ([], data[820],
                                {'units': 'm',
                                 'long_name': 'Satellite distance to earth center',
                                 'description': ''
                                }),
                'first_sample_slant_range': ([], data[821],
                                {'units': 'm',
                                 'long_name': 'Distance from satellite to first range pixel in the full SLC image',
                                 'description': ''
                                }),
            }),
        
            "/raw_data_analysis_mds": xr.Dataset({
                'num_gaps': ([], data[50],
                             {'units': 'gaps',
                              'long_name': 'Number of input data gaps',
                              'description': 'A gap is defined as a predetermined number of range lines.'
                             }),
                'num_missing_lines': ([], data[51],
                                      {'units': 'lines',
                                       'long_name': 'Number of missing lines, excluding data gaps ',
                                       'description': ''
                                      }),
                'range_skamp_skip': ([], data[52],
                                     {'units': 'samples',
                                      'long_name': 'Range sample skipping factor for raw data analysis',
                                      'description': ''
                                     }),
                'ranges_lines_skip': ([], data[53],
                                      {'units': 'lines',
                                       'long_name': 'Range lines skipping factor for raw data analysis',
                                       'description': ''
                                      }),
                'calc_i_bias': ([], data[54],
                                {'units': '',
                                 'long_name': 'Calculated I channel bias',
                                 'description': ''
                                }),
                'calc_q_bias': ([], data[55],
                                {'units': '',
                                 'long_name': 'Calculated Q channel bias',
                                 'description': ''
                                }),
                'calc_i_std_dev': ([], data[56],
                                   {'units': '',
                                    'long_name': 'Calculated I channel standard deviation',
                                    'description': ''
                                   }),
                'calc_q_std_dev': ([], data[57],
                                   {'units': '',
                                    'long_name': 'Calculated Q channel standard deviation',
                                    'description': ''
                                   }),
                'calc_gain': ([], data[58],
                              {'units': '',
                               'long_name': 'Calculated I/Q gain imbalance ',
                               'description': ''
                              }),
                'calc_quad': ([], data[59],
                              {'units': '',
                               'long_name': 'Calculated I/Q quadrature departure',
                               'description': ''
                              }),
                'i_bias_max': ([], data[60],
                               {'units': '',
                                'long_name': 'I bias upper bound',
                                'description': ''
                               }),
                'i_bias_min': ([], data[61],
                               {'units': '',
                                'long_name': 'I bias lower bound',
                                'description': ''
                               }),
                'q_bias_max': ([], data[62],
                               {'units': '',
                                'long_name': 'Q bias upper bound',
                                'description': ''
                               }),
                'q_bias_min': ([], data[63],
                               {'units': '',
                                'long_name': 'Q bias lower bound ',
                                'description': ''
                               }),
                'gain_min': ([], data[64],
                             {'units': '',
                              'long_name': 'I/Q gain lower bound',
                              'description': ''
                             }),
                'gain_max': ([], data[65],
                             {'units': '',
                              'long_name': 'I/Q gain upper bound',
                              'description': ''
                             }),
                'quad_min': ([], data[66],
                             {'units': '',
                              'long_name': 'I/Q quadrature departure lower bound ',
                              'description': ''
                             }),
                'quad_max': ([], data[67],
                             {'units': '',
                              'long_name': 'I/Q quadrature departure upper bound',
                              'description': ''
                             }),
                'i_bias_flag': ([], data[68],
                                {'units': 'flag',
                                 'long_name': 'I bias significance',
                                 'description': ('0 = I bias falls within acceptable range\n'
                                                 '1 = I bias falls outside acceptable range')
                                }),
                'q_bias_flag': ([], data[69],
                                {'units': 'flag',
                                 'long_name': 'Q bias Significance',
                                 'description': ''
                                }),
                'gain_flag': ([], data[70],
                              {'units': '',
                               'long_name': 'I/Q Gain Significance',
                               'description': ('0 = Gain falls within acceptable range\n'
                                               '1 = Gain falls outside acceptable range')
                              }),
                'quad_flag': ([], data[71],
                              {'units': '',
                               'long_name': 'I/Q Quadrature Departure Significance',
                               'description': ('0 = Quadrature departure falls within acceptable range\n'
                                               '1 =Quadrature departure falls outside acceptable range')
                              }),
                'used_i_bias': ([], data[72],
                                {'units': '',
                                 'long_name': 'I channel bias used for correction',
                                 'description': 'May be different from measured value.'
                                }),
                'used_q_bias': ([], data[73],
                                {'units': '',
                                 'long_name': 'Q channel bias used for correction',
                                 'description': 'May be different from measured value.'
                                }),
                'used_gain': ([], data[74],
                              {'units': '',
                               'long_name': 'I/Q gain imbalance used for correction',
                               'description': 'May be different from measured value.'
                              }),
                'used_quad': ([], data[75],
                              {'units': '',
                               'long_name': 'I/Q quadrature departure used for correction',
                               'description': 'May be different from measured value.'
                              })
            }),

            
        
            "/start_time": xr.Dataset({
                'first_obt': (['dim_0'], np.array(data[76:78]),
                              {'units': '',
                               'long_name': 'On-board time of first input line processed',
                               'description': 'LSB accurate to 15.26 s. (Contained in two long integers).'
                              }),
                'first_mjd': ([], self.convert_mjd(data[78:81]),
                              {'long_name': 'Sensing time of first input line processed converted from satellite binary time',
                               'description': ''
                              }),
            }),
        
            "/parameters_code": xr.Dataset({
                'swst_code': ([], data[81:86][0],
                              {'units': '',
                               'long_name': 'Sampling Window Start time code of first processed line',
                               'description': ''
                              }),
                'last_swst_code': ([], data[86:91][0],
                                  {'units': '',
                                   'long_name': 'Sampling Window Start time code of last processed line ',
                                   'description': ''
                                  }),
                'pri_code': ([], data[91:96][0],
                             {'units': '',
                              'long_name': 'Pulse Repetition Interval code ',
                              'description': ''
                             }),
                'tx_pulse_len_code': ([], data[96:101][0],
                                      {'units': '',
                                       'long_name': 'Tx pulse length',
                                       'description': ''
                                      }),
                'tx_bw_code': ([], data[101:106][0],
                               {'units': '',
                                'long_name': 'Tx pulse bandwidth ',
                                'description': ''
                               }),
                'echo_win_len_code': ([], data[106:111][0],
                                      {'units': '',
                                       'long_name': 'Echo Window Length ',
                                       'description': ''
                                      }),
                'up_code': ([], data[111:116][0],
                            {'units': '',
                             'long_name': 'Upconverter Level',
                             'description': 'Upconverter gain set on the instrument'
                            }),
                'down_code': ([], data[116:121][0],
                              {'units': '',
                               'long_name': 'Downconverter Level',
                               'description': 'Downconverter gain set on the instrument'
                              }),
                'resamp_code': ([], data[121:126][0],
                                {'units': '',
                                 'long_name': 'Resampling factor for echo data',
                                 'description': ''
                                }),
                'beam_adj_code': ([], data[126:131][0],
                                  {'units': '',
                                   'long_name': 'Beam adjustment delta',
                                   'description': ''
                                  }),
                'beam_set_num_code': ([], data[131:136][0],
                                      {'units': '',
                                       'long_name': 'Antenna Beam Set Number',
                                       'description': ''
                                      }),
                'tx_monitor_code': ([], data[136:141][0],
                                    {'units': '',
                                     'long_name': 'Auxiliary Tx Monitor Level',
                                     'description': ''
                                    }),
                            }),
        
            "/error_counters": xr.Dataset({
                    'num_err_swst': ([], data[141],
                                     {'units': '',
                                      'long_name': 'Number of errors detected in Sampling Window start time field',
                                      'description': ''
                                     }),
                    'num_err_pri': ([], data[142],
                                    {'units': '',
                                     'long_name': 'Number of errors detected in PRI code field',
                                     'description': ''
                                    }),
                    'num_err_tx_pulse_len': ([], data[143],
                                             {'units': '',
                                              'long_name': 'Number of errors detected in Tx pulse length field',
                                              'description': ''
                                             }),
                    'num_err_tx_pulse_bw': ([], data[144],
                                            {'units': '',
                                             'long_name': 'Number of errors detected in Tx pulse bandwidth field',
                                             'description': ''
                                            }),
                    'num_err_echo_win_len': ([], data[145],
                                             {'units': '',
                                              'long_name': 'Number of errors detected in Echo Window Length field',
                                              'description': ''
                                             }),
                    'num_err_up': ([], data[146],
                                   {'units': '',
                                    'long_name': 'Number of errors detected in Upconverter Level field',
                                    'description': ''
                                   }),
                    'num_err_down': ([], data[147],
                                     {'units': '',
                                      'long_name': 'Number of errors detected in Downconverter Level field',
                                      'description': ''
                                     }),
                    'num_err_resamp': ([], data[148],
                                       {'units': '',
                                        'long_name': 'Number of errors detected in Resampling factor for echo data field',
                                        'description': ''
                                       }),
                    'num_err_beam_adj': ([], data[149],
                                         {'units': '',
                                          'long_name': 'Number of errors detected in Beam adjustment delta field',
                                          'description': ''
                                         }),
                    'num_err_beam_set_num': ([], data[150],
                                             {'units': '',
                                              'long_name': 'Number of errors detected in Antenna Beam Set Number field',
                                              'description': ''
                                             })
                }),
                                          
        "/image_parameters": xr.Dataset({
                'swst_value': ([], data[151:156][0],
                               {'units': 's',
                                'long_name': 'Sampling Window Start time of first processed line',
                                'description': ''
                               }),
                'last_swst_value': ([], data[156:161][0],
                                    {'units': 's',
                                     'long_name': 'Sampling Window Start time of last processed line',
                                     'description': ''
                                    }),
                'swst_changes': ([], data[161:166][0],
                                 {'units': '',
                                  'long_name': 'Number of Sample Window Start Time changes within a beam',
                                  'description': ''
                                 }),
                'prf_value': ([], data[166:171][0],
                              {'units': 'Hz',
                               'long_name': 'Pulse Repetition Frequency ',
                               'description': ''
                              }),
                'tx_pulse_len_value': ([], data[171:176][0],
                                       {'units': 'Hz',
                                        'long_name': 'Tx pulse length',
                                        'description': ''
                                       }),
                'tx_pulse_bw_value': ([], data[176:181][0],
                                      {'units': 'Hz',
                                       'long_name': 'Tx pulse bandwidth',
                                       'description': ''
                                      }),
                'echo_win_len_value': ([], data[181:186][0],
                                       {'units': 's',
                                        'long_name': 'Echo Window Length',
                                        'description': ''
                                       }),
                'up_value': ([], data[186:191][0],
                             {'units': 'dB',
                              'long_name': 'Upconverter Level',
                              'description': 'Upconverter gain set on the instrument.'
                             }),
                'down_value': ([], data[191:196][0],
                               {'units': 'dB',
                                'long_name': 'Downconverter Level',
                                'description': 'Downconverter gain set on the instrument.'
                               }),
                'resamp_value': ([], data[196:201][0],
                                {'units': '',
                                 'long_name': 'Resampling factor',
                                 'description': ''
                                }),
                'beam_adj_value': ([], data[201:206][0],
                                  {'units': 'deg',
                                   'long_name': 'Beam adjustment delta',
                                   'description': ''
                                  }),
                'beam_set_value': ([], data[206:211][0],
                                   {'units': '',
                                    'long_name': 'Antenna Beam Set Number',
                                    'description': ''
                                   }),
                'tx_monitor_value': ([], data[211:216][0],
                                     {'units': '',
                                      'long_name': 'Auxiliary Tx Monitor Level',
                                      'description': ''
                                     }),
                'rank': ([], data[216:221][0],
                         {'units': '',
                          'long_name': 'Rank',
                          'description': 'The number of PRI between transmitted pulse and return echo.'
                         })
            }),
            
            "/bandwidth": xr.Dataset({
                'look_bw_range': ([], data[234:239][0],
                                  {'units': 'Hz',
                                   'long_name': 'Range Look Bandwidth (null to null)',
                                   'description': ''
                                  }),
                'tot_bw_range': ([], data[239:244][0],
                                 {'units': 'Hz',
                                  'long_name': 'Total processed range bandwidth (null to null)',
                                  'description': ''
                                 })
            }),
            
            "/nominal_chirp": xr.Dataset({
                'norm_chirp_amp': (['dim_0'], np.array(data[244:248]),
                                   {'units': '-, s-1, s-2, s-3',
                                    'long_name': '4 nominal chirp amplitude coefficients',
                                    'description': ''
                                   }),
                'norm_chirp_phs': (['dim_0'], np.array(data[248:252]),
                                   {'units': 'cycles, Hz, Hz/s, Hz/s2',
                                    'long_name': '4 nominal chirp phase coefficients',
                                    'description': ''
                                   })
            }),
            
            "/calibration_factors": xr.Dataset({
                'proc_scaling_fact': ([], data[269],
                                      {'units': '',
                                       'long_name': 'Processor scaling factor',
                                       'description': 'Factor units are linear when using the Range/Doppler algorithm, dB when Specan is used.'
                                      }),
                'ext_scal_fact': ([], data[270],
                                  {'units': '',
                                   'long_name': 'External Calibration Scaling Factor',
                                   'description': 'Mode/swath/polarization dependent.'
                                  })
            }),
            
            "/noise_estimation": xr.Dataset({
                'noise_power_corr': ([], data[271:276][0],
                                     {'units': '',
                                      'long_name': 'Noise power correction factor',
                                      'description': ''
                                     }),
                'num_noise_lines': ([], data[276:281][0],
                                    {'units': '',
                                     'long_name': 'Number of noise lines used to calculate correction factors',
                                     'description': ''
                                    })
            }),
            
            "/output_statistics": xr.Dataset({
                'out_mean': ([], data[281],
                             {'units': '',
                              'long_name': 'Output data mean',
                              'description': 'Magnitude for detected products, real sample mean for SLC products.'
                             }),
                'out_imag_mean': ([], data[282],
                                  {'units': '',
                                   'long_name': 'Output imaginary data mean',
                                   'description': 'Used for SLC products only (set to zero otherwise).'
                                  }),
                'out_std_dev': ([], data[283],
                                {'units': '',
                                 'long_name': 'Output data standard deviation',
                                 'description': 'Magnitude std. dev. for detected products, real sample std. dev. for SLC products.'
                                }),
                'out_imag_std_dev': ([], data[284],
                                     {'units': '',
                                      'long_name': 'Output imaginary data standard deviation',
                                      'description': 'Used for SLC products only (set to zero otherwise).'
                                     })
            }),

            "/orbit_state_vectors": xr.Dataset(
                data_vars={
                'x_pos': (['state_vect_time'], np.array([data[i] for i in range(333, 375, 9)]) * 1e-2,
                            {'units': 'm',
                             'long_name': 'X position in Earth fixed reference frame',
                             'description': ''
                            }),
                'y_pos': (['state_vect_time'], np.array([data[i] for i in range(334, 375, 9)]) * 1e-2,
                            {'units': 'm',
                             'long_name': 'Y position in Earth fixed reference frame',
                             'description': ''
                            }),
                'z_pos': (['state_vect_time'], np.array([data[i] for i in range(335, 375, 9)]) * 1e-2,
                            {'units': 'm',
                             'long_name': 'Z position in Earth fixed reference frame',
                             'description': ''
                            }),
                'x_vel': (['state_vect_time'], np.array([data[i] for i in range(336, 375, 9)]) * 1e-5,
                            {'units': 'm/s',
                             'long_name': 'X velocity relative to Earth fixed reference frame',
                             'description': ''
                            }),
                'y_vel': (['state_vect_time'], np.array([data[i] for i in range(337, 375, 9)]) * 1e-2,
                            {'units': 'm/s',
                             'long_name': 'Y velocity relative to Earth fixed reference frame',
                             'description': ''
                            }),
                'z_vel': (['state_vect_time'], np.array([data[i] for i in range(338, 375, 9)]) * 1e-2,
                            {'units': 'm/s',
                             'long_name': 'Z velocity relative to Earth fixed reference frame',
                             'description': ''
                            }),
                },
                coords={'state_vect_time': (['state_vect_time'], np.array([self.convert_mjd(data[i:i+3]) for i in range(330, 375, 9)]),
                                            {'long_name': 'Time of state vector', 
                                             'description': ''}),
                       }),
        
            "/cal_info": xr.Dataset({
                'max_cal': (['dim_0', 'dim_1'], np.array([data[i:i+3] for i in range(398, 750, 11)]),
                            {'units': 'deg',
                             'long_name': 'Max of cal pulses 1, 2, and 3 amplitude',
                             'description': ''
                            }),
                'avg_cal': (['dim_0', 'dim_1'], np.array([data[i:i+3] for i in range(401, 750, 11)]),
                            {'units': '',
                             'long_name': ('Average of Cal pulse 1, 2, and 3 amplitude '
                                           'above the predetermined threshold relative to the max amplitude'),
                             'description': 'Nominal threshold is 0.707 of max amplitude.'
                            }),
                'avg_val_1a': (['dim_0'], np.array([data[i] for i in range(404, 750, 11)]),
                            {'units': '',
                             'long_name': 'Average of Cal pulse 1A over the sample window',
                             'description': ''
                            }),
                'phs_cal': (['dim_0', 'dim_2'], np.array([data[i:i+4] for i in range(405, 750, 11)]),
                            {'units': 'm/s',
                             'long_name': 'Extracted phase for calibration pulse 1, 1A, 2, and 3',
                             'description': ''
                            }),
            
            }),
        
            "/first_line_tie_points": xr.Dataset(
                data_vars={
                'range_samp_nums': (['dim_0'], np.array(data[753:756]),
                                    {'units': '',
                                     'long_name': 'Range sample number',
                                     'description': ('Gives the range location of the grid points. '
                                                     'First range sample is 1, last is M (includes zero filled samples)')
                            }),
                'slant_range_times': (['dim_0'], np.array(data[756:759]),
                                    {'units': 'ns',
                                     'long_name': '2 way slant range time to range sample',
                                     'description': ''
                            }),
                'inc_angles': (['dim_0'], np.array(data[759:762]),
                                    {'units': '',
                                     'long_name': 'Incidence Angle at range sample',
                                     'description': ''
                            }),
                'lats': (['dim_0'], np.array(data[762:765])*1e-6,
                                    {'units': '',
                                     'long_name': 'Geodetic latitude of range sample (positive north)',
                                     'description': ''
                            }),
                'longs': (['dim_0'], np.array(data[765:768])*1e-6,
                                    {'units': '',
                                     'long_name': 'Geodetic longitude of range sample (positive east)',
                                     'description': ''
                            }),
            },
                coords={
                'first_line_time': (['first_line_time'], np.array(self.convert_mjd(data[750:753])).reshape((1,)),
                                    {'long_name': 'Zero Doppler Time at first line of imagette'}),
                }),
            
            "/mid_line_tie_points": xr.Dataset(
                data_vars={
                'range_samp_nums': (['dim_0'], np.array(data[772:775]),
                                    {'units': '',
                                     'long_name': 'Range sample number',
                                     'description': ('Gives the range location of the grid points. '
                                                     'First range sample is 1, last is M (includes zero filled samples)')
                            }),
                'slant_range_times': (['dim_0'], np.array(data[775:778]),
                                    {'units': 'ns',
                                     'long_name': '2 way slant range time to range sample',
                                     'description': ''
                            }),
                'inc_angles': (['dim_0'], np.array(data[778:781]),
                                    {'units': '',
                                     'long_name': 'Incidence Angle at range sample',
                                     'description': ''
                            }),
                'lats': (['dim_0'], np.array(data[781:784])*1e-6,
                                    {'units': '',
                                     'long_name': 'Geodetic latitude of range sample (positive north)',
                                     'description': ''
                            }),
                'longs': (['dim_0'], np.array(data[784:787])*1e-6,
                                    {'units': '',
                                     'long_name': 'Geodetic longitude of range sample (positive east)',
                                     'description': ''
                            }),
            },
                coords={'mid_line_time': (['mid_line_time'], np.array(self.convert_mjd(data[768:771])).reshape((1,)),
                                          {'long_name': 'Zero Doppler Time at centre line of imagette'}),
                        'mid_range_line_nums': data[771]
                       }),
        
            "/last_line_tie_points": xr.Dataset(
                data_vars={
                'range_samp_nums': (['dim_0'], np.array(data[791:794]),
                                    {'units': '',
                                     'long_name': 'Range sample number',
                                     'description': ('Gives the range location of the grid points. '
                                                     'First range sample is 1, last is M (includes zero filled samples)')
                            }),
                'slant_range_times': (['dim_0'], np.array(data[794:797]),
                                    {'units': 'ns',
                                     'long_name': '2 way slant range time to range sample',
                                     'description': ''
                            }),
                'inc_angles': (['dim_0'], np.array(data[797:800]),
                                    {'units': '',
                                     'long_name': 'Incidence Angle at range sample',
                                     'description': ''
                            }),
                'lats': (['dim_0'], np.array(data[800:803])*1e-6,
                                    {'units': '',
                                     'long_name': 'Geodetic latitude of range sample (positive north)',
                                     'description': ''
                            }),
                'longs': (['dim_0'], np.array(data[803:806])*1e-6,
                                    {'units': '',
                                     'long_name': 'Geodetic longitude of range sample (positive east)',
                                     'description': ''
                            }),
                },
                coords={'last_line_time': (['last_line_time'], np.array(self.convert_mjd(data[787:790])).reshape((1,)),
                                          {'long_name': 'Zero Doppler Time at last line of imagette'}),
                        'last_range_line_nums': data[790]},
            ),
        
            "/elevation_pattern": xr.Dataset({
                'slant_range_time': (['dim_0'], np.array(data[822:833]),
                                    {'units': 'ns',
                                     'long_name': '2 way slant range times',
                                     'description': ''
                            }),
                'elevation_angles': (['dim_0'], np.array(data[833:844]),
                                    {'units': 'deg',
                                     'long_name': 'Corresponding elevation angles',
                                     'description': ''
                            }),
                'antenna_pattern': (['dim_0'], np.array(data[844:855]),
                                    {'units': 'dB',
                                     'long_name': 'Corresponding two-way antenna elevation pattern values ',
                                     'description': ''
                })
            })
        }
        processing_parameters_ds = dtt.DataTree.from_dict(processing_parameters_dict)
        
        return processing_parameters_ds