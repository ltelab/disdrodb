#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script used to assemble the data from a parsivel device into a parquet file
Once merged all the files and converted to .parquet, it create a README.txt and a log<datetime>.txt file in /<campaign_name>/processed
If there are more than one sub-folder inside the data folder (the folder where are saved all the parsivel data files),
it will create a parquet file for every folder, and it will call <folder_name>_<campaign_name>_Processed_data.parquet inside /<campaign_name>/processed folder 
    

    Attributes
    ----------
    campaign_name: str
        the name of the parsival data collection campaing
    campaign_path: path
        path location of the parent folder's campaign
    file_list : list
        a list of all the single data (.dat) inside the campaign path
    header_column_list : 
        the header of the table on the .dat files (25 columns)        
    dtype_set : set
        rappresent the datatype of the dataframe's columns
    chunk_size : int
        the chunk size use in read_csv process
    skip_files = dictionary
        store skipped files for folder during files reading
    file_list : list
        list files to read
    folder_file_list : dictionary
        list files and folder to read
    __temp_list : list
        temporary list to merge the rows of the data files
    __temp_skip_files : int
        counter for skipped file during data files reading
    __temp_skip_files_list : list
        store skipped file error during files reading
    

    Methods
    -------
    create_parquet
        Merge all the data files found in the path folder into a parquet file inside a folder called 'processed', or into 'processed'/<subfolder> if there are subfolders
    __getDataFiles()
        Populate the _temp_list with parsivel data files in the path set on campaign_path
    __merge_files_data()
        Convert all the data files in self.__getDataFiles() into a Dask dataframe, drop all the Nan rows and append in every dataframe read into self.__temp_list
    __concat_frames()
        Concat the dataframes inside self.__temp_list
    __save_parquet()
        Save a dask dataframe into a parquet file inside a folder called 'processed', or 'processed'/<subfolder>/Processed_data_<campaing_name>_<subfolder>.parquet
    __check_processed_folder()
        Check if exist /<campaign_name>/processed folder, if not will create it and also the subfolder if need it
    __getDataFiles()
        Populate the file_list with parsivel data files in the path set on campaign_path, and populate also the subfolder into folder_file_list if need it
    __createREADME()
        Create README.txt in processed folder to store campaign info
    __createLog()
        Create lof.txt in processed folder to store conversion info and errors
"""

class Campaign:
    '''
    A class used to assemble in a parquet file a collection of data from a Parsivel device
    
    Attributes
    ----------
    campaign_name: str
        the name of the parsival data collection campaing
    campaign_path: path
        path location of the parent folder's campaign
    file_list : list
        a list of all the single data (.dat) inside the campaign path
    header_column_list : 
        the header of the table on the .dat files (25 columns)        
    dtype_set : set
        rappresent the datatype of the dataframe's columns
    chunk_size : int
        the chunk size use in read_csv process
    skip_files = dictionary
        store skipped files for folder during files reading
    file_list : list
        list files to read
    folder_file_list : dictionary
        list files and folder to read
    __temp_list : list
        temporary list to merge the rows of the data files
    __temp_skip_files : int
        counter for skipped file during data files reading
    __temp_skip_files_list : list
        store skipped file error during files reading
    

    Methods
    -------
    create_parquet
        Merge all the data files found in the path folder into a parquet file inside a folder called 'processed', or into 'processed'/<subfolder> if there are subfolders
    __getDataFiles()
        Populate the _temp_list with parsivel data files in the path set on campaign_path
    __merge_files_data()
        Convert all the data files in self.__getDataFiles() into a Dask dataframe, drop all the Nan rows and append in every dataframe read into self.__temp_list
    __concat_frames()
        Concat the dataframes inside self.__temp_list
    __save_parquet()
        Save a dask dataframe into a parquet file inside a folder called 'processed', or 'processed'/<subfolder>/Processed_data_<campaing_name>_<subfolder>.parquet
    __check_processed_folder()
        Check if exist /<campaign_name>/processed folder, if not will create it and also the subfolder if need it
    __getDataFiles()
        Populate the file_list with parsivel data files in the path set on campaign_path, and populate also the subfolder into folder_file_list if need it
    __createREADME()
        Create README.txt in processed folder to store campaign info
    __createLog()
        Create lof.txt in processed folder to store conversion info and errors
    
    
    '''
    
    __temp_list = []
    
    __temp_skip_files = 0
        
    __temp_skip_files_list = []
    
    skip_files = {}
    
    file_list = []
    
    folder_file_list = {}
    
    header_column_list = ['ID',
                          'Geo_coordinate_x',
                          'Geo_coordinate_y',
                          'Timestamp',
                          'Datalogger_temp',
                          'Datalogger_power',
                          'Datalogger_communication',
                          'Rain_intensity',
                          'Rain_amount_accumulated',
                          'Weather_code_SYNOP_according_table_4680',
                          'Weather_code_SYNOP_according_table_4677',
                          'Radar_reflectivity',
                          'MOR_visibility_precipitation',
                          'Signal_amplitude_laser_strip',
                          'Number_detected_particles',
                          'Temperature_sensor',
                          'Current_through_heating_system',
                          'Power_supply_voltage',
                          'Sensor_status',
                          'Rain_amount_absolute',
                          'Error_code',
                          'FieldN',
                          'FieldV',
                          'RAW_data',
                          'Unknow_column'
                          ]

    dtype_set= {'ID': 'int16',
                'Geo_coordinate_x': 'object',
                'Geo_coordinate_y': 'object',
                'Timestamp': 'datetime64[ns]',
                'Datalogger_temp': 'object',
                'Datalogger_power': 'object',
                'Datalogger_communication': 'uint8',
                'Rain_intensity': 'float',
                'Rain_amount_accumulated': 'float',
                'Weather_code_SYNOP_according_table_4680': 'uint8',
                'Weather_code_SYNOP_according_table_4677': 'uint8',
                'Radar_reflectivity': 'float',
                'MOR_visibility_precipitation': 'uint16',
                'Signal_amplitude_laser_strip': 'uint32',
                'Number_detected_particles': 'uint32',
                'Temperature_sensor': 'int8',
                'Current_through_heating_system': 'float',
                'Power_supply_voltage': 'float',
                'Sensor_status': 'int8',
                'Rain_amount_absolute': 'float32', #Da domandare se è troppo
                'Error_code': 'object',
                'FieldN': 'object',
                'FieldV': 'object',
                'RAW_data': 'object',
                'Unknow_column': 'float'
                }

    def __init__(self, campaign_path, campaign_name, chunk_size = None):
        """
        Parameters
        ----------
        
        campaign_name: str
            the name of the parsival data collection campaing
        campaign_path: path
            path location of the parent folder's campaign
        chunk_size: int
            custom chunk size for dask read_csv function
            
        """

        self.campaign_path = campaign_path
        self.campaign_name = campaign_name
        self.chunk_size = chunk_size
          
    def __merge_files_data(self, folder_name = None):
        """Convert all the data files in self.__getDataFiles() into a Dask dataframe, drop all the Nan rows and append in self.__temp_list

        Parameters
        ----------
        folder_name: string (optional)
            explicit the folder name inside <campaing_name>/data/<FOLDER_NAME>/*.dat, need it for loop every folder inside data.
            If is None, it mean the data folder hasn't subfolder


        Raises
        ------
        Exception, ValueError
            On .dat files error's, skip the file and write the file name and error into log file in <campaign_name>/processed/log_<datetime>.txt
        """
        
        import dask.dataframe as dd
        
        #If has argument, it mean there are folders to process
        if folder_name:
            self.file_list = []
            for file in self.folder_file_list[folder_name]:
                self.file_list.append(file)
                    
        for filename in self.file_list:
                try:
                    
                    if filename.endswith('.gz'):
                        df = dd.read_csv(filename, compression = 'gzip', delimiter = ',', on_bad_lines='skip', engine="python", blocksize=None, names = self.header_column_list, dtype = self.dtype_set)
                    
                        df = df.dropna(how='any')    
                    
                    else:
                        df = dd.read_csv(filename, delimiter = ',', on_bad_lines='skip', engine="python", blocksize=None, names = self.header_column_list, dtype = self.dtype_set)
                        
                        df = df.dropna(how='any')
                        
                    self.__temp_list.append(df)
                    
                except (Exception, ValueError) as e:
                  print("{} has been skipped. The error is {}".format(filename, e))
                  self.__temp_skip_files_list.append("{} has been skipped. The error is {}".format(filename, e))
                  self.__temp_skip_files += 1
                  
            
        print()
        print('{} files on {} has been skipped.'.format(self.__temp_skip_files, len(self.file_list)))
        print()
        
        #For log
        if folder_name:
            self.skip_files[folder_name] = self.__temp_skip_files_list
        
        self.__temp_skip_files = 0
        
    def __concat_frames(self):
        """Concat the frame inside __temp_list

        Parameters
        ----------
        None


        TODO
        Raises
        ------
        AttributeError, TypeError
            If can't concat, throw error
        """
        
        import dask.dataframe as dd 
        
        try:
            df = dd.concat(self.__temp_list, axis=0, ignore_index = True)
            return df
        except (AttributeError, TypeError) as e:
            print("Can not create concat data files. Error: {}".format(e))
            raise
    
    def __save_parquet(self, folder_name = None):
        """Save a dask dataframe into a parquet file inside a folder called 'processed', or 'processed'/<subfolder>/Processed_data_<campaing_name>_<subfolder>.parquet

        Parameters
        ----------
        None

        Raises
        ------
        Exception
            I don't know ...
        """
        
        import os
        
        if folder_name:
            __folder_path = r'/' + folder_name
        else:
            __folder_path = ''
        
        print('Starting conversion to parquet file')
        
        try:
            self.__concat_frames().to_parquet(os.path.join(self.campaign_path + r'/processed' + __folder_path,r'Processed_data_' + self.campaign_name + '_' + folder_name +'.parquet'), schema='infer')
        except (Exception) as e:
              print("Can not convert to parquet file. The error is {}".format(e))
              raise
        
        print('Processed_data' + self.campaign_name + '_' + folder_name + '.parquet is saved in {}'.format(self.campaign_path))
        
    def __check_processed_folder(self):
        """Check if exist /<campaign_name>/processed folder, if not will create it and also the subfolder if need it

        Parameters
        ----------
        None


        TODO
        Raises
        ------
        Exception
            If can't create folder, quit the script
        """
        
        import os
        
        if not os.path.isdir(self.campaign_path + r'/processed'):
            try:
                os.mkdir(self.campaign_path + r'/processed')
            except Exception() as e:
                print("Can not create folder <processed>. Error: {}".format(e))
                raise SystemExit(0)
        
        # Create subfolder for every device in processed folder
        if self.folder_file_list:
            for folder in self.folder_file_list.keys():
                if not os.path.isdir(self.campaign_path + r'/processed/' + folder):
                    try:
                        os.mkdir(self.campaign_path + r'/processed/' + folder)
                    except Exception() as e:
                        print("Can not create the device folder into {}/processed. Error: {}".format(self.campaign_path , e))
                        raise SystemExit(0)
                    except FileExistsError:
                        pass
            
    def __getDataFiles(self):
        """Populate the file_list with parsivel data files in the path set on campaign_path, and populate also the subfolder into folder_file_list if need it

        Parameters
        ----------
        None

        Raises
        ------
        IndexError
            No files to process, quit the script
        
        """
        
        import glob
        
        from pathlib import Path
        
        #Exit script if folder does not exist
        if not Path(self.campaign_path + '/data').exists():
            raise ValueError("The data folder doesn't exist, check the path")
               
                
         #Check if there are more than 1 Parsivel device (looking for subfolder)
        if glob.glob(self.campaign_path + '/data/*/'):
            for folder in glob.glob(self.campaign_path + 'data/*/'):
                try:
                    files = glob.glob(folder + "/*.dat*", recursive = True)
                    self.folder_file_list[folder.split('data/')[-1][:-1]] = files
                    files = []
                except IndexError:
                    pass
        
        #If file_list is empty, no files found
        if self.file_list:
            raise ValueError('None .dat files founds, check the path')
        else:
            try:
                self.file_list = glob.glob(self.campaign_path + "/**/*.dat*", recursive = True)
            except ValueError:
                raise('None .dat files founds, check the path')
        
    
    def __createREADME(self):
        """Create README.txt in processed folder to store campaign info
        
        Parameters
        ----------
        None

        Raises
        ------
        None
        
        """
        
        info = "###   Campaign info   ###\n\n"
        info += 'Campaign name: {}\n'.format(self.campaign_name)
        info += 'Campaign path: {}\n'.format(self.campaign_path)
        info += 'Campaign date: {}\n'.format('Unknown')
        info += 'Campaign location: {}\n'.format('Unknown')
        if self.folder_file_list:
            devices_list = ''
            for key in self.folder_file_list.keys():
                devices_list += '{}, '.format(key)
            info += 'Campaign devices: {}\n'.format(devices_list)
        
        with open(self.campaign_path + r'/processed/README.txt', 'w') as f:
            f.write(info)
            return print('README.txt is saved in {}processed/'.format(self.campaign_path))
        
        
    def __createLog(self):
        """Create lof.txt in processed folder to store conversion info and errors
        
        Parameters
        ----------
        None

        Raises
        ------
        None
        
        """
        
        from datetime import datetime
        
        ora = datetime.now().strftime('%Y%m%d%H%M%S')
        
        info = "###   Log info   ###\n\n"
        info += 'Script ran: ' + datetime.now().strftime('%d.%m.%Y %H:%M:%S') + '\n\n'
        
        if self.folder_file_list:
            for folder in self.folder_file_list.keys():
                info += 'Files processed in folder {}{}:\n\n'.format(self.campaign_path,folder)
                for file in self.folder_file_list[folder]:
                    info += '{}\n'.format(file)
                if self.skip_files:
                    info += "\nErrors in folder {}: \n'".format(folder)
                    for e in self.skip_files[folder]: info += e + '\n'
                info += "\n"
            info += "\n"
            info += 'Total processed files: {}\n\n'.format(sum(map(len, self.folder_file_list.values())))
            info += 'Total errors: {} on {} files\n\n'.format(sum(map(len, self.skip_files.values())),sum(map(len, self.folder_file_list.values())))
        else:
            info += 'Files processed: {}\n\n'.format(self.file_list)
            info += "File processed list: \n'"
            for p in self.file_list: info += p + '\n'
            if self.__temp_skip_files_list:
                info += "\nErrors: \n'"
                for p in self.__temp_skip_files_list: info += p + '\n'
        
        with open(self.campaign_path + r'/processed/log_{}.txt'.format(ora), 'w') as f:
            f.write(info)
            return print('log_{}.txt is saved in {}processed/'.format(ora, self.campaign_path))
        
        
    def create_parquet(self):
        """Merge all the data files found in the path folder into a parquet file inside a folder called 'processed', or into 'processed'/<subfolder> if there are subfolders


        Raises
        ------
        None, LOL
        
        """

        self.__getDataFiles()
        
        print(' ### BEGIN ### ')
        
        print()
        
        if self.folder_file_list:
            for dev in self.folder_file_list.keys():
                
                print("Processing {}{} folder".format(self.campaign_path, dev))
                
                print()
                
                self.__merge_files_data(dev)
        
                self.__check_processed_folder()
                
                self.__save_parquet(dev)
                
                print()
                
                print("Finish processing {}{} folder".format(self.campaign_path, dev))
                
                print()
        else:
            self.__merge_files_data()
        
            self.__check_processed_folder()
            
            self.__save_parquet()
        
        self.__createREADME()
        
        self.__createLog()
        
        print()
        print(' ### FINISH! ### ')

        
        
# -----------------------------------------------------------------------------------


camp = Campaign('/SharedVM/Campagne/Ticino_2018/', 'Ticino_2018')

camp.create_parquet()

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        