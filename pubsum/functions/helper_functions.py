import tarfile
import urllib.request
import pandas as pd
from os.path import exists

def download_files_from_list(
    files,
    url,
    output_dir,
    reconcat
):
    '''Downloads files from list of filenames and FTP url only if file doesn't
    already exist in output directory, returns reconcat = true if new files
    are downloaded.'''

    print()

    # Loop on files
    for file in files:

        # Check if file already exists in output dir
        if exists(f'{output_dir}/{file}') == False:

            # If we don't have this file, download it
            print(f'Retrieving: {url}/{file}')
            urllib.request.urlretrieve(f'{url}/{file}', f'{output_dir}/{file}')

            # Since we didn't have this file yet, we will need to concatenate
            # at the end to get the complete file list.
            reconcat = True

        # If we already have this file, skip it
        elif exists(f'{output_dir}/{file}') == True:
            print(f'{file} exists, skipping')

    return reconcat

def extract_tarballs(
    files, 
    data_dir, 
    tarball_dir, 
    reconcat
):
    '''Takes a list of tarballs as input and extracts them to disk using the
    PMC part number as the destination directory name.'''

    # Loop on files
    for file in files:

        # Get PMC part number from filename
        destination_dir = file.split('.')[1]

        # Check to see if destination directory exists, if it does not
        # then extract the corresponding tarball there
        if exists(f'{data_dir}/{destination_dir}') == False:

            reconcat = True

            with tarfile.open(f'{tarball_dir}/{file}') as tar:
                tar.extractall(f'{data_dir}')

        else:
            print(f'{destination_dir} exists, skipping.')

    return reconcat

def concatenate_csv_files(
    files, 
    input_directory, 
    output_file, 
    reconcat
):
    '''Takes a list of input files and and output destination. Loads and
    concatenates the input files and saves result to output file. Only runs
    if reconcat is true or output file does not already exist.'''

    # If we don't have the master file list yet or reconcat is true for 
    # any other reason, generate the master file list
    if (exists(output_file) == False) or (reconcat == True):

        # Empty list to contain dataframes of each individual file list
        file_list_dfs = []

        # Loop on files
        if len(files) > 0:
            print('\nConcatenating file lists')

            for file in files:

                # Load file into pandas df
                file_list = pd.read_csv(f'{input_directory}/{file}')

                # Add pandas df to list
                file_list_dfs.append(file_list)

            # Concatenate all of the dfs in the 
            file_list_df = pd.concat(file_list_dfs)

            # Write to disk
            file_list_df.to_csv(output_file)

            # Show first couple of rows
            print(file_list_df)

    else:
        print('\nCurrent concatenated file list exists.')

    print()

    return True