import config as conf
import functions.helper_functions as helper_funcs

'''Downloads the article file tarballs and corresponding file lists
for the Pub Med Central Open Access collection, commercial license
subset.'''

if __name__ == "__main__":

    reconcat = False

    # Get article file tarballs
    reconcat = helper_funcs.download_files_from_list(
        conf.TARBALLS,
        conf.BASE_URL,
        conf.TARBALL_DIR,
        reconcat
    )

    # Extract article file tarballs
    reconcat = helper_funcs.extract_tarballs(
        conf.TARBALLS,
        conf.DATA_DIR,
        conf.TARBALL_DIR,
        reconcat
    )

    # Get article file lists
    reconcat = helper_funcs.download_files_from_list(
        conf.FILE_LISTS,
        conf.BASE_URL,
        conf.FILE_LIST_DIR,
        reconcat
    )

    # Concatenate individual file lists into master file
    # list and save to disk
    reconcat = helper_funcs.concatenate_csv_files(
        conf.FILE_LISTS, 
        conf.FILE_LIST_DIR, 
        conf.MASTER_FILE_LIST, 
        reconcat
    )