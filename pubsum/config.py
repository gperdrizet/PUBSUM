import os
import do_not_commit as secrets

# Get path to this config file so that we can define
# other paths relative to it
PROJECT_ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

#################################################################
# PMC Open Access, commercial license article subset parameters #
#################################################################

BASE_URL = 'https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml'
DATA_DIR = f'{PROJECT_ROOT_PATH}/PMC_OA_comm_data'
TARBALL_DIR = f'{DATA_DIR}/tarballs'
FILE_LIST_DIR = f'{DATA_DIR}/file_lists'
MASTER_FILE_LIST = f'{DATA_DIR}/master_file_list.csv'

# Article tarballs
TARBALLS = [
    'oa_comm_xml.PMC000xxxxxx.baseline.2023-09-23.tar.gz',
    'oa_comm_xml.PMC001xxxxxx.baseline.2023-09-23.tar.gz',
    'oa_comm_xml.PMC002xxxxxx.baseline.2023-09-23.tar.gz',
    'oa_comm_xml.PMC003xxxxxx.baseline.2023-09-23.tar.gz',
    'oa_comm_xml.PMC004xxxxxx.baseline.2023-09-23.tar.gz',
    'oa_comm_xml.PMC005xxxxxx.baseline.2023-09-23.tar.gz',
    'oa_comm_xml.PMC006xxxxxx.baseline.2023-09-23.tar.gz',
    'oa_comm_xml.PMC007xxxxxx.baseline.2023-09-23.tar.gz',
    'oa_comm_xml.PMC008xxxxxx.baseline.2023-09-23.tar.gz',
    'oa_comm_xml.PMC009xxxxxx.baseline.2023-09-23.tar.gz',
    'oa_comm_xml.PMC010xxxxxx.baseline.2023-09-23.tar.gz'
]

# File lists for each article tarball
FILE_LISTS = [
    'oa_comm_xml.PMC000xxxxxx.baseline.2023-09-23.filelist.csv',
    'oa_comm_xml.PMC001xxxxxx.baseline.2023-09-23.filelist.csv',
    'oa_comm_xml.PMC002xxxxxx.baseline.2023-09-23.filelist.csv',
    'oa_comm_xml.PMC003xxxxxx.baseline.2023-09-23.filelist.csv',
    'oa_comm_xml.PMC004xxxxxx.baseline.2023-09-23.filelist.csv',
    'oa_comm_xml.PMC005xxxxxx.baseline.2023-09-23.filelist.csv',
    'oa_comm_xml.PMC006xxxxxx.baseline.2023-09-23.filelist.csv',
    'oa_comm_xml.PMC007xxxxxx.baseline.2023-09-23.filelist.csv',
    'oa_comm_xml.PMC008xxxxxx.baseline.2023-09-23.filelist.csv',
    'oa_comm_xml.PMC009xxxxxx.baseline.2023-09-23.filelist.csv',
    'oa_comm_xml.PMC010xxxxxx.baseline.2023-09-23.filelist.csv'
]

##################################
# PostgreSQL database parameters #
##################################

HOST = '192.168.2.1'
DB_NAME = 'postgres'
USER = 'postgres'
PASSWD = secrets.passwd

# XML parsing settings
WORKUNIT_SIZE = 10000
NUM_WORKERS = 16

###############################################
# Benchmarking independent vars. & parameters #
###############################################

device_map_benchmark_results_dir = f'{PROJECT_ROOT_PATH}/benchmarks/device_map'
device_map_benchmark_output_file_name = 'results.csv'
device_map_num_abstracts = 30
device_map_strategies = ['CPU only', 'multi-GPU', 'single GPU', 'balanced', 'balanced_low_0', 'sequential']