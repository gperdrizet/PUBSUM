import psycopg2
from typing import List

########################################################################
def use_gpu(device_map_strategy: str) -> bool:
    '''Takes human readable/plot-able device map strategy string and
    determines whether or not to use GPU.'''

    if 'GPU' in device_map_strategy:
        return True
    
    elif device_map_strategy == 'balanced':
        return True
      
    elif device_map_strategy == 'sequential':
        return True
    
    return False


########################################################################
def translate_device_map_strategy(device_map_strategy: str) -> str:
    '''Translate human readable/plot-able device map strategy 
    to huggingface device map string.'''

    # Set device map to CPU so that if all else fails, at least we
    # have something to return
    device_map = 'cpu'

    if device_map_strategy == 'multi-GPU':
        device_map = 'auto'

    elif device_map_strategy == 'single GPU':
        device_map = 'cuda:0'

    elif device_map_strategy == 'balanced':
        device_map = 'balanced'

    elif device_map_strategy == 'balanced_low_0':
            device_map = 'balanced_low_0'

    elif device_map_strategy == 'sequential':
        device_map = 'sequential'

    return device_map


########################################################################
def get_rows(
    db_name: str, 
    user: str, 
    passwd: str, 
    host: str, 
    num_abstracts: int
) -> List[List[str]]:
        
    # Open connection to PUBMED database on postgreSQL 
    # server, create connection
    connection = psycopg2.connect(
        f'dbname={db_name} user={user} password={passwd} host={host}'
    )

    # Start new reader cursor
    read_cursor = connection.cursor()

    # Loop until we have num_abstracts non-empty rows to return.
    # Note: ideally we would go back to the article parsing script 
    # and not put empty abstracts into the SQL database. Let's do
    # that later, but this will work for now to get us were we want
    # to go. Also, this is not being timed as part of the benchmark,
    # so any inefficacy is irrelevant to the results

    # On the first loop, the number of abstract to get is the total number we want
    abstracts_remaining = num_abstracts

    # Collector for result
    rows = []

    # Loop until we have enough abstracts
    while abstracts_remaining > 0:

        # Get abstracts
        read_cursor.execute(
            'SELECT * FROM abstracts ORDER BY random() LIMIT %s', 
            (abstracts_remaining,)
        )

        # Loop on the returned abstracts
        for row in read_cursor:

            # Get the abstract text
            abstract = row[1]

            # If we got a non-empty abstract, add the row to the result
            if abstract != None:
                rows.append(row)

        # Update abstracts remaining
        abstracts_remaining = num_abstracts - len(rows)
        
    read_cursor.close()

    return rows