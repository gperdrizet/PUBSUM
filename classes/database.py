import psycopg2
import psycopg2.extras
import config as conf

class Database:
    '''Class to hold SQL database connection stuff'''

    def __init__(self):

        # Open connection to PUBMED database on postgreSQL server, create connection
        self.con = psycopg2.connect(f'dbname={conf.DB_NAME} user={conf.USER} password={conf.PASSWD} host={conf.HOST}')
        self.cur = self.con.cursor()

        # Next, we need a table for the abstract summaries. First check to see if it exists already by 
        # trying to select it
        self.cur.execute('select exists(select * from information_schema.tables where table_name=%s)', ('abstract_summaries',))
        data_exists = self.cur.fetchone()[0]

        # If we get nothing back, we need to create the table
        if data_exists == False:
            print('\nCreating SQL table with PMC ID and abstract summary\n')

            self.cur.execute('''
                CREATE TABLE IF NOT EXISTS abstract_summaries(
                pmc_id varchar(12), abstract_summary text)
            ''')

            self.con.commit()

        # If the abstract summary table exists, make sure it's empty to start the run
        else:
            print('\nAbstract summary table exists\n')
            self.cur.execute('TRUNCATE abstract_summaries RESTART IDENTITY')
            self.con.commit()

        # Create a second cursor for reading and looping on chunks of rows from the database
        self.cur2 = self.con.cursor()

        # Get n rows from table of abstracts to be summarized
        self.cur2.execute('SELECT * FROM abstracts LIMIT %s', (conf.num_abstracts,))

    def insert(self, summary, pmcid):
        # Insert the new summary into the table
        self.cur.execute("INSERT INTO abstract_summaries (pmc_id, abstract_summary) VALUES(%s, %s)", (pmcid, summary))
        self.con.commit()