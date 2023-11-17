import re
import psycopg2
import xml.etree.ElementTree as ET

import functions.error_handling_functions as error_funcs

import os
import sys
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/..')

import config as conf

def parse_pmc_xml(workunit_article_paths):

    # Worker needs it's own connection to the database
    con = psycopg2.connect(f'dbname={conf.DB_NAME} user={conf.USER} password={conf.PASSWD} host={conf.HOST}')

    cur = con.cursor()

    # Counter for articles parsed
    article_count = 0

    # Iterate on target files
    for path in workunit_article_paths:

        # Get root of XML tree - protect me with try except!
        tree = None

        try:
            tree = ET.parse(str(path))

        # Catch and handle parse errors
        except ET.ParseError as e:
            print(f'Caught: {e}')

            # Try to recover tree after unbound prefix error
            tree = error_funcs.fix_unbound_prefix(path)

            # If we got a tree back, then we are all set
            if tree != None:
                print('Successfully recovered tree with lxml')

            # If value of tree is still None, we are out of options and
            # have to skip this article
            elif tree == None:
                print('Could not recover tree, skipping')

            print()
  
        # If we have a tree for this article
        if tree != None:
            # Find root
            root = tree.getroot()

            # Get article metadata
            front = root.find('front')
            article_meta = front.find('article-meta')

            # Get article data
            pmc_id = get_pmc_id_from_xml(article_meta, path)
            subject_data = get_subjects(article_meta, pmc_id)
            title_data = get_title(article_meta, pmc_id)
            abstract_data = get_abstract(article_meta, pmc_id)

            # Get refs
            back = root.find('back')
            ref_data = get_refs(back, pmc_id)

        # If we don't have a tree (meaning file loading or parsing failed) we still
        # want to put an empty entry in for this article so that we know that we have
        # parsed it
        if tree == None:
            # As a last resort, get the PMC ID from the file path
            pmc_id = pmc_id_from_path(path)
            subject_data = (pmc_id, None)
            title_data = (pmc_id, None)
            abstract_data = (pmc_id, None)
            ref_data = (pmc_id, None)

        # Insert subjects, title abstract and references
        cur.executemany("INSERT INTO subjects (pmc_id, subject) VALUES(%s, %s)", subject_data)
        cur.execute("INSERT INTO titles (pmc_id, title) VALUES(%s, %s)", title_data)
        cur.execute("INSERT INTO abstracts (pmc_id, abstract) VALUES(%s, %s)", abstract_data)
        cur.executemany("INSERT INTO refs (pmc_id, ref) VALUES(%s, %s)", ref_data)
        con.commit()

        # On to the next article
        article_count += 1

    # Close up shop for this workunit
    con.close()
    print(f'Completed {article_count} articles ending on: {str(path)}')

    # Noting to return so just send True for successful completion of workunit
    return True

def clean_text(text):
    '''Takes a block of text and removes junk and formatting, returns text'''
    # Remove latex
    remover = re.compile('\\\\.*{.*}')
    text = re.sub(remover, '', text)

    # Remove tabs and newlines ad replace with space
    remover = re.compile('[\t\n]')
    text = re.sub(remover, ' ', text)

    # Remove repeated spaces and replace with a single space
    remover = re.compile(' +')
    text = re.sub(remover, ' ', text)

    return text

def pmc_id_from_path(path):
    '''Takes a path to and article file and extracts PMC ID from filename,
    returns string PMC ID'''
    
    path = path.split('/')
    path = path[-1].split('.')
    pmc_id = path[0]

    return pmc_id

def get_pmc_id_from_xml(article_meta, path):
    '''Takes an XML tree containing the article metadata, also wants 
    the article's file path in case getting the PMC ID from the XML fails.
     Returns string PMC ID or None if not found'''

    # Set to None initially, so we don't return an uninitialized value if we can't find it
    pmc_id = None

    # Loop on article IDs
    for article_id in article_meta.findall('article-id'):

        # When we come across the PMC ID, grab it
        if article_id.get('pub-id-type') == 'pmc':
            pmc_id = article_id.text

    # If we scanned all of the article-ids and didn't find the PMC ID, try getting it from the filepath
    if pmc_id == None:
        pmc_id = pmc_id_from_path(path)

    # Return whatever we've got
    return pmc_id

def get_subjects(article_meta, pmc_id):
    '''Takes article metadata xml and PMC ID string, returns list of tuples each one containing the
    PMC ID and an article subject tag'''
    
    # If nothing else, at least return none
    subject_data = (pmc_id, None)

    # Make sure the article metadata is not empty
    if article_meta != None:

        # Get catagories from within the metadata
        article_categories = article_meta.find('article-categories')

        # Make sure there is a least one category
        if article_categories != None:
            subject_data = []

            # Loop on subjects in category
            for subject in article_categories.iter('subject'):

                # Get subject text and append to subject data list as tuple of PMC ID, and subject text
                subject_data.append((pmc_id, subject.text))

            # If we received no subject data, set return value to tuple of PMC ID and 'None'
            if len(subject_data) == 0:
                subject_data = (pmc_id, None)

    # Return whatever we've got
    return subject_data

def get_title(article_meta, pmc_id):
    '''Takes article metadata xml and pmc id string. Finds article title and returns tuple
    of PMC ID and article title.'''

    # If nothing else, at least return none
    title_data = (pmc_id, None)

    # Make sure metadata is not empty
    if article_meta != None:

        # Get the title group from within the metadata
        title_group = article_meta.find('title-group')

        # Make sure that the title group is not empty
        if title_group != None:

            # Find the actual title from inside the title group
            article_title = title_group.find('article-title')

            # Holder for title text
            title = []

            # If article text is not empty
            if article_title != None:

                # Extract all text in article title node
                for text in article_title.itertext():
                    title.append(text)

            # Make sure title is not empty
            if len(title) != 0:

                # Join each segment of text in the list
                title = ''.join(title)

                # Send to text cleaning function to get rid of
                # latex and extra tabs/spaces used for formatting
                title = clean_text(title)
            
            # If we couldn't find the title, set return value to None
            elif len(title) == 0:
                title = None

            # Construct tuple with result
            title_data = (pmc_id, title)

    # Return whatever we've got
    return title_data

def get_abstract(article_meta, pmc_id):
    '''Takes article metadata xml and PMC ID string, returns article abstract text.'''
    # If nothing else, at least return none
    abstract_data = (pmc_id, None)

    # Find article's abstract
    if article_meta != None:
        abstract = article_meta.find('abstract')

        # Collect text
        abstract_text = []

        if abstract != None:
            for text in abstract.itertext():
                abstract_text.append(text)

        if len(abstract_text) != 0:
            abstract_text = ''.join(abstract_text)

            # Do our best to clean up multi-paragraph abstracts with common section headings
            headings = [
                'Background', 
                'Introduction',
                'Rationale',
                'Methods', 
                'Results', 
                'Discussion', 
                'Conclusion', 
                'Significance', 
                'Summary'
            ]

            for heading in headings:
                remover = re.compile(heading)
                abstract_text = re.sub(remover, f' {heading}: ', abstract_text)

            # Remove the leading newline we may have just added
            abstract_text = abstract_text.lstrip()

            abstract_text = clean_text(abstract_text)
        
        elif len(abstract_text) == 0:
            abstract_text = None

        # Construct tuple with result
        abstract_data = (pmc_id, abstract_text)

    return abstract_data

def get_refs(back, pmc_id):
    '''Takes article back matter xml and PMC ID, returns list of tuples 
    of article PMC ID and PMC ID of each reference.'''

    # Set default return value to none, incase we don't find anything.
    ref_data = [(pmc_id, None)]

    # Make sure back matter xml is not empty
    if back != None:

        # Get the list of references
        ref_list = None
        ref_list = back.find('ref-list')

        # Make sure reference list is not empty
        if ref_list != None:

            # Scan the reference list
            for ref in ref_list.iter('ref'):

                # Find the citation tag for each reference
                element_citation = None
                element_citation = ref.find('element-citation')
                
                # If the citation exists 
                if element_citation != None:

                    # Find the DOI number from the citation
                    for pub_id in element_citation.findall('pub-id'):
                        if pub_id.get('pub-id-type') == 'doi':
                            doi = pub_id.text

                            print(f'Ref. DOI: {doi}')
                            # Add tuple of result to ref data list for return
                            ref_data.append((pmc_id, doi))

                # Also, look for mixed citations in case this reference is one of those
                # and do the same as for element-citation
                mixed_citation = None
                mixed_citation = ref.find('mixed-citation')

                if mixed_citation != None:
                    for pub_id in mixed_citation.findall('pub-id'):
                        if pub_id.get('pub-id-type') == 'doi':
                            doi = pub_id.text

                            ref_data.append((pmc_id, doi))

                # Lastly look for a citation tag, some earlier papers have those
                # do the same as for element-citation
                citation = None
                citation = ref.find('citation')

                if citation != None:
                    for pub_id in citation.findall('pub-id'):
                        if pub_id.get('pub-id-type') == 'doi':
                            doi = pub_id.text

                            ref_data.append((pmc_id, doi))

    # Return what we've got if we found anything, minus the None we started
    # the ref list with
    if len(ref_data) > 1:
        ref_data = ref_data[1:]

    return ref_data