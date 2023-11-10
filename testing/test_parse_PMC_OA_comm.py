import pathlib
import xml.etree.ElementTree as ET

import sys
sys.path.append('..')
import functions.xml_parsing_functions as xml_funcs

article_files_path = '../PMC_OA_comm_data/PMC000xxxxxx'
files = [file for file in pathlib.Path().glob(f'{article_files_path}/*.xml')]

for file in files[:10]:
    print(file)
    # Get root of XML tree - protect me with try except!
    tree = ET.parse(str(file))
    root = tree.getroot()

    # Get article metadata
    front = root.find('front')
    article_meta = front.find('article-meta')

    pmc_id = xml_funcs.get_pmc_id_from_xml(article_meta, file)
    subject_data = xml_funcs.get_subjects(article_meta, pmc_id)
    title_data = xml_funcs.get_title(article_meta, pmc_id)
    abstract_data = xml_funcs.get_abstract(article_meta, pmc_id)

    back = root.find('back')
    ref_data = xml_funcs.get_refs(back, pmc_id)

    print(f'PMC ID: {pmc_id}')
    #print(f'Title: {title_data[1]}\n')
    print(f'Subjects: {subject_data}')
    #print(f'Abstract: {abstract_data[1]}\n')
    print(f'Refs: {ref_data}')
    print('\n')
