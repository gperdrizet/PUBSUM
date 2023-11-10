# PUBSUM: PUBMED Open Access article abstract summarization

The project goal is to provide high level summaries of current biomedical scientific findings which span multiple publications (think automatic literature reviews). To accomplish this the plan is to build an API which gives access to plain english summaries of new scientific publications added to the National Library of Medicine's Pub Med Central Open Access collection. Ideally, these summaries would span a publication cycle or more of a specific journal, journals or topic area and present developments in that scientific area.

## Progress

1. Demonstrated proof-of-concept scientific abstract summarization and model fine tuning using Huggingface and the haining/scientific_abstract_simplification model.
2. Created in house SQL database containing article metadata and text abstracts for all 3.68 million articles in the PUBMED Central Open Access Collection.
3. Started work on summarizing all or as many of those articles as possible.
