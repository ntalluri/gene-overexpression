# What is each script doing

## prep_data.py

1) getting the normalized imputed data and cleaning it to be only the fitness scores
...
(will likely be other ways to fix the features and other data)

## distribution.py

I wanted to see how the OE fitness scores are distributed per strain and across all of the strains.

The data is normal distributed per strain. Some strains skew negative. In DBVPG1373, there are two humps; the mroe negative one is because there was a lot of imputations in the data

## get_yorf_to_accession_mapping.py

This code is used to prepare a file that tells us what genes yorf ids map to what uniprot accession ids
- this will help us in understanding what genes map to what protiens 

There are 3 sets of 2 genes that map to the same protein 
YKR059W: listed under From YJL138C (Entry P10081)
YNL030W: listed under From YBR009C (Entry P02309)
YNL031C: listed under From YBR010W (Entry P61830)

## baslines.py

## datasail_chtc_scripts

## clean_fasta.bash


