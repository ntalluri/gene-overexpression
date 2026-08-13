# Code work from "Natural variation in the consequences of gene overexpression and its implications for evolutionary trajectories"

# Overview 


# Normalization

I started by trying to recreate their data from the Moby BarSeq data https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE171586 to recreate the data found in supplement 4

The steps I took to recreate their data for both the imputed and unimputed data:
1. Summed the three sequencing lanes -> raw counts per sample
2. Counts per million: divided each sample by its own total, x 1e6
3. Conducted Imputation: genes with >=20 normalized counts in all three Gen0 replicates
     but no reads at Gen10 get a pseudocount of 1 added to their Gen10 counts
4. Fitness score = log2(normalized Gen10 / normalized Gen0) and then taking the average between replicates.

## Notes

Overall, my numbers are just similar to what is in Supplement 4 but not exact

For the unnormalized read counts, my unnormalized read counts done during step 1 are not close in counts at all. However, when I did the cpm normalization, my normalized values get closer to what is in the normalized read counts

Since the values were similar but not exact for the normalized and unnormalized read counts, the average log2 fitness scores I calculated were close but not exact to what is in supplemnt 4.

I also tried to use their unnomalized read counts data from supplement directly, instead of the Moby BarSeq raw data. This didn't get anywhere close to what I was seeing in the supplement 4 for any of the data; the numbers were completely off. So I essentially don't know where the unnormalized read counts came from.

# FDR

In the paper, they used analyzed the data using edgeR version 3.22.1, using a linear model with generation (0 or 10) as a factor. Genes whose barcodes were significantly different after 10 generations of growth in each strain at an FDR<0.05 were taken as significant. The FDR was calculated using the Benjamini and Hochberg method. 

For each strain, I fit a genewise negative binomial generalized linear model to the raw counts (imputed and not imputed) using edgePython. Counts were normalized by TMM library-size factors and gene-level dispersions were estimated before fitting. P-values came from a likelihood ratio test on the Gen0-to-Gen10 coefficient, then were adjusted across all genes within a strain by the Benjamini-Hochberg procedure to get the false discovery rate.

## Notes

Instead of using edgeR version 3.22.1, I used edgePython version 0.2.6 https://github.com/pachterlab/edgePython, a Python implementation of the Bioconductor edgeR package.

This was done using the data I created from the raw Moby BarSeq data. So the FDR was not exact, but did seem to be similar to what was in supplement 4.

# Commonly Deleterious, Strain Specific Benefical, and Strain Specific Deleterious 

I followed the procedure defined in the paper to find each set from the normalized imputed data:
1. To be strain-specific deleterious or strain-specific beneficial, the gene can only be in max 3 strains and have an FDR <=0.05 and have a positive fitness scores to be beneficla and negative fitness scores to be deleterious 
2. To be commonly deleterious, the gene needs to be in at least 10 of the 15 strains (66%), an FDR <=0.05, and have a negative fitness score

I then compared the data I got to what is shown in supplement 4

## Notes

### Commonly deleterious genes

Comparison of my commonly deleterious genes found against the paper's supplement,
split by whether the two sources report the same number of affected strains.

| Category                   | Strain count flag: False | Strain count flag: True |
| -------------------------- | -----------------------: | ----------------------: |
| `cd` (mine only)           |                        0 |                      19 |
| `supp_cd` (supplement only)|                        0 |                     110 |
| `both`                     |                      161 |                     160 |

- 19 genes are found only by me and do not appear in the supplement.
- 110 genes appear in the supplement but are not recovered by my analysis.
- The 160 genes are the set where my results and the paper agree on both the
  genes chosen and the strain count. The 161 genes found are the set where my results and the paper agree on both the
  genes chosen but do not aggree on the strain count.
- The strain count flag is only interpretable for the `both` row. Genes present
  in a single source have nothing to compare against, so they fall entirely on
  one side of the split by construction.

### Strain-specific genes

Column definitions for both tables below:

- `mine`: strain-specific genes found by my analysis
- `supp`: strain-specific genes reported in the paper's supplement
- `overlap`: genes found by both
- `jaccard`: `overlap / (mine + supp - overlap)`

YPS606 has no entry in the supplement, so overlap and Jaccard are undefined.

#### Beneficial

| strain     | mine | supp | overlap | jaccard |
| ---------- | ---: | ---: | ------: | ------: |
| BC187      |  400 |  490 |     335 |   0.604 |
| BY4743     |  285 |  444 |     276 |   0.609 |
| DBVPG1373  |  270 |  110 |      70 |   0.226 |
| NCYC3290   |  215 |   63 |      40 |   0.168 |
| Y12        |   60 |   36 |      19 |   0.247 |
| Y2209      |   39 |   94 |      38 |   0.400 |
| Y389       |  372 |  429 |     292 |   0.574 |
| Y7568      |  285 |  473 |     272 |   0.560 |
| YJM1273    |  282 |  145 |      89 |   0.263 |
| YJM1389    |  242 |   97 |      56 |   0.198 |
| YJM1592    |   69 |  108 |      45 |   0.341 |
| YJM978     |  176 |  165 |     108 |   0.464 |
| YPS128     |  185 |  163 |      90 |   0.349 |
| YPS163     |  347 |  297 |     181 |   0.391 |
| YPS606     |   59 |  n/a |     n/a |     n/a |

#### Deleterious

| strain     | mine | supp | overlap | jaccard |
| ---------- | ---: | ---: | ------: | ------: |
| BC187      |  245 |  142 |     106 |   0.377 |
| BY4743     |  205 |  112 |      95 |   0.428 |
| DBVPG1373  |  847 |  515 |     256 |   0.231 |
| NCYC3290   |  340 |  395 |      96 |   0.150 |
| Y12        |  551 |  990 |     113 |   0.079 |
| Y2209      |   35 |   16 |      11 |   0.275 |
| Y389       |  253 |  104 |      91 |   0.342 |
| Y7568      |  166 |   92 |      59 |   0.296 |
| YJM1273    |  258 |  155 |      98 |   0.311 |
| YJM1389    |  356 |  413 |      83 |   0.121 |
| YJM1592    |   58 |   23 |      19 |   0.306 |
| YJM978     |  115 |   44 |      32 |   0.265 |
| YPS128     |  185 |   92 |      58 |   0.252 |
| YPS163     |  404 |  232 |     160 |   0.336 |
| YPS606     |   58 |  n/a |     n/a |     n/a |


# Hierachial clustering

I tried to recreate the heatmaps and clustering that was done in the paper

## Notes

I didn't use the same code that was done in the paper, but from the figures I made, they seem similar. 
- I wasn't sure how the Imputed Defects were defined so I left those out