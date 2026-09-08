# Reproducing Robinson et al. 2021 (eLife 70564)

Python reimplementation of the Bar-seq analysis in *Natural variation in the
consequences of gene overexpression and its implications for evolutionary
trajectories* (Robinson, Place, Hose, Jochem, Gasch; eLife 2021;10:e70564).

The pipeline goes from the sequencing lanes in GEO series
[GSE171586](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE171586) to the
gene lists reported in the Results section:

```
FASTQ lanes
  -> unnormalized read counts          (Supplementary file 4, Tab 1)
  -> normalized read counts            (Tab 2)
  -> fitness scores, no imputation     (Tab 3)
  -> fitness scores, imputed           (Tab 4)
  -> commonly deleterious genes        (Tab 5)
  -> strain-specific deleterious       (Tab 6)
  -> strain-specific beneficial        (Tab 7)
  -> figures and reported statistics
  -> functional and biophysical enrichments
  -> RNA-seq differential expression (GSE171585) and cross-reference
  -> rendered figures
```
---

## Before you start: what is verified and what is not

I read the paper's Materials and methods and the Gasch lab's original counting
script ([github.com/wiscplace/BarSeq](https://github.com/wiscplace/BarSeq), by
Mike Place), and the matching rules in `step02` are transcribed from that
script. The `GENE_UPTAG_Decode.txt` file in `data/reference/` is the one from
that repository: 4,871 MoBY uptags, mostly 20 bases, ranging 10 to 22.

Three things I could **not** verify because the environment I wrote this in
could not reach NCBI or the eLife CDN:

1. **The layout of GSE171586.** How many FASTQs there are, whether they are raw
   multiplexed lanes or per-sample files, and what the GEO sample titles look
   like. `step00` fetches that for you and `step01` measures the read structure
   rather than assuming it.
2. **The exact column headers in Supplementary file 4.** `validate_against_supp4.py`
   inspects the workbook and matches columns by strain name, and prints what it
   found before comparing.
3. **Two definitional ambiguities in the methods.** Both are called out in the
   relevant script docstrings and are switchable on the command line: what
   counts as "missing reads from the end-point analysis" for imputation
   (`step04 --endpoint-rule`), and whether the "not more than two others" in the
   strain-specific definition counts same-direction or any significant effects
   (`step06 --specificity-rule`). Run both and see which reproduces the
   published list sizes.

The code itself is tested, in two ways.

`src/test_statistics.py` is a 15-check regression suite that needs no data and
no network. It verifies the negative binomial deviance against the
log-likelihood identity, the step-05 closed form and the step-10 IRLS fit
against `statsmodels`, the sum-to-zero contrast, Benjamini-Hochberg, and that
the enrichment engine finds planted signal while rejecting noise. Run it after
touching anything statistical:

```bash
python src/test_statistics.py
```

That suite exists because it caught a real bug. An early version of
`nb_deviance` had a sign error inside the logarithm, which left log2 fold
changes correct but inflated the likelihood ratio statistic and therefore the
significance calls. On the synthetic benchmark it dropped precision on
deleterious genes from ~1.00 to ~0.40. It is fixed, and the test now pins the
deviance to `2*(saturated - model)` log-likelihood to machine precision. If you
pulled an earlier copy of this code, re-run step 05.

The suite also cross-checks the `edgepython` backend against the built-in one
when edgePython is installed, confirming they agree on ranking and on the bulk
of fold changes.

`make_test_data.py` plus `check_against_truth.py` is the end-to-end check. On a
4-strain, 400-gene, 960k-read simulation the recovered log2 fold changes
correlate with truth at r = 0.96 to 0.97, recall on deleterious genes is 0.96 to
0.98, precision is 0.98 to 1.00, and the imputation step catches exactly the
genes that were forced to drop out.

---

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`rpy2` is optional and only needed for `step05 --backend edger`.

---

## Getting the data

### 1. Sequencing lanes

```bash
python src/step00_fetch_metadata.py
```

This writes `data/reference/samples_draft.tsv` and saves the raw GEO SOFT record
so you can read it yourself. Check every row against the GEO page, then download
the reads. If GEO lists SRA runs rather than supplementary FASTQs:

```bash
# conda install -c bioconda sra-tools
prefetch SRRxxxxxxx
fasterq-dump --split-files -O data/raw SRRxxxxxxx
gzip data/raw/*.fastq        # optional, the pipeline reads .gz directly
```

Fill in the `fastq` column and save the checked file as
`data/reference/samples.tsv`.

### 2. Work out the read structure

```bash
python src/step01_inspect_reads.py data/raw/<one_lane>.fastq.gz
```

This reports the uptag offset that maximizes exact matches and lists the most
common read prefixes. If one prefix per sample dominates, the lane is still
multiplexed: put those sequences in the `index_seq` column. If no prefix
dominates, the file is already demultiplexed: leave `index_seq` empty and give
each sample its own FASTQ.

Set `counting.uptag_offset` in `config/config.json` to whatever step 01
recommends. The default is 28, the value hard-coded in the original script.

### 3. Reference values for checking your work

```bash
mkdir -p data/external
curl -L -o data/external/elife-70564-supp4-v2.xlsx \
  https://cdn.elifesciences.org/articles/70564/elife-70564-supp4-v2.xlsx
curl -L -o data/external/elife-70564-fig1-data1-v2.txt \
  https://cdn.elifesciences.org/articles/70564/elife-70564-fig1-data1-v2.txt
curl -L -o data/external/elife-70564-fig3-data1-v2.txt \
  https://cdn.elifesciences.org/articles/70564/elife-70564-fig3-data1-v2.txt
```

---

## Running

Try the self-test first. It runs the statistical suite, simulates lanes, runs
everything, and checks the result against known truth. Takes about two minutes
and needs no downloads:

```bash
python run_all.py --self-test
```

Then the real thing:

```bash
python run_all.py --validate --plots          # core pipeline
python run_all.py --full --plots --validate   # plus annotations, enrichments, RNA-seq
```

Or one step at a time:

```bash
python src/step02_count_barcodes.py     # FASTQ -> counts
python src/step03_normalize.py          # -> normalized counts
python src/step04_impute.py             # -> imputed counts
python src/step05_fitness.py            # -> log2 fitness scores and FDR
python src/step06_gene_lists.py         # -> gene lists
python src/step07_core_analyses.py      # -> figure data and statistics
python src/step08_annotations.py        # -> annotation tables
python src/step09_enrichments.py        # -> enrichment results
python src/step10_rnaseq.py             # -> RNA-seq DE and cross-reference
python src/plots.py                     # -> rendered figures
python src/validate_against_supp4.py    # -> comparison with the published tables
```

Steps 08 to 10 need annotation files that GEO does not supply. Start with:

```bash
python src/fetch_external_data.py --list    # what is present, what is missing
python src/fetch_external_data.py           # download what can be automated
```

Nothing downstream needs all of them. Step 08 builds whatever it can from the
files present and prints which analyses are unavailable as a result, so you can
begin with GO annotations alone and add BioGRID or IUPred later.

---

## What each step does, and where it comes from in the paper

### step02: FASTQ -> read counts

Transcribes the matching logic of `BarSeq.py`. A read is assigned to a sample by
an exact match to its inline index at the start of the read. The gene uptag is
looked up at a fixed offset, trying window lengths 10 through 22 in ascending
order. Reads with no exact hit fall through to a one-mismatch dictionary
searched across offsets 25 to 30. Substituted bases are G, C, A, T and N, so a
tag containing one N still matches.

Outputs `counts_exact.tsv` and `counts_1mismatch.tsv`. The second is exact plus
one-mismatch summed, matching the original's `mergeCounts()`, and is what the
later steps use.

Two behaviours inherited from the original that are worth knowing about. The
one-mismatch dictionary is first-writer-wins, so a sequence reachable from two
different uptags is assigned to whichever gene appears first in the decode file.
This affects 17 sequences out of roughly 390,000; the full list is written to
`results/01_counts/fuzzy_tag_collisions.tsv`, and `--strict-fuzzy` drops them
instead. Separately, the original scans length 10 first, so a read whose first
10 bases happen to be a valid short uptag is assigned there even if its 20-mer
is also valid. That ordering is preserved.

The speed-up: every uptag is at least 10 bases, so a window whose first 10 bases
are not a known prefix cannot match at any length. One dictionary probe rejects
it instead of thirteen. Assignments are unchanged.

### step03: read counts -> normalized counts

> "barcode reads were divided by the total barcode read count in the sample,
> multiplied by 1 million to rescale for edgeR analysis. The latter provided the
> most robust procedure with the fewest assumptions."

Plain counts-per-million per sample. TMM is implemented behind `--method tmm`
only so you can reproduce the comparison the authors say they made; it is not
the pipeline default.

### step04: imputation

> "genes with at least 20 normalized read counts (>5th percentile of normalized
> reads) in all three replicates of the starting pool but missing reads from the
> end-point analysis received a pseudocount of 1 added to the barcode reads at
> 10 generations."

The threshold applies to normalized generation-0 counts in every replicate; the
pseudocount is added to raw generation-10 reads, so the matrix is renormalized
afterwards. An imputation mask is written so that later steps can exclude
imputed values, which the paper does for Figures 2B and 3B.

### step05: fitness scores and FDR

> "analyzed using edgeR version 3.22.1, using a linear model with generation
> (0 or 10) as a factor... Fitness scores were calculated by taking the ratio of
> normalized reads at generation 10 divided by reads at generation 0."

Each strain is analyzed separately. Two backends:

Three backends, chosen with `--backend`:

`edgepython` (recommended) uses [pachterlab/edgePython](https://github.com/pachterlab/edgePython)
(`pip install edgepython`), a pure-Python port of edgeR's internals: TMM,
weighted-likelihood empirical Bayes dispersion shrinkage, `glm_fit`, `glm_lrt`,
`top_tags`. It reproduces edgeR's dispersion moderation rather than
approximating it, and needs no R. `run_all.py` picks this automatically when it
is installed. One caveat worth stating plainly: edgePython's own README says it
was written mainly by Claude and Codex under Lior Pachter's direction, and its
validating preprint post-dates my training cutoff, so I am relying on the repo's
edgeR-vs-Python comparison notebooks rather than an independent benchmark. The
`edger` backend is kept precisely so you can confirm agreement yourself.

`python` (built-in, no extra dependency) is a negative binomial GLM with Cox-Reid
adjusted profile likelihood dispersion estimation and a likelihood ratio test.
Because step 03 makes every column sum to exactly 1e6, all offsets are equal, and
in a negative binomial GLM with a log link and a group-means design that makes
the maximum likelihood fitted value for each group the arithmetic group mean,
whatever the dispersion. Verified numerically against `statsmodels`: fitted
values, coefficients and the likelihood ratio statistic agree to machine
precision. So the log2 fold change is exactly `log2(mean at gen 10 / mean at
gen 0)`, the ratio the paper describes. What remains approximate is only the
empirical Bayes dispersion moderation, done here with a dispersion grid and
nearest-neighbour smoothing over abundance, so genes near the FDR cutoff can
flip relative to edgeR.

`edger` calls the real edgeR through rpy2. The reference implementation.

The three agree closely. On the synthetic benchmark, `python` and `edgepython`
log2 fold changes have Spearman correlation 1.00 and significance-call Jaccard
0.90 to 0.99 per strain; the only material difference is a handful of imputed
near-zero genes, where the built-in backend reports the raw ratio and edgePython
shrinks it via a prior count. The paper excludes imputed values from its figures
anyway. `test_statistics.py` pins this agreement.

Both write `fitness_scores_no_imputation.tsv` (Tab 3) and
`fitness_scores_imputed.tsv` (Tab 4).

### step06: gene lists

Commonly deleterious: significant deleterious in at least 10 of 15 strains. The
paper states this two ways, "at least 66% of strains" and ">=10 strains", which
give the same threshold. Target: 431 genes.

Strain-specific: significant in one strain and not more than two others,
separated into deleterious and beneficial. Targets: 41 genes in Y2209, 1763 in
Y12. YPS606 is excluded from these tallies by default, as in Figure 5, because
only duplicates were sequenced for that strain.

Also writes the four background sets described in Supplementary file 8, which
the enrichment tests in step 07 and beyond depend on.

### step07: figures and reported statistics

Replicate correlations, Figures 2A, 2B, 2C, 3A, 3B, the hierarchically clustered
matrix behind Figure 1B, and the beneficial-gene cluster from the Figure 6
section. Each prints the paper's corresponding number next to yours.

---

## Numbers to check against

| Quantity | Paper | Where |
|---|---|---|
| Genes significant in at least one strain | 4,064 | step06 |
| Median significant genes per strain | 1,726 | step06 |
| Commonly deleterious genes | 431 | step06 |
| Deleterious genes, Y2209 (lowest) | 635 | step07 figure2a |
| Deleterious genes, Y12 (highest) | 3,060 | step07 figure2a |
| Strain-specific genes, Y2209 | 41 | step06 |
| Strain-specific genes, Y12 | 1,763 | step06 |
| Beneficial cluster size | 21 | step07 |
| Median reads per barcode sample | 7,570,975 | step02 QC |
| Mean replicate correlation, most strains | 0.74 to 0.89 | step07 |
| Mean replicate correlation, Y2209/YJM1592/YJM978 | 0.55 to 0.65 | step07 |
| Deleterious genes in BY4743 overlapping Makanae 2013 | 851, p = 8e-45 | step09 |
| Commonly deleterious genes have more interactions | p = 4.0e-69, Wilcoxon | step09 |
| Commonly deleterious genes include more complex members | p = 6.6e-12, hypergeometric | step09 |
| Commonly deleterious genes are more disordered | p < 4.7e-12, Wilcoxon | step09 |
| RNA-seq genes significant in at least one strain | 4,802 | step10 |

If your counts and normalized counts correlate above ~0.99 with Tabs 1 and 2 but
the gene lists disagree, the problem is a definition, not the data. Re-run
step 06 with the other `--specificity-rule` and step 04 with the other
`--endpoint-rule`.

---

## Steps 08 to 10: the annotation-driven half

### step08: annotation tables

Turns whatever is in `data/external/` into two tables the tests consume:
`categories.tsv` (gene, category) and `features_continuous.tsv` (gene by numeric
feature). Sources are GO annotations from a GAF file, SGD chromosomal features,
protein sequences, BioGRID interaction counts, complex membership, ESR gene
sets, and any numeric table you supply. Every source is optional and an
`annotation_coverage.tsv` report says which ones loaded.

Derived features include per-amino-acid percentage composition, so `pct_W` is
the tryptophan measure behind the DBVPG1373 result, and distance to the nearest
centromere, for the beneficial-cluster result.

### step09: enrichments

Implements the paper's two test types on the paper's own backgrounds:

> "Wilcoxon rank-sum tests for continuous data... and Hypergeometric tests for
> categorical terms, taking as the background data set the total number of
> measured genes (except for strain-specific gene lists, in which the background
> data set was a list of insignificant genes in that strain with FDR>0.1...)"

and its significance rule:

> "We therefore took a stringent p-value of 5x10-4 as significant, but also cite
> FDR significance in data files."

Four analyses: the commonly deleterious set (run twice, with and without
translation-related genes, since the paper checks its enrichments survive that
removal); each strain's specific lists against its own no-effect background; the
Makanae et al. 2013 overlap; and the two Balance Hypothesis claims.

### step10: RNA-seq

The transcriptome arm differs from the Bar-seq model in two ways that matter:
each strain is compared against the mean of all strains rather than a control,
and replicate is a blocking factor. So the design is `~ replicate + strain`
fitted across all strains at once with sum-to-zero contrasts, which is what makes
each strain coefficient a deviation from the overall mean.

That is a genuine multi-coefficient GLM with no closed form. The built-in
`python` backend fits by IRLS; its coefficients match `statsmodels` to about
3e-6 and its likelihood ratio statistic to about 2e-9, both pinned by the test
suite. `--backend edgepython` runs the same design through the edgeR port and is
recommended; `--backend edger` uses real edgeR. All three take the sum-to-zero
strain contrast, so each strain coefficient is its deviation from the overall
mean.

`crossref` then intersects each strain's specific OE lists with its up- and
down-regulated genes, which is the connection the paper's strain-specific models
rest on.

### plots.py

Renders Figures 1B, 2A, 2B, 2C, 3A, 3B, Figure 2 supplement 1, and the replicate
correlation summary, with the paper's blue-yellow fitness scale and lineage
colouring. Everything is read from step 07 and step 09 output, so a wrong number
in a plot means the problem is upstream.

---

## External data you still need

`fetch_external_data.py` automates the downloads that have stable URLs. These do
not, and step 08 reports each one as missing until you supply it:

| Conclusion in the paper | File to provide |
|---|---|
| Overlap with Makanae et al. 2013 (851 genes, p = 8e-45) | `makanae2013_deleterious.txt`, one ORF per line |
| Translation, ribosome, ESR, essential-gene enrichments | `elife-70564-supp2-v2.xlsx` for the translation set; `esr_genes.tsv` from Gasch 2000 |
| Balance Hypothesis: complex membership | `cyc2008_complexes.tsv` from Pu et al. 2009 |
| Higher intrinsic disorder | `iupred_median.tsv`, from running IUPred2A over the proteome |
| Expression is not a predictor once translation factors are removed | `abundance.tsv` with mRNA and protein columns |
| Nonsynonymous SNP and amino acid differences vs S288c | `strain_snps.tsv`, derived from the 1002 Yeast Genomes Project, Strope 2015 or Bergström 2014 |
| Y7568 transcription factor targets | `tf_targets.tsv` from YEASTRACT+ (Monteiro 2020) |
| RNA-seq arm | `elife-70564-supp5-v2.xlsx`, or your own counts from GSE171585 |
| Empty-vector growth defect vs deleterious gene count (r = 0.7) | Figure 4C/4D doubling times, which have no machine-readable supplement |
| 2-micron REP1 abundance per strain | published DNA-seq per strain, mapped and RPKM-normalized |

Run `python src/fetch_external_data.py --list` for the exact expected format of
each.

## Layout

```
config/config.json           every tunable parameter, with the paper citation
data/reference/              decode file, sample sheet
data/raw/                    FASTQ lanes
data/external/               Supplementary file 4 and other published tables
results/01_counts/           counts_exact.tsv, counts_1mismatch.tsv, QC
results/02_normalized/       normalized_counts.tsv
results/03_imputed/          imputed counts, imputation_mask.tsv
results/04_fitness/          fitness_scores_*.tsv, per-strain edgeR-style tables
results/05_gene_lists/       commonly deleterious, strain-specific, backgrounds
results/06_analyses/         figure data and reported statistics
results/07_annotations/      categories.tsv, features_continuous.tsv, coverage
results/08_enrichments/      hypergeometric and Wilcoxon results
results/09_rnaseq/           RNA-seq DE and the OE cross-reference
results/10_figures/          rendered figures
src/                         the pipeline
```

## Sources

- Robinson D, Place M, Hose J, Jochem A, Gasch AP (2021). eLife 10:e70564.
  https://doi.org/10.7554/eLife.70564
- Original counting script: https://github.com/wiscplace/BarSeq
- MoBY 2.0 library: Ho et al. 2009 (Nat Biotechnol 27:369); Magtanong et al.
  2011 (Nat Biotechnol 29:505)
