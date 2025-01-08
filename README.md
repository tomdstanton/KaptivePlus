# KaptivePlus
#### A [Kaptive](https://kaptive.readthedocs.io) add-on for annotating genes in the context of locus typing results

## Overview
**KaptivePlus** is a flexible [**Kaptive**](https://kaptive.readthedocs.io) add-on which is basically a context-aware 
gene-annotation and gene-cluster-finding tool; the context in this case being locus typing results.

For each input assembly, the **KaptivePlus** pipeline performs 4 tasks:

### 1. Typing
The assembly is typed with [**Kaptive**](https://kaptive.readthedocs.io) using the chosen database to get the 
coordinates of the locus within the assembly.

### 2. ORF prediction
ORFs are predicted on each assembly contig using `pyrodigal`; this generates the proteome to be searched against
the input HMM profiles.

### 3. Annotation
Amino-acid sequences from each ORF are annotated with profiles using `pyhmmer`; the annotation is selected based on the 
profile hit with the highest score.

### 4. Finding Gene Clusters
For each contig, annotated genes are grouped together by their positional index, allowing
for some unannotated proteins (see `--skip-n-unannotated`). Groups of genes >= the minimum number of genes 
(see `--min-n-gene`) are considered to be complete gene cluster.

Take this example, where `A` = an annotated gene and `X` = an unannotated gene:

`[A1][A2][A3][X1][X2][A4]`

Here, 6 genes are grouped together consisting of 4 annotated genes and skipping over 2 unannotated genes.
To enforce this behavior, the flags `--min-n-gene=4` and `--skip-n-unannotated=2` would be set.

## Installation

**KaptivePlus** depends on [kaptive](https://kaptive.readthedocs.io/), [pyhmmer](https://pyhmmer.readthedocs.io/) and 
[pyrodigal](https://pyrodigal.readthedocs.io/); all of which can be installed with `pip`, but
installing **KaptivePlus** with pip will take care of these, 
which can be installed directly from source with:

```shell
pip install git+https://github.com/tomdstanton/KaptivePlus.git
```

## Usage

### Quickstart
```shell
kaptiveplus kpsc_k Wzy.hmm assembly.fasta > results.tsv
```

### All CLI options
```
usage: kaptiveplus <db> <hmm> <assembly> [<assembly> ...] [options]

A Kaptive add-on for annotating genes in the context of locus typing results

Inputs:

  db path/keyword       Kaptive database path or keyword
  hmm                   HMMER-formatted profile HMM file for hmmsearch
                        Note if pressed, hmmscan will be performed instead
  assembly              Assemblies in fasta(.gz|.xz|.bz2) format

Output Options:

  --tsv file            Output file to write/append per-gene tabular results to (default: -)
  --faa [dir/file]      Turn on gene protein fasta output
                        Accepts a single file or a directory (default: .)
  --ffn [dir/file]      Turn on gene nucleotide fasta output
                        Accepts a single file or a directory (default: .)
  --genbank [dir/file]  Turn on Gene Cluster Genbank output
                        Accepts a single file or a directory (default: .)
  --fasta [dir/file]    Turn on Gene Cluster nucleotide fasta output
                        Accepts a single file or a directory (default: .)

ORF Options:
  
  Options for tuning Pyrodigal

  --training-info       Pyrodigal training info (default: None)
  --good-assembly       Assembly to use for training the GeneFinder (default: None)
  --min-gene            The minimum gene length (default: 90)
  --min-edge-gene       The minimum edge gene length (default: 60)
  --max-overlap         The maximum number of nucleotides that can overlap between two genes on the same
                        strand (default: 60)
  --min-mask            The minimum mask length, when region masking is enabled. Regions shorter than the
                        given length will not be masked, which may be helpful to prevent masking of single
                        unknown nucleotides (default: 50)
  --meta                Run in metagenomic mode, using a pre-trained profiles for better results with
                        metagenomic or progenomic inputs (default: False)
  --closed              Consider sequences ends closed, which prevents proteins from running off edges
                        (default: False)
  --mask                Prevent genes from running across regions containing unknown nucleotides
                        (default: False)

HMM Options:
  
  Options for tuning PyHMMER

  --E                   The per-target E-value threshold for reporting a hit (default: 1e-20)
  --bit-cutoffs         The model-specific thresholding option to use for reporting hits
                        (default: None)

Gene Cluster Options:

  --min-n-genes         Minimum number of genes in each cluster (default: number of HMM files)
  --skip-n-unannotated 
                        Skip N unannotated genes when grouping neighbouring annotated genes
                        together in a cluster (default: 2)

Other Options:

  --no-header           Suppress header line
  -t int, --threads int
                        Number of alignment threads or 0 for all available (default: 0)
  -V, --verbose         Print debug messages to stderr
  -v, --version         Show version number and exit
  -h, --help            show this help message and exit

kaptiveplus v0.0.0b1
```

## Input files

**KaptivePlus** requires 3 positional arguments:
1. `<db>`: this can be a file or [keyword](https://kaptive.readthedocs.io/en/latest/Databases.html#database-keywords).
2. `<hmm>`: A [HMMER-formatted](https://www.genome.jp/tools/motif/hmmformat.htm) profile HMM file for annotating genes.
By default, `hmmsearch` will use the HMMs as queries and the proteome as the database, however if "pressed",
`hmmscan` will use the HMMs as a database, and each protein sequence as a query.
3. `<assembly>`: One or more bacterial genome assemblies in fasta format.

## Output files

### Tabular output
**KaptivePlus** will output a tabular report with one line per annotated gene it finds. The columns are as follows:

 - **Assembly**: Name of the input assembly
 - **Locus**: Kaptive locus call
 - **Phenotype**: Kaptive phenotype (type) call
 - **Confidence**: Kaptive confidence
 - **Contig**: The name of the contig sequence the gene is on
 - **Start**: Start coordinate of the gene (**_0-based_**)
 - **End**: End coordinate of the gene (**_end-inclusive_**)
 - **Strand**: Watson/Crick strand of the gene
 - **Gene**: Name of the gene (format: {contig}_000001)
 - **Best_HMM**: Accession of the HMM with the top hit
 - **Best_HMM_Score**: Score of the HMM with the top hit
 - **Locus_gene**: Locus gene if the gene overlaps with one
 - **Locus_gene_type**: Locus gene type (expected inside locus, etc.) if the gene overlaps with one
 - **Gene_cluster**: Name of the gene cluster the gene is part of (format: {contig}_000001)
 - **Problems**: If the ORF has a partial start/end

### Fasta output

### Genbank output
It is possible to write all gene clusters found to a file in Genbank format for
inclusion in the Kaptive database. One file will be written per assembly and
follow the general format of `{assembly}.gbk` in a directory specified
by the `--gbk` flag (defaulting to the current working directory).

These gene cluster Genbank files can then be used for adding "extra loci" to **Kaptive** databases.

## Performance
**KaptivePlus** is in no way optimized, but it should take ~3s per assembly once the database has been loaded
and the `pyrodigal.GeneFinder` instance has been trained.

## API
Like the [**Kaptive** API](https://kaptive.readthedocs.io/en/latest/Usage.html#api), the **KaptivePlus** API is
very basic, but it allows you to easily incorporate it into your own Python programs/notebooks/pipelines etc.

```python
from kaptiveplus import plus_pipeline

if result := plus_pipeline('assembly.fasta', 'kpsc_k', 'Wzy.hmm'):  # If KaptivePlus ran correctly
    print(result.format('tsv'), end='')  # TSV format will end in a newline, so we set end to ''
```
This returns a `kaptiveplus.KaptivePlusResult` instance which can be formatted to a string in tabular format using 
the `format()` method.

This instance also contains the `kaptive.typing.TypingResult` instance containing the **Kaptive** typing results which
can also be written to separate files if you wish.

If running on multiple assemblies, it is much more efficient to pre-load the **Kaptive** database and HMM file, and
also pre-train the `pyrodigal.GeneFinder`. Here is another example:

```python
from kaptiveplus import plus_pipeline, load_hmms, get_gene_finder
from kaptive.database import load_database
from pathlib import Path

db = load_database('kpsc_k')  # Load Kaptive database
hmms = load_hmms('Wzy.hmm')  # Load HMMs
gene_finder = get_gene_finder(good_assembly='complete_genome.fasta')  # Train GeneFinder instance on your good assembly

with open('pyrodigal.trn', 'wb') as f:
    gene_finder.training_info.dump(f)  # Save training info for later

# Get the KaptivePlus results on a folder of assembly files
results = [r for a in Path('assemblies').glob('*.fasta') if (r := plus_pipeline(a, db, hmms, gene_finder=gene_finder))]
```

**Note**
The `kaptive.assembly.Contig` instances are permanently overwritten by `kaptiveplus.Contig` instances, which are
subclasses of `Bio.SeqRecord.SeqRecord` objects; this was to make it easier to format the `ExtraLocus` objects
to Genbank format.

## Contact
[Tom Stanton](mailto:tom.stanton@monash.edu)

## Acknowledgements
Thank you to [Jo Kenyon](mailto:j.kenyon@griffith.edu.au) for providing the inspiration and use-case to put this together.

