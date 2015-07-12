# Spark eQTL 

This code enables eQTL analysis in Apache Spark, using Spark's python API
and has been tested with Spark 1.3.1 and 1.4.0.

[Klick here](http://tammor.github.io/content/report.pdf) for the correspoding report, explaining motivation, design and outline of the algorithm.

## Requirements:

- Spark and Python installation
- scipy.stats


## Quick start guide:

1. Start a spark master and submit some workers
2. Set up your Spark context within a python shell (see *spark\_context.py* for an example, no of cores and amount of memory is defined here.)
3. Define paths to your data inside the *trans\_analysis.py*
4. Logging behavious is defined in your spark directory (very verbose by default).
5. Within the python shell call *trans\_analysis.py*. Command line arguments define the output name and chromosome to be analyzed. E.g.::
   $run trans\_analyis.py 'full\_analysis_chrom\_1' 'chr1'


An shell script that automates all these task and analyzes the whole genome is: *run_full_example.sh*