# spark_eqtl

This code enables eQTL analysis in Apache Spark, using Spark's python API
and has been tested with Spark 1.3.1 and 1.4.0.

Requirements:
- Spark and Python installation
- scipy.stats

Quick start guide:
0. Start a spark master and submit some workers
1. Set up your Spark context within a python shell (see *spark_context.py* for an example, no of cores and amount of memory is defined here.)
2. Define paths to your data inside the *trans_analysis.py*
3. Logging behavious is defined in your spark directory and could be very verbose by default.
4. Within the python shell call *trans_analysis.py*, defining output name and chromosome to be analyze. E.g.::
   $run trans_analyis.py 'full_analysis_chrom_1' 'chr1'

An shell script that automates all these task and analyzes the whole genome is: *run_full_example.sh*