# This file contains useful functions and main function that loads vcf and gct data
# and saves resulting data frames.

from __future__ import print_function
from pyspark import SparkContext, SparkConf

from pyspark.sql import Row, SQLContext
from pyspark.sql.types import *
from pyspark.streaming import StreamingContext
from scipy import stats

import numpy as np
import scipy.sparse as scs
import socket
import logging
import time
import pickle
import sys
import os
# from sklearn import preprocessing

from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg import Matrix

import pdb # remove for production

# os.environ['PYSPARK_PYTHON'] = 'python2.7'

def get_time():
    return time.strftime("%y:%m:%d_%H:%M:%S")

def _mean_and_std(X, axis=0, with_mean=True, with_std=True):
    """Compute mean and std deviation for centering, scaling.
    Zero valued std components are reset to 1.0 to avoid NaNs when scaling.
    """
    X = np.asarray(X)
    Xr = np.rollaxis(X, axis)

    if with_mean:
        mean_ = Xr.mean(axis=0)
    else:
        mean_ = None

    if with_std:
        std_ = Xr.std(axis=0)
        if isinstance(std_, np.ndarray):
            std_[std_ == 0.] = 1.0
        elif std_ == 0.:
            std_ = 1.
    else:
        std_ = None

    return mean_, std_

def scale(X, axis=0, with_mean=True, with_std=True, copy=True):
    """ taken from scikit learn."""

    if scs.issparse(X):
        if with_mean:
            raise ValueError(
                "Cannot center sparse matrices: pass `with_mean=False` instead"
                " See docstring for motivation and alternatives.")
        if axis != 0:
            raise ValueError("Can only scale sparse matrix on axis=0, "
                             " got axis=%d" % axis)
        if not scs.isspmatrix_csr(X):
            X = X.tocsr()
            copy = False
        if copy:
            X = X.copy()
        _, var = mean_variance_axis(X, axis=0)
        var[var == 0.0] = 1.0
        inplace_column_scale(X, 1 / np.sqrt(var))
    else:
        X = np.asarray(X)
        mean_, std_ = _mean_and_std(
            X, axis, with_mean=with_mean, with_std=with_std)
        if copy:
            X = X.copy()
        # Xr is a view on the original array that enables easy use of
        # broadcasting on the axis in which we are interested in
        Xr = np.rollaxis(X, axis)
        if with_mean:
            Xr -= mean_
            mean_1 = Xr.mean(axis=0)
            # Verify that mean_1 is 'close to zero'. If X contains very
            # large values, mean_1 can also be very large, due to a lack of
            # precision of mean_. In this case, a pre-scaling of the
            # concerned feature is efficient, for instance by its mean or
            # maximum.
            if not np.allclose(mean_1, 0):
                warnings.warn("Numerical issues were encountered "
                              "when centering the data "
                              "and might not be solved. Dataset may "
                              "contain too large values. You may need "
                              "to prescale your features.")
                Xr -= mean_1
        if with_std:
            Xr /= std_
            if with_mean:
                mean_2 = Xr.mean(axis=0)
                # If mean_2 is not 'close to zero', it comes from the fact that
                # std_ is very small so that mean_2 = mean_1/std_ > 0, even if
                # mean_1 was close to zero. The problem is thus essentially due
                # to the lack of precision of mean_. A solution is then to
                # substract the mean again:
                if not np.allclose(mean_2, 0):
                    warnings.warn("Numerical issues were encountered "
                                  "when scaling the data "
                                  "and might not be solved. The standard "
                                  "deviation of the data is probably "
                                  "very close to 0. ")
                    Xr -= mean_2
    return X

def fast_corr(x,y):
    ''' This is copied from scipy.pearsonr, but without calculation the p value (saves time). '''
    x = np.asarray(x)
    y = np.asarray(y)
    mx = x.mean()
    my = y.mean()
    xm, ym = x - mx, y - my
    r_num = np.add.reduce(xm * ym)
    r_den = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
    r = r_num / r_den
    return np.abs(r)


def my_corr(gt_sparse, exp_mat, thres=.5):
    ''' For production use with matrix_mapping. Assuming exp_mat roes are alrdy demeaned and has unit variance.
        Still blowing up sparse genotype data to calculate correlation. not so nice.'''
    gt_idx = gt_sparse[1]
    gt_array = gt_sparse[0].toarray()
    gt_scaled = scale([float(x) for x in gt_array[0]]) # why is that not done during loading? -> sparsity

    corr = (np.dot(exp_mat, gt_scaled)/gt_sparse[0].shape[1]).tolist()
    corr = [(gt_idx,x,y) for x,y in enumerate(corr) if abs(y) > thres]
    
    return corr

def my_p(gt_sparse, exp_mat, thres =.2):

    gt_idx = gt_sparse[1]
    gt_array = gt_sparse[0].toarray()[0]

    corr = [stats.pearsonr(exp_mat[i],gt_array)[1] for i in range(len(exp_mat))]
    corr = [(gt_idx,x,y) for x,y in enumerate(corr) if y < thres]

    return corr

def my_p_corr(gt_sparse, exp_mat, thres=.1):
    ''' For production use with matrix_mapping. Assuming exp_mat roes are alrdy demeaned and has unit variance.
        Still blowing up sparse genotype data to calculate correlation. not so nice.'''
    
    gt_idx = gt_sparse[1]
    gt_array = gt_sparse[0].toarray()
    gt_scaled = scale([float(x) for x in gt_array[0]]) # why is that not done during loading? -> sparsity

    corr = []
    for i in range(len(exp_mat)):
        if np.abs(np.dot(exp_mat[i], gt_scaled)/gt_sparse[0].shape[1]) > .5:
            p = stats.spearmanr(exp_mat[i],gt_scaled)[1]
            if p <= thres:
                corr.append((gt_idx,i,p))  
    
    return corr

    
    
def get_start_end_chrom(gene_id, gene_locations):
    ''' int comparsion is much cheaper than string comparsion!'''
    return next([x[0],int(x[1]),int(x[2])] for x in gene_locations if int(x[3].strip()) == gene_id)

def is_in_gene_list(gene_id, gene_locations):
    try:
        dummy = next([int(x[1]),int(x[2])] for x in gene_locations if int(x[3].strip()) == gene_id)
        return True
    except:
        print('\n Careful, '+str(gene_id)+' is not available in the gene locations file and will be skipped.')
        return False

def extend_and_format(row):
    '''given a row from a SNP file, it is formated according to the scheme and
       split (this needs flatMap().) if it decodes information about two or more SNPs.'''
    
    if not ',' in row[4]: # return rows with only a single SNP (this is the vast majority)
        return [[int(row[1])]+[row[2]]+[row[3]+'to'+row[4]]+\
                [x.split(':')[0].count('1') for x in row[9:]]]
    else:
        targets = row[4].split(',') # Find different target nuceloitides
        # duplicate the row for every different target
        return [[int(row[1])]+[row[2]]+[row[3]+'to'+targets[i]]+\
                [x.split(':')[0].count(str(i+1)) for x in row[9:]]\
                for i in range(len(targets))]

        
def sparsify_list(in_array):
    in_array = np.asarray(in_array) # not necessary, checking too expensive(?)
    indices = in_array.nonzero()[0]
    values = in_array[indices]
    return (len(in_array), indices, values)
 
def within_cis_range(SNP_position, gene_range, max_distance):
    ''' Return True/False whether the given SNP lies within the specified distance of the gene. '''
    return gene_range[0] - max_distance < SNP_position < gene_range[1] + max_distance


def get_common_samples(vcf_source, gct_source, sample_list):
    ''' Retains order of vcf file to reduce shuffling. '''

    v = sc.textFile(vcf_source)
    lists_of_samples = v.filter(lambda line: '#CHROM' in line)\
                        .map(lambda line: line.split('\t')[9:]).collect()

    if False in [lists_of_samples[0]==lists_of_samples[j] for j in range(len(lists_of_samples))]:
        raise IOError('vcf files in data directory contain different samples')
    else:
        v_samples = lists_of_samples[0]
                
    # v = open(vcf_source, 'r')
    # for line in v:
    #     if '#CHROM' in line:
    #         v_samples = [x.strip() for x in line.split('\t')[9:]]
    #         break

    g = open(gct_source, 'r')
    for line in g:
        if 'NAME' in line:
            g_samples = [x.split('.')[0].strip() for x in line.split('\t')[2:]]
            break
            
    # list of the intersection of sample
    common_samples = [x for x in v_samples if x in g_samples]

    if sample_list != all:
        common_samples = [x for x in common_samples if x in sample_list]
    
    # indices of the respectice intersection samples
    v_indices = [v_samples.index(common_samples[j]) for j in range(len(common_samples))]
    g_indices = [g_samples.index(common_samples[j]) for j in range(len(common_samples))]

    return v_indices, g_indices, common_samples
            
def gct_to_mat(source, gene_location_file, exp_idx, logger, my_default_partitions):
    ''' REWRITE: scale and write expression values to simlpe local numpy ndarray.
        genes that are not present in the gene_location file are omitted.
        First dimension is gene expression, second dimension is sample.
        (data is transposed for later matrix-like operations.
        This should be extended to return an object that facilitates
        identification of the gene by it's index.'''

    with open(gene_location_file) as f:
        gene_locations = f.readlines() # maybe it's faster to pass only the path to the get_start_end() (?)
        gene_locations = [x.split('\t') for x in gene_locations] # and is this the best place to do this?
        
    exp_mat = sc.textFile(source) # repartitioning changes the order! 
    
    # This is a pretty bad way to filter comments, which are usually short.
    exp_mat = exp_mat.filter(lambda line: len(line) > 20 )

    exp_mat = exp_mat.map(lambda line: line.split('\t'))

    exp_mat = exp_mat.filter(lambda line: line[0] != 'NAME')

    exp_mat = exp_mat.filter(lambda line: is_in_gene_list(int(line[0]),gene_locations)) # only genes that are in gene_list
    
    exp_mat = exp_mat.map(lambda list: list[:2]+[float(x) for x in list[2:]]) # cast to float
    
    exp_mat = exp_mat.filter(lambda row: sum(x > 0 for x in row[2:]) >= max(len(row[2:])/20,5)) # at least 5% or 5 nz elements

    # take only a subset, for testing (20k samples runs on rbalhpc) and repartition

    exp_mat = sc.parallelize(exp_mat.collect(),numSlices=my_default_partitions*4)

    exp_ids = exp_mat.map(lambda line: line[:2] + get_start_end_chrom(int(line[0]), gene_locations))\
                     .zipWithIndex().map(lambda tuple: [int(tuple[1])]+tuple[0]) # id, name, descr., position 

    exp_mat = exp_mat.map(lambda row: row[2:])

    exp_mat = exp_mat.map(lambda list: scale([list[i] for i in exp_idx]))

    return exp_mat, exp_ids

def vcf_to_mat(source, gt_idx, logger, my_default_partitions):
    ''' Create RDD of sparse python matrices (1d), each containing data for a single SNP.
        Sort by position later? '''
    
    gt_mat = sc.textFile(source)# .repartition(8) # repartitioning mixes up order, but ids still work
    gt_mat = gt_mat.filter(lambda line: line[0] != '#')
    gt_mat = gt_mat.map(lambda line: line.split('\t'))

    gt_mat = gt_mat.flatMap(extend_and_format) # This could be optimized for matrix_mapping method
    # at least 5% or 5  nonzero elements
    gt_mat = gt_mat.filter(lambda row: sum(x > 0 for x in row[3:]) >  max(len(row[3:])/20,5))

    gt_mat.cache() # keep in memory since the next step needs materialization anyways
    
    gt_ids = gt_mat.map(lambda list: [int(list[0])]+list[1:3])\
                      .zipWithIndex().map(lambda tuple: [int(tuple[1])]+tuple[0])
    # take only values from common indices
 
    gt_mat = gt_mat.map(lambda list: scs.csr_matrix([list[3:][idx] for idx in gt_idx]))

    gt_mat = gt_mat.zipWithIndex()

    gt_mat.unpersist()

    gt_mat = gt_mat.repartition(my_default_partitions)

    return gt_mat, gt_ids
       

def crazy_analysis(gt_sparse, exp_mat_bc, sign_idx_pairs):
    gt_idx = gt_sparse[1]

    for idx in sign_idx_pairs:
        if idx[0] == gt_idx:
            scipy.stats.pearsonr(gt_sparse[0],exp_mat_bc.value[idx[1]])

    return 0

     
if __name__ == "__main__":
        
    hostname = socket.gethostname()
    conf = SparkConf().set("spark.python.profile", "true")\
                      .set('spark.logConf', 'true')\
                      .set('spark.eventLog.dir','.')\
                      .set('spark.eventLog.compress','true')\
                      .set('spark.eventLog.enabled','true')

    
    ### for rbalhpc05
    if hostname == 'rbalhpc05':
        conf = SparkConf().setAppName('eqtlanalysis').setMaster('spark://rbalhpc05:7077')\
           .set('spark.cores.max', sys.argv[2])\
           .set('spark.worker.memory','16g')\
           .set('spark.executor.memory','16g')\
           .set('spark.driver.memory','64g')\
           .set('spark.storage.memoryFraction','0.1') 
           # .set('spark.executor.heartbeatInterval','1000')\

        rbalhpc_data= '/data64/misc/hadoopeval/eqtl/tammo_testing/'
        sc = SparkContext(conf=conf)
        sqlContext = SQLContext(sc)
        ssc = StreamingContext(sc,1)
        ssc.checkpoint('/data64/misc/hadoopeval/eqtl/tammo_testing/rdd_checkpoints') 


        my_default_partitions = 8184
        
        ExprFile = '/data64/udis/ngs/USER_dropzone/RNASeq/10_processed/1000Genomes/1000Genomes_counts.gct'
        # ExprFile = '/homebasel/biocomp/rukatt/Dropbox/eQTLs/expression_dummy.gct' # for testing on rbal

        # SNPFile = rbalhpc_data+'vcf_files/*chr[0-9].*.vcf'
        # SNPFile = rbalhpc_data+'vcf_files/*.vcf'
        SNPFile = rbalhpc_data+'vcf_files/*'+sys.argv[4]+'.*.vcf'
        # SNPFile = rbalhpc_data+'ALL.'+chrom+'.phase1_release_v3.20101123.snps_indels_svs.genotypes.vcf'
        # SNPFile = '/homebasel/biocomp/rukatt/Dropbox/eQTLs/SNP_dummy.vcf' # 

        # gene location refr.
        gene_location_file = '/data64/misc/hadoopeval/eqtl/reference/entrez_gene_coordinates.bed'
        data_dir = '/data64/misc/hadoopeval/eqtl/tammo_testing/'
        drop_dir = '/homebasel/biocomp/rukatt/Dropbox/eQTLs/python/rbalhpc/'
                     

        ### for local tests
    elif hostname == 'hugo':
        conf = SparkConf().setAppName('eqtlanalysis').setMaster('spark://hugo:7077')\
           .set('spark.cores.max','2')\
           .set('spark.python.worker.memory','512m')
        
        sc = SparkContext(conf=conf)
        sqlContext  = SQLContext(sc)
        ssc = StreamingContext(sc,1)
        ssc.checkpoint('./RDD_checkpoints') 


        my_default_partitions = sc.defaultParallelism*2

        # path to the vcf file with genotype data
        SNPFile = '/home/tammo/Dropbox/eQTLs/eqtl_data/vcf_files/*.vcf'

        # path to gct file with expression data
        ExprFile = "/home/tammo/Dropbox/eQTLs/eqtl_data/expression_dummy.gct"

        # path to gene location file
        gene_location_file = '/home/tammo/Dropbox/eQTLs/eqtl_data/entrez_gene_coordinates.bed'
        data_dir = '/home/tammo/Dropbox/eQTLs/python/local/'
        drop_dir = '/home/tammo/Dropbox/eQTLs/python/local/'        

    try:
        outname = sys.argv[1]
    except:
        outname = 'unnamed_test'+get_time()
    logname = drop_dir+outname+'.log'

    logger = logging.getLogger('spark_eqtl_log') # py4j, pyspark
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(logname)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info(' Starting eQTL analysis for '+get_time()+'\n')        

    t_mm_0 = time.time()
    ### Filter list of samples, that are to be analysed
    sample_info = sc.textFile(data_dir+'population_information.txt')
    sample_list = sample_info.map(lambda line: line.split('\t'))

    ## Filter people with family status
    sample_list = sample_list.filter(lambda line: len(line[4]) < 2)

    ## Filter for gender
    # sample_list = sample_list.filter(lambda line: line[5] == 'Female')

    ## Filter for ethnicity
    if sys.argv[3] == 'EUR':
        EUR = ['British','Tuscan','European Utah residents','Finnish','Spanish Iberian']
        sample_list = sample_list.filter(lambda line: line[8] in EUR)
    elif sys.argv[3] == 'EAS':
        EAS = ['Denver Chinese','Han Chinese, Beijing','Han Chinese, Southern','Japanese']
        sample_list = sample_list.filter(lambda line: line[8] in EAS)
    elif sys.argv[3] == 'AFR':
        AFR = ['Yoruba','African-American','Luhya']
        sample_list = sample_list.filter(lambda line: line[8] in AFR)
    elif sys.argv[3] == 'AMR':
        AMR = ['Puerto Rican','Colombian','Mexican-American']
        sample_list = sample_list.filter(lambda line: line[8] in AMR)
        

    ## eventually collect sample names
    sample_list = sample_list.map(lambda line: line[6]).collect()

    # filter expression data for chromosome? --> not at the moment
    gt_idx, exp_idx, common_samples = get_common_samples(SNPFile, ExprFile, sample_list = sample_list)

    exp_mat, exp_ids = gct_to_mat(ExprFile, gene_location_file, exp_idx, logger, my_default_partitions)
    gt_mat, gt_ids = vcf_to_mat(SNPFile, gt_idx, logger, my_default_partitions)
    exp_mat = exp_mat.collect()



    # subsamples SNPs
    # gt_mat = gt_mat.sample(withReplacement=False,fraction=sys.argv[4])
    # gt_mat = gt_mat.repartition(my_default_partiget_common_samplestions*4)

    gt_mat.cache()
    gt_mat.count()
    no_of_comprs = len(exp_mat)*gt_mat.count()

    # gt_mat.saveAsPickleFile(data_dir+'gt_mat.pkl')
    # exp_mat.saveAsPickleFile(data_dir+'exp_mat.pkl')
    
    t_mm_1 = time.time()

    ## broadcast exp_mat manually or not?
    # exp_mat_bc = sc.broadcast(exp_mat.collect()) 

    
    corr = gt_mat.flatMap(lambda gt_row: my_p_corr(gt_row, exp_mat, thres=0.1))\
                 .filter(lambda line: not not line)
    
    corr.cache()
    corr.count()

    t_mm_0 = time.time() - t_mm_0
    t_mm_1 = time.time() - t_mm_1
    logger.info('matrix mapping complete, took '+str(t_mm_1)+'. With loading data: '+str(t_mm_0))

    sign_gt_ids_numbers = corr.map(lambda list: list[0])
    sign_gt_ids_numbers_collected = sign_gt_ids_numbers.collect()
    sign_gt_ids = gt_ids.filter(lambda list: list[0] in sign_gt_ids_numbers_collected)
    
    ## save output as dictionary
    output = {}
    output['no_of_cmprs'] = no_of_comprs
    output['exp_ids'] = exp_ids.collect()
    output['gt_ids'] = sign_gt_ids.collect()
    output['timing'] = (t_mm_0,t_mm_1)
    output['corr'] = corr.collect()
    pickle.dump(output, open(drop_dir+outname+'.pkl', 'wb'))

    ## save ids
    # pickle.dump( exp_ids.collect(), open(drop_dir+outname+'exp_ids.pkl', "wb"))
    # pickle.dump( gt_ids.collect(), open(drop_dir+outname+'gt_ids.pkl', "wb"))
    # pickle.dump( t_mm_1, open(drop_dir+outname+"timing.pkl", "wb"))
    # pickle.dump( corr.collect(), open(drop_dir+outname+'.pkl', "wb"))

    ## save to rbals data directory
    # corr.saveAsPickleFile(data_dir+outname+'.pkl')
    # exp_ids.saveAsPickleFile(data_dir+outname+'exp_ids.pkl')
    # gt_ids.saveAsPickleFile(data_dir+outname+'gt_ids.pkl')

    
    # exp_mat.unpersist()
    gt_mat.unpersist()
    corr.unpersist()

    # sc.stop()
    raise SystemExit
    #################################
    #### 2nd part of the analysis ###
    #################################

        
    sign_idx_pairs = corr.map(lambda list: (list[0],list[1]))
    corr.unpersist()
    
    sign_idx_gt = list(set(sign_idx_pairs.map(lambda line: line[0]).collect())) 
    sign_idx_exp = list(set(sign_idx_pairs.map(lambda line: line[1]).collect())) 

    # create RDD of sparse vectors with only significant SNPs
    gt_mat_sign = gt_mat.filter(lambda line: line[1] in sign_idx_gt)
    gt_mat.unpersist()

    exp_mat_sign = [exp_mat_bc.value[i] for i in sign_idx_exp]
    exp_ids_sign = [[j]+exp_ids.collect()[i][1:] for (j,i) in enumerate(sign_idx_exp)]

    raise SystemExit
    gt_mat_sign.map(lambda line: crazy_analysis(line, exp_mat_sign, sign_idx_pairs.collect()))
    