import numpy as np
import pandas as pd
from scipy.sparse import issparse
import iss_analysis.io as io
import scanpy as sc




def calculate_expression_per_taxa(panel_reference, taxa_label):

    '''
    Generates a matrix of taxa x genes, where each entry corresponds to the trimmed (25-75) mean expression per taxa. 
    '''

    #initialise the dataframe
    genes = panel_reference.var_names

    clusters = panel_reference.obs[taxa_label].to_numpy()
    uniq_clusters, inverse = np.unique(clusters, return_inverse=True) #inverse shows which elements belong to which cluster in the original matrix

    expression_percluster = pd.DataFrame(index=uniq_clusters, columns=genes)

    n_genes = panel_reference.n_vars
    print(f'There are {n_genes} genes in this panel')

    for k, cl in enumerate(uniq_clusters):
        # rows for this cluster
        rows = np.nonzero(inverse == k)[0]
        if rows.size == 0:
            continue

        # submatrix (cells_in_cluster x genes), dense for vectorized percentile ops
        Xc = panel_reference.X[rows, :].toarray() if issparse(panel_reference.X) else np.asarray(panel_reference.X[rows, :])

        # per-gene quartiles (vectorized across genes)
        p25, p75 = np.percentile(Xc, [25, 75], axis=0)

        # trim per gene
        mask = (Xc > p25) & (Xc < p75)            # shape: (cells_in_cluster, n_genes)
        count_trim = mask.sum(axis=0)             # per-gene counts after trimming
        sum_trim = (Xc * mask).sum(axis=0)        # per-gene sums after trimming

        # avoid division-by-zero when all values fall outside (rare but possible)
        trimmed_mean = np.divide(
            sum_trim, count_trim,
            out=np.zeros_like(sum_trim, dtype=float),
            where=count_trim > 0
        )

        # add to our matrix
        expression_percluster.loc[cl] = trimmed_mean

    return expression_percluster

def box_penalty(entropy, maxmean, k=5, entropy_threshold = 3.28, maxmean_threshold = 2):
    
    z_entropy = 1-np.array([1/(1+np.power(np.e, -k*(value-entropy_threshold))) for value in entropy])
    z_maxmean= [1/(1+np.power(np.e, -k*(value-maxmean_threshold))) for value in maxmean]
    penalty = np.minimum(z_entropy, z_maxmean)
    return penalty

def glia_expression_penalty(datapath, genes_in_reference, ALM_glia_subclasses=['Oligo', 'Astro', 'VLMC', 'Endo'], k=6, threshold=7):

    '''
    Calculate a penalty for genes expressed in glial subclasses. The penalty is a number 
    between 0 and 1, where 0 means no expression in glia, and 1 means high expression in glia.

    datapath: path to the yao_2021 dataset
    genes_in_reference: list of genes that are filtered in on the probeset selection dataset (through sc.pp.filter_genes)
    ALM_glia_subclasses: list of glial subclasses to consider. Defaults to omitted subclasses bc of low expression/cell counts
    k: steepness of the sigmoid function
    threshold: value of max expression at which the penalty is 0.5

    returns: list of penalties for each gene in the dataset, in the same order as the genes in the dataset
    '''
 

    glia_penalty_reference = io.load_yao_2021_to_anndata(datapath, 'ALM', ALM_glia_subclasses)

    glia_penalty_reference = glia_penalty_reference[: , glia_penalty_reference.var_names.isin(genes_in_reference)] #subset to the genes in the probeset selection dataset
    sc.pp.normalize_total(glia_penalty_reference)
    sc.pp.log1p(glia_penalty_reference)
    sc.pp.highly_variable_genes(glia_penalty_reference,flavor="cell_ranger",n_top_genes=10000) #for a good panel

    expression_persubclass = calculate_expression_per_taxa(glia_penalty_reference, 'subclass_label')

    max_values = expression_persubclass.max()

    penalty = 1-np.array([1/(1+np.power(np.e, -k*(value-threshold))) for value in max_values])
    
    return penalty