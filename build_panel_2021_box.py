import scanpy as sc
import spapros as sp
import numpy as np
import os
import utils as ut
from scipy.stats import entropy
import json
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import iss_analysis.io as io
import iss_analysis.pick_genes as pick
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache


### parameters


ALM_subclasses = [
    "Vip",
    "Lamp5",
    "Scng",
    "Sst Chodl",
    "Sst",
    "Pvalb",
    "L2/3 IT CTX",
    "L4/5 IT CTX",
    "L5 IT CTX",
    "L6 IT CTX",
    "L5 PT CTX",
    "L4 RSP-ACA",
    "L5/6 NP CTX",
    "L6 CT CTX",
    "L6b CTX",
]

datapath = Path('/nemo/lab/znamenskiyp/home/shared/projects/colasa_MOs_panel/yao_2021')

savepath = "/nemo/lab/znamenskiyp/home/shared/projects/colasa_MOs_panel/panels/spapros_10K_box_sparseness"

prior_marker = [
    "Chodl",
    "Cux2",
    "Fezf2",
    "Foxp2",
    "Rorb",
    "Vip",
    "Pvalb",
    "Sst",
    "Lamp5",
    "Adamts2",
    "Slco2a1",
]

excluded_marker = [
    "Vip", 
    "Sst"
]


### Build reference dataset

reference = io.load_yao_2021_to_anndata(datapath, 'ALM', ALM_subclasses)

for gene in prior_marker:
    assert len(reference.var.index[reference.var.index==gene])>0

#log nosmalize, get highly variable genes as candidates
sc.pp.filter_genes(reference, min_cells=1) # it crashes with a lot of genes expressed in no cells
sc.pp.normalize_total(reference)
sc.pp.log1p(reference)
sc.pp.highly_variable_genes(reference,flavor="cell_ranger",n_top_genes=10000) #for a good panel

os.makedirs(savepath, exist_ok=True)

### Calculate sparseness constraints

#Evaluate all genes
all_genes_subclass = ut.calculate_expression_per_taxa(reference, 'subclass_label')
all_genes_H = entropy(all_genes_subclass.to_numpy(dtype = float), base = 2, axis = 0)
assert list(reference.var_names)==list(all_genes_subclass.columns), 'Something is wrong with the sorting'
all_genes_maxmean = all_genes_subclass.max() - all_genes_subclass.mean()
penalty = ut.box_penalty(all_genes_H, all_genes_maxmean)
reference.var['score'] = penalty #we want the score to be 1 if the gene belongs to class 1: in training, that is "good"

#### Run spapros

selector = sp.se.ProbesetSelector(reference, n=100, 
                                  celltype_key="subclass_label", 
                                  verbosity=1, 
                                  n_jobs=-1, 
                                  save_dir=savepath, 
                                  preselected_genes=prior_marker, 
                                  pca_penalties=['score'], 
                                  DE_penalties=['score'])

selector.select_probeset()

eval_savepath = "/nemo/lab/znamenskiyp/home/shared/projects/colasa_MOs_panel/panels/evaluation_2021"

os.makedirs(eval_savepath, exist_ok=True)

selected = selector.probeset[selector.probeset["selection"]].copy()
spapros_panel = list(selected.index)

evaluator = sp.ev.ProbesetEvaluator(reference, 
                                    verbosity=2, 
                                    celltype_key = 'subclass_label', 
                                    results_dir=eval_savepath, 
                                    n_jobs=-1, 
                                    scheme = 'full')

reference_sets = sp.se.select_reference_probesets(reference, n = 100, obs_key = "subclass_label")
for set_id, df in reference_sets.items():
    gene_set = df[df["selection"]].index.to_list()
    evaluator.evaluate_probeset(gene_set, set_id=set_id)


set_id = "spapros_10K_2021_box_sparseness_panel"
evaluator.evaluate_probeset(spapros_panel, set_id=set_id)

print('DONE!!!')