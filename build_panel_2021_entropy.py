import scanpy as sc
import spapros as sp
import anndata as ann
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
REDUNDANT_PANEL = False
GLIAL_PENALTY = False
PANEL_ID = "spapros_Rob_VISp_40"
AREA = ["VISp"+"ALM"]
EVALUATE = True
N_GENES = 104 #100 read by sequencing, 4 in a hyb round

###

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

VISp_subclasses = [
        "L4/5 IT CTX",
        "Sst",
        "L6 IT CTX", 
        "Vip",
        "Pvalb",
        "Lamp5",
        "L2/3 IT CTX",
        "L6 CT CTX",
        "L5 IT CTX",
        "L5 PT CTX",
        "L5/6 NP CTX",
        "L6b CTX",
        "Sst Chodl",
        "Car3",
        "Sncg"
    ]

if AREA == "ALM":
    subclasses = ALM_subclasses 
elif AREA == "VISp":
    subclasses = VISp_subclasses 
elif isinstance(AREA, list):
    subclasses = list(set(ALM_subclasses + VISp_subclasses))
else:
    raise ValueError("Specify the set of subclasses for the chosen area")

datapath = Path('/nemo/lab/znamenskiyp/home/shared/projects/colasa_MOs_panel/yao_2021')

savepath = "/nemo/lab/znamenskiyp/home/shared/projects/colasa_MOs_panel/panels/" + PANEL_ID

redundant_marker = '/nemo/lab/znamenskiyp/home/shared/projects/colasa_MOs_panel/panels/spapros_10K_taylored_2021'



prior_marker = [
    "Chodl",
    "Cux2",
    "Fezf2",
    "Foxp2",
    "Rorb",
    "Lamp5",
    "Adamts2",
    "Slco2a1",
    "Vip", 
    "Sst", 
    "Pvalb",
    "Cdh13", 
    "Pcp4", 
    "Sncg", 
    "Slc17a7", 
    "Gad1"
]

excluded_marker = [
    "Rmst"]


### Build reference dataset

if isinstance(AREA, list):
    reference_list = []
    for area in AREA:
        ref_area = io.load_yao_2021_to_anndata(datapath, area, subclasses)
        reference_list.append(ref_area)
    reference = ann.concat(reference_list, join='inner', axis=0)
else:
    reference = io.load_yao_2021_to_anndata(datapath, AREA, subclasses)

for gene in prior_marker:
    assert len(reference.var.index[reference.var.index==gene])>0

#log nosmalize, get highly variable genes as candidates
sc.pp.filter_genes(reference, min_cells=1) # it crashes with a lot of genes expressed in no cells
sc.pp.normalize_total(reference)
sc.pp.log1p(reference)
sc.pp.highly_variable_genes(reference,flavor="cell_ranger",n_top_genes=10000) #for a good panel

os.makedirs(savepath, exist_ok=True)

### Calculate sparseness constraints
with open("logreg_params.json") as f:
  params = json.load(f)

#create a new model with same shape
full_model = LogisticRegression()
full_model.classes_ = np.array([0, 1])  # must set classes manually
full_model.coef_ = np.array(params["coef"])
full_model.intercept_ = np.array(params["intercept"])
print("Intercept:", full_model.intercept_)
print("Coefficients:", full_model.coef_[0])

#Evaluate all genes
all_genes_subclass = ut.calculate_expression_per_taxa(reference, 'subclass_label')
all_genes_H = entropy(all_genes_subclass.to_numpy(dtype = float), base = 2, axis = 0)
assert list(reference.var_names)==list(all_genes_subclass.columns), 'Something is wrong with the sorting'
all_genes_entropies = pd.DataFrame(all_genes_H, index=reference.var_names)
all_genes_entropies = all_genes_entropies.fillna(100) #basically set the score to 0 if your entropy is a NaN
score = full_model.predict_proba(all_genes_entropies)

#Calculate glia penalty
if GLIAL_PENALTY:
    genes_in_reference = list(reference.var_names)
    glia_score = ut.glia_expression_penalty(datapath, genes_in_reference, ALM_glia_subclasses=['Oligo', 'Astro', 'VLMC', 'Endo'], k=6, threshold=7)

    #Choose the most restrictive of the two penalties
    final_score = np.minimum(score[:, 1], glia_score)

else:
    
    final_score = score[:, 1]

reference.var['score'] = final_score #we want the score to be 1 if the gene belongs to class 1: in training, that is "good"

for marker in excluded_marker:
    reference.var.loc[marker, 'score'] = 0 #exclude some markers

#Load redundant markers
if REDUNDANT_PANEL:
    old_panel_df = pd.read_csv(redundant_marker + '/probeset.csv', index_col=0)
    old_panel = list(old_panel_df.index[old_panel_df['selection']])
    prior_marker = old_panel

#### Run spapros

selector = sp.se.ProbesetSelector(reference, n=N_GENES, 
                                  celltype_key="subclass_label", 
                                  verbosity=1, 
                                  n_jobs=-1, 
                                  save_dir=savepath, 
                                  preselected_genes=prior_marker, 
                                  pca_penalties=['score'], 
                                  DE_penalties=['score']
                                  )

selector.select_probeset()

if EVALUATE:

    selector.select_probeset()

    eval_savepath = "/nemo/lab/znamenskiyp/home/shared/projects/colasa_MOs_panel/panels/evaluation_2023"

    os.makedirs(eval_savepath, exist_ok=True)

    selected = selector.probeset[selector.probeset["selection"]].copy()
    spapros_panel = list(selected.index)

    evaluator = sp.ev.ProbesetEvaluator(reference, 
                                        verbosity=2, 
                                        celltype_key = 'subclass_label', 
                                        results_dir=eval_savepath, 
                                        n_jobs=-1, 
                                        scheme = 'full')

    #reference_sets = sp.se.select_reference_probesets(reference, n = 100, obs_key = "subclass_label")
    #for set_id, df in reference_sets.items():
    #    gene_set = df[df["selection"]].index.to_list()
    #    evaluator.evaluate_probeset(gene_set, set_id=set_id)


    set_id = PANEL_ID
    evaluator.evaluate_probeset(spapros_panel, set_id=set_id)

print('DONE!!!')