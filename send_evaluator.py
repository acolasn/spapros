import scanpy as sc
import spapros as sp
import numpy as np
import pandas as pd
from pathlib import Path
import os
import iss_analysis.io as io
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache

download_base = Path('/nemo/lab/znamenskiyp/home/shared/projects/colasa_MOs_panel')
abc_cache = AbcProjectCache.from_cache_dir(download_base)

abc_cache.current_manifest

# Start with train dataset

reference = io.main_yao_2023(abc_cache, 
                  'WMB-10Xv3', 
                  ['WMB-10Xv3-Isocortex-2', 'WMB-10Xv3-Isocortex-1'], 
                  download_base,
                  taxa_class = ['01 IT-ET Glut', '06 CTX-CGE GABA', '07 CTX-MGE GABA',  '08 CNU-MGE GABA', '09 CNU-LGE GABA'],
                  region_of_interest = 'MO-FRP', 
                  neurotransmitters = ['GABA', 'Glut'], 
                  extract_csv = False)

sc.pp.normalize_total(reference)
sc.pp.log1p(reference)
sc.pp.highly_variable_genes(reference,flavor="cell_ranger",n_top_genes=1000)

#load the spapros panel
selector = sp.se.ProbesetSelector(reference, n=100, celltype_key="subclass", verbosity=1, n_jobs=-1, save_dir="/nemo/lab/znamenskiyp/home/shared/projects/colasa_MOs_panel/panels/spapros")
lookup = pd.DataFrame({
    "gene_symbol": reference.var['gene_symbol'],
    "feature_id": reference.var.index
})

selected = selector.probeset[selector.probeset["selection"]].copy()

selected = selected.merge(lookup, 
                          left_index=True,
                          right_on="feature_id",
                          how="left")

spapros_panel = list(selected.index)

spapros_panel

#load the znam panel
znam = np.load('/nemo/lab/znamenskiyp/home/shared/projects/colasa_MOs_panel/panels/greedy_panels/greedy_panelsgenes_subclass_e0.01_s1_20250919_165916.npz', allow_pickle=True)
panel = list(znam['gene_names'][znam['include_genes'] == True])
panel = pd.DataFrame(panel)

# Build dict: gene_symbol → list of feature_ids
#SOME GENE SYMBOLS HAVE MORE THAN ONE EMSEBBL ENTRY BUT NOT OURS
multi_mapping = lookup.groupby("gene_symbol")["feature_id"].apply(list).to_dict()

# Map panel values, may return lists
panel_feature_ids = panel[0].map(multi_mapping)

petr_panel = list(panel_feature_ids)
petr_str = [str(x[0]) for x in petr_panel]


#compare in train dataset
reference_sets = sp.se.select_reference_probesets(reference, n = 100, obs_key = "subclass")
savepath = "/nemo/lab/znamenskiyp/home/shared/projects/colasa_MOs_panel/panels/evaluation_sbatch"
os.makedirs(savepath, exist_ok=True)
evaluator = sp.ev.ProbesetEvaluator(reference, verbosity=2, celltype_key = 'subclass', results_dir=savepath, n_jobs=-1)
#for set_id, df in reference_sets.items():
#    gene_set = df[df["selection"]].index.to_list()
#    evaluator.evaluate_probeset(gene_set, set_id=set_id)


#set_id = "greedy_panel"
#evaluator.evaluate_probeset(petr_str, set_id=set_id)
#set_id = "spapros_panel"
#evaluator.evaluate_probeset(spapros_panel, set_id=set_id)


#and now on test dataset
del reference

reference = io.main_yao_2023(abc_cache, 
                  'WMB-10Xv2', 
                  ['WMB-10Xv2-Isocortex-3', 'WMB-10Xv2-Isocortex-1', 'WMB-10Xv2-Isocortex-4', 'WMB-10Xv2-Isocortex-2'], 
                  download_base,
                  taxa_class = ['01 IT-ET Glut', '06 CTX-CGE GABA', '07 CTX-MGE GABA',  '08 CNU-MGE GABA', '09 CNU-LGE GABA'],
                  region_of_interest = 'MO-FRP', 
                  neurotransmitters = ['GABA', 'Glut'], 
                  extract_csv = False)

sc.pp.filter_genes(reference, min_cells=1)
sc.pp.normalize_total(reference)
sc.pp.log1p(reference)
sc.pp.highly_variable_genes(reference,flavor="cell_ranger",n_top_genes=1000)

reference_sets = sp.se.select_reference_probesets(reference, n = 100, obs_key = "subclass")
savepath = "/nemo/lab/znamenskiyp/home/shared/projects/colasa_MOs_panel/panels/evaluation_testset"
os.makedirs(savepath, exist_ok=True)
evaluator = sp.ev.ProbesetEvaluator(reference, verbosity=2, celltype_key = 'subclass', results_dir=savepath, n_jobs=-1)
#for set_id, df in reference_sets.items():
#    gene_set = df[df["selection"]].index.to_list()
#    evaluator.evaluate_probeset(gene_set, set_id=set_id)


set_id = "greedy_panel"
evaluator.evaluate_probeset(petr_str, set_id=set_id)
set_id = "spapros_panel"
evaluator.evaluate_probeset(spapros_panel, set_id=set_id)

print('DONE!!!')