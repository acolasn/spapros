import iss_analysis.io as io
import iss_analysis.pick_genes as pick
import numpy as np
import os
import scanpy as sc
from pathlib import Path
import spapros as sp

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

datapath = Path('/nemo/lab/znamenskiyp/home/shared/projects/colasa_MOs_panel/yao_2021/')

savepath = Path("/nemo/lab/znamenskiyp/home/shared/projects/colasa_MOs_panel/panels/greedy_2021/")

eval_savepath = Path("/nemo/lab/znamenskiyp/home/shared/projects/colasa_MOs_panel/panels/evaluation_2021/")


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
    "Rrad",
    "Adamts2",
    "Slco2a1",
    "Npsr1", 
    "Slc17a7"
]

os.makedirs(savepath, exist_ok=True)
#pick.main(
#    savepath,
#    efficiency=0.01,
#    datapath=datapath,
#    subsample=1,
#    classify="subclass_label",
#    dataset = 'yao_2021', 
#    area = 'ALM', 
#    include_subclasses = ALM_subclasses, 
#    gene_set = prior_marker
#)


#evaluate

reference = io.load_yao_2021_to_anndata(datapath, 'ALM', ALM_subclasses)
#log nosmalize, get highly variable genes as candidates
sc.pp.filter_genes(reference, min_cells=1) # it crashes with a lot of genes expressed in no cells
sc.pp.normalize_total(reference)
sc.pp.log1p(reference)
sc.pp.highly_variable_genes(reference,flavor="cell_ranger",n_top_genes=10000) #for a good panel

os.makedirs(eval_savepath, exist_ok=True)

evaluator = sp.ev.ProbesetEvaluator(reference, 
                                    verbosity=2, 
                                    celltype_key = 'subclass_label', 
                                    results_dir=eval_savepath, 
                                    n_jobs=-1, 
                                    scheme = 'full')

npz_files = list(savepath.glob("*.npz"))


#znam = np.load('/nemo/lab/znamenskiyp/home/shared/projects/colasa_MOs_panel/panels/greedy_2021genes_subclass_label_e0.01_s1_20250924_185537.npz', allow_pickle=True)
znam = np.load(npz_files[0], allow_pickle = True)
panel = list(znam['gene_names'][znam['include_genes'] == True])


set_id = "greedy_2021"
evaluator.evaluate_probeset(panel, set_id=set_id)

print('DONE!!!')