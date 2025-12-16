import scanpy as sc
import spapros as sp
from pathlib import Path
import iss_analysis.io as io
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache


download_base = Path('/nemo/lab/znamenskiyp/home/shared/projects/colasa_MOs_panel')
abc_cache = AbcProjectCache.from_cache_dir(download_base)

abc_cache.current_manifest

reference = io.main_yao_2023(abc_cache, 
                  'WMB-10Xv2', 
                  ['WMB-10Xv2-Isocortex-3', 'WMB-10Xv2-Isocortex-1', 'WMB-10Xv2-Isocortex-4', 'WMB-10Xv2-Isocortex-2'], 
                  download_base,
                  taxa_class = ['01 IT-ET Glut', '06 CTX-CGE GABA', '07 CTX-MGE GABA',  '08 CNU-MGE GABA', '09 CNU-LGE GABA'],
                  region_of_interest = 'MO-FRP', 
                  neurotransmitters = ['GABA', 'Glut'], 
                  extract_csv = False)