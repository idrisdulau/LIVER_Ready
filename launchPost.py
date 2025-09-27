import os
import subprocess


### COMPARE

for DB in [str(e) for e in range(18,38)]:
    for MODE in ["_pred","_close","_pred_one","_close_one",]:
        gtPath = os.path.join("_datasets","stacks","targetSplitted","test",DB)
        predPath = os.path.join("post",DB+MODE)
        subprocess.run([
            "python3", os.path.join("_run","compare.py"),\
                gtPath,\
                predPath,\
                os.path.join("stats"),\
                "average",\
                "CSV",\
                "post"+MODE,\
                predPath.split("/")[-2]+" "+predPath.split("/")[-1]
        ])
