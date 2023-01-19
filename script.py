import os
import time 

# FORMAT of EXPERIMENTS:
# dataset
# name
# part
# swapped
# re_enter
# noisy
#  save_logs

data_transformations = ["normal", "clust_centroids", "enn", "ros", 
                        "rus", "smote", "smoteenn", "adasyn", "borderlinesmote", 
                        "kmeanssmote", "tomeklinks", "smotetomek"]

swapped = [0.15]
datasets = ["Ozone", "adult", "credit", "mnist"]
parts = [1]
re_enter = [1]
noisy = [1]
save_logs = 1


for d in datasets:
    for n in data_transformations:
        for part in range(parts):
            for sw in swapped:
                for re in re_enter:
                    for noise in noisy:
                        start = time.time()

                        os.system(f"python run_sgd.py {d} {n} {str(part)} {str(sw)} {str(re)} {str(noise)} {str(save_logs)}")
                        end = time.time()
                        times = end - start
                        print(times)


