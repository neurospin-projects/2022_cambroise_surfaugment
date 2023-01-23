import json
import argparse
import re
import os
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str)

parser.add_argument("--regex", type=str, default="")

parser.add_argument("--evaluate", action="store_true")

args = parser.parse_args()

metrics = ["real_mae", "r2", "real_rmse", "correlation"]
best_values = {metric: {} for metric in metrics}
best_epochs = {metric: {} for metric in metrics}
best_stds = {metric: {} for metric in metrics}
for directory in os.listdir(args.path):
    path_to_dir = os.path.join(args.path, directory)
    matches_regex = False if re.search(args.regex, directory) is None else True
    if os.path.isdir(path_to_dir) and matches_regex:
        path_to_res = os.path.join(path_to_dir, 'best_values.json')
        path_to_epochs = os.path.join(path_to_dir, 'best_epochs.json')
        path_to_stds = os.path.join(path_to_dir, 'best_stds.json')
        if args.evaluate:
            path_to_res = os.path.join(path_to_dir, 'final_values.json')
            path_to_stds = os.path.join(path_to_dir, 'final_stds.json')
        if os.path.isfile(path_to_res):
            with open(path_to_res, 'r') as fp:
                best_values_per_metric = json.load(fp)
            if not args.evaluate:
                with open(path_to_epochs, 'r') as fp:
                    best_epochs_per_metric = json.load(fp)
            with open(path_to_stds, 'r') as fp:
                best_stds_per_metric = json.load(fp)
            
            for metric in metrics:
                if not args.evaluate:
                    best_values[metric][directory] = best_values_per_metric[metric][0]
                else:
                    best_values[metric][directory] = best_values_per_metric[metric]

            if not args.evaluate:
                for metric in metrics:
                    best_epochs[metric][directory] = best_epochs_per_metric[metric][0]

            for metric in metrics:
                if not args.evaluate:
                    best_stds[metric][directory] = best_stds_per_metric[metric][0]
                else:
                    best_stds[metric][directory] = best_stds_per_metric[metric]

the_lower_the_better = ["real_mae", "real_rmse"]
res_dict = {"path": []}
for i, metric in enumerate(metrics):
    res_dict["{}_avg".format(metric)] = []
    res_dict["{}_std".format(metric)] = []
    if not args.evaluate:
        res_dict["{}_best_epoch".format(metric)] = []
    for path in best_values[metric]:
        if i == 0:
            res_dict["path"].append(path)
        res_dict["{}_avg".format(metric)].append(best_values[metric][path])
        res_dict["{}_std".format(metric)].append(best_stds[metric][path])
        if not args.evaluate:
            res_dict["{}_best_epoch".format(metric)].append(best_epochs[metric][path])

results = pd.DataFrame(res_dict)
sorted_res_by_mae = results.sort_values(by="real_mae_avg").reset_index(drop=True)
sorted_res_by_r2 = results.sort_values(by="r2_avg", ascending=False).reset_index(drop=True)

print(results.shape)
if not args.evaluate:
    print(sorted_res_by_mae[["real_mae_avg", "real_mae_std", "real_mae_best_epoch"]].head(10))
else:
    print(sorted_res_by_mae[["real_mae_avg", "real_mae_std"]].head(10))
for i in range(min(10, len(results))):
    print(sorted_res_by_mae.iloc[i, 0])
if not args.evaluate:
    print(sorted_res_by_r2[["r2_avg", "r2_std", "r2_best_epoch"]].head(10))
else:
    print(sorted_res_by_r2[["r2_avg", "r2_std"]].head(10))
print(sorted_res_by_mae.iloc[0, 0])
    # sorted_values = list(sorted(x.items(), key=lambda item: item[1], reverse=metric not in the_lower_the_better))
    # print(metric)
    # print(sorted_values)
    # print([epochs[value[0]] for value in sorted_values])