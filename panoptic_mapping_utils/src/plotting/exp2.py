import os
import csv
import numpy as np
from matplotlib import pyplot as plt

# Params
keys = [
    'MeanGTError [m]', 'StdGTError [m]', 'GTRMSE [m]', 'TotalPoints [1]',
    'UnknownPoints [1]', 'TruncatedPoints [1]', 'GTInliers [1]',
    'MeanMapError [m]', 'StdMapError [m]', 'MapRMSE [m]', 'MapInliers [1]',
    'MapOutliers [1]'
]
output_dir = '/home/lukas/Documents/PanopticMapping/Results/exp2/'
output_name = 'exp2'

key = 0  # 9: Map RMSE, 13: Coverage
store_output = True
use_percentage = False

input_path = '/home/lukas/Documents/PanopticMapping/Exp2/'
input_dirs = [
    'SingleTsdf/run2_with_map', 'SingleTsdf/run2_no_map', 'longterm_pcl/10',
    'GT/conservative', 'Detectron/detectron'
]
legend = [
    'Monolithic with map', 'Monolithic no map', 'Long-term fusion',
    'Ours (ground truth)', 'Ours (detectron)'
]
labels = keys + ["Coverage [1]", "Coverage [%]"]  # [
#     'MeanError [m]', 'Coverage [1]', 'Correct Points [1]',
#     'Incorrect Points [1]'
# ]
styles = ['g-', 'g--', 'k-.', 'b-', 'b--']

# Read data
data = {}  # {dir: {field: values}}
for d in input_dirs:
    with open(os.path.join(input_path, d, 'evaluation_data.csv'),
              'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        fields = []
        values = []
        for row in csv_reader:
            if row[0] == "MeanGTError [m]":
                fields = row
                values = [[] for i in range(len(row))]
            else:
                for i, r in enumerate(row):
                    values[i].append(r)
        datum = {}
        for i, f in enumerate(fields):
            datum[f] = np.array(values[i], dtype=float)
        data[d] = datum

# Plot
plt.rcParams.update({'font.size': 12})
for i, d in enumerate(input_dirs):
    if key < 12:
        y = data[d][keys[key]]
    elif key == 12:
        # Coverage
        y = data[d]['TotalPoints [1]'] - data[d]['UnknownPoints [1]']
    elif key == 13:
        # Coverage with inliers
        y = data[d]['GTInliers [1]'] / data[d]['TotalPoints [1]'] * 100
    if use_percentage:
        y = y / 31165.62
        str_list = list(labels[key])
        str_list[-2] = "%"
        labels[key] = "".join(str_list)
    x = np.arange(len(y)) * 10
    plt.plot(x, y, styles[i])

# Axes
plt.xlabel("Frame Number")
plt.ylabel(labels[key])

# Legend
plt.legend(legend)
plt.tight_layout()

# Save
if store_output:
    output_path = os.path.join(
        output_dir, output_name + "_" + labels[key].split(" ", 1)[0])
    if key in [1, 2, 3] and use_percentage:
        output_path += "_perc"
    output_path += ".jpg"
    plt.savefig(output_path)
plt.show()