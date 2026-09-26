import argparse
import os

import uproot
import numpy as np
import awkward as ak

parser = argparse.ArgumentParser(description="Convert an edge_tree TCT file into a waves TTree.")
parser.add_argument("input_file", help="Input file ending in .txt.root")
args = parser.parse_args()

input_file = args.input_file
if not input_file.endswith(".txt.root"):
    parser.error(f"Input file must end in .txt.root: {input_file}")
output_file = input_file[:-len(".txt.root")] + ".root"
if not os.path.isfile(input_file):
    parser.error(f"Input file not found: {input_file}")

scalar_branches = {
    "x": "raw/x",
    "y": "raw/y",
    "z": "raw/z",
    "Vbias": "raw/Vbias",
    "Itot": "raw/Itot",
    "Temp": "raw/Temp",
    "LPower": "proc/LPower",
    "LPower2": "proc/LPower2",
    "LNph": "proc/LNph",
    "PVOA": "proc/PVOA",
}

with uproot.open(input_file) as infile:
    tree = infile["ch0"]
    event_data = tree["raw/event"].array(library="np")
    volt_np = ak.to_numpy(tree["raw/volt"].array(library="ak"))
    time_np = ak.to_numpy(tree["raw/time"].array(library="ak"))
    scalar_data = {name: tree[path].array(library="np") for name, path in scalar_branches.items()}

ch1_data, ch2_data = np.split(volt_np, 2, axis=1)
time_data, _ = np.split(time_np, 2, axis=1)


def to_jagged_float32(arr):
    return ak.from_regular(ak.Array(np.asarray(arr, dtype=np.float32)))


output_dict = {
    "event": np.asarray(event_data, dtype=np.uint64),
    "time": to_jagged_float32(time_data),
    "ch1": to_jagged_float32(ch1_data),
    "ch2": to_jagged_float32(ch2_data),
}
branch_types = {
    "event": np.uint64,
    "time": "var * float32",
    "ch1": "var * float32",
    "ch2": "var * float32",
}
for name, values in scalar_data.items():
    output_dict[name] = np.asarray(values, dtype=np.float64)
    branch_types[name] = np.float64

with uproot.recreate(output_file) as outfile:
    outfile.mktree("waves", branch_types)
    outfile["waves"].extend(output_dict)

print(f"Successfully parsed into {output_file} with updated branches.")