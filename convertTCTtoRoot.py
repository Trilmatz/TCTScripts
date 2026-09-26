import uproot
import numpy as np
import awkward as ak

input_file = "../data/CNM/W4/H21/TCT/2026_09_25_16_17_52_CNM_18399_W4_H21.txt.root"
output_file = "../data/CNM/W4/H21/TCT/2026_09_25_16_17_52_CNM_18399_W4_H21.root"

branch_map = {
    "event": "event",
    "volt": "ch1",
    "time": "time",
}

# 3. Open the input file and extract the data
with uproot.open(input_file) as infile:
    tree = infile["ch0"]["raw"]
    # Read all branches into an awkward array record
    original_data = tree.arrays(library="ak")

event_data = original_data["event"]
x_data = original_data["x"]
y_data = original_data["y"]
bias_data = original_data["Vbias"]
volt_np = ak.to_numpy(original_data["volt"])
time_np = ak.to_numpy(original_data["time"])

ch1_data, ch2_data = np.split(volt_np, 2, axis=1)
time_data, _ = np.split(time_np, 2, axis=1)


def to_jagged_float32(arr):
    return ak.from_regular(ak.Array(np.asarray(arr, dtype=np.float32)))


output_dict = {
    "event": np.asarray(event_data, dtype=np.uint64),
    "time": to_jagged_float32(time_data),
    "ch1": to_jagged_float32(ch1_data),
    "ch2": to_jagged_float32(ch2_data),
    "ch3": to_jagged_float32(ch2_data),
}
branch_types = {
    "event": np.uint64,
    "time": "var * float32",
    "ch1": "var * float32",
    "ch2": "var * float32",
    "ch3": "var * float32",
}

with uproot.recreate(output_file) as outfile:
    outfile.mktree("waves", branch_types)
    outfile["waves"].extend(output_dict)

print(f"Successfully parsed into {output_file} with updated branches.")
# python3 convertTCTtoRoot.py