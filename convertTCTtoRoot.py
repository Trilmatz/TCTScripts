import uproot
import numpy as np
import awkward as ak

# 1. Define your files and tree
input_file = "../data/CNM/W4/C18/TCT/2026_08_13_12_09_09_CNM_18399_W4_C18.txt.root"
output_file = "../data/CNM/W4/C18/TCT/2026_08_13_12_09_09_CNM_18399_W4_C18.root"

# 2. Define your branch renaming map
# Format is {"old_name": "new_name"}
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
    
    # 4. Create a new dictionary to hold the updated names
    new_data = {}
    
    # Loop through existing branch names (fields in the awkward array)
    for old_name in original_data.fields:
        if old_name in branch_map:
            # If the name is in our map, use the new name
            new_name = branch_map[old_name]
            new_data[new_name] = original_data[old_name]
        # else:
        #     # Otherwise, keep the original name
        #     new_data[old_name] = original_data[old_name]


    
    branch_types = {
        "event": np.uint64,
        "time": "var * float32",
        "ch1": "var * float32",
    }

    with uproot.recreate(output_file) as outfile:
        outfile.mktree("waves", branch_types)
        outfile["waves"].extend(new_data)

print(f"Successfully parsed into {output_file} with updated branches.")