import uproot
import awkward as ak
import numpy as np
from pathlib import Path
import argparse

def parse_tct_ascii_to_root(input_file, is_timing=False):
    input_path = Path(input_file)
    output_file = input_path.with_name(f"{input_path.stem}_waveforms.root")
    
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_file}")

    mode_str = "TIMING MODE (Split CH1)" if is_timing else "STANDARD MODE (Dual Channel)"
    print(f"Reading TCT ASCII data from: {input_file} | {mode_str}")

    # ==========================================
    # 1. PARSE THE HEADER
    # ==========================================
    record_length = None
    sampling_period = None
    header_passed = False
    
    with open(input_path, 'r') as f:
        lines = f.readlines()
        
    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx].strip()
        
        if line.startswith("RecordLength:"):
            record_length = int(line.split(":")[1].strip())
        elif line.startswith("SamplingPeriod[s]:"):
            sampling_period = float(line.split(":")[1].strip())
            
        if line.startswith("timestamp Vset[V]"):
            line_idx += 2
            header_passed = True
            break
            
        line_idx += 1

    if not header_passed or record_length is None or sampling_period is None:
        raise ValueError("Could not parse the header. Missing RecordLength, SamplingPeriod, or data start marker.")
        
    print(f"Header parsed -> RecordLength: {record_length} points, SamplingPeriod: {sampling_period} s")

    # ==========================================
    # 2. EXTRACT AND RESTRUCTURE THE DATA
    # ==========================================
    remaining_text = " ".join(lines[line_idx:])
    tokens = remaining_text.split()
    
    # The file physically ALWAYS has 2 channels worth of tokens per event block
    physical_channels_in_file = 2 
    tokens_per_event = 9 + (record_length * physical_channels_in_file)
    total_events = len(tokens) // tokens_per_event
    
    print(f"Identified tokens for {total_events} intact events. Processing...")

    # Calculate Time Axis based on the mode
    if is_timing:
        pts_per_channel = record_length // 2
    else:
        pts_per_channel = record_length
        
    time_axis = (np.arange(pts_per_channel) * sampling_period).astype(np.float32)
    
    ch_data = {"ch1": [], "ch2": []}
    time_data = []
    event_data = []
    x_data = []  
    y_data = []  

    idx = 0
    event_num = 0
    
    while idx + tokens_per_event <= len(tokens):
        # Extract X and Y metadata 
        x_data.append(float(tokens[idx + 3]))
        y_data.append(float(tokens[idx + 4]))

        # Read the raw CH1 data from the file
        ch1_start = idx + 9
        ch1_end = ch1_start + record_length
        raw_ch1 = np.array(tokens[ch1_start:ch1_end], dtype=np.float32)

        if is_timing:
            # Timing Mode: Split physical CH1 in half, ignore physical CH2
            ch_data["ch1"].append(raw_ch1[:pts_per_channel])
            ch_data["ch2"].append(raw_ch1[pts_per_channel:])
        else:
            # Standard Mode: Keep physical CH1, extract physical CH2
            ch2_start = ch1_end
            ch2_end = ch2_start + record_length
            raw_ch2 = np.array(tokens[ch2_start:ch2_end], dtype=np.float32)
            
            ch_data["ch1"].append(raw_ch1)
            ch_data["ch2"].append(raw_ch2)
        
        time_data.append(time_axis)
        event_data.append(event_num)
        
        idx += tokens_per_event
        event_num += 1

    print(f"Successfully extracted {event_num} events.")

    # ==========================================
    # 3. WRITE DIRECTLY TO ROOT
    # ==========================================
    output_dict = {
        "event": np.array(event_data, dtype=np.uint64),
        "x": np.array(x_data, dtype=np.float32),      
        "y": np.array(y_data, dtype=np.float32),      
        "time": ak.Array(time_data),
        "ch1": ak.Array(ch_data["ch1"]),
        "ch2": ak.Array(ch_data["ch2"]),
        "ch3": ak.Array(ch_data["ch2"]),
    }
    
    branch_types = {
        "event": np.uint64,
        "x": np.float32,
        "y": np.float32,
        "time": "var * float32",
        "ch1": "var * float32",
        "ch2": "var * float32",
        "ch3": "var * float32"
    }

    with uproot.recreate(output_file) as outfile:
        outfile.mktree("waves", branch_types)
        outfile["waves"].extend(output_dict)
        
    print(f"Data successfully written to {output_file} as a standard TTree named 'waves'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TCT ASCII files to ROOT TTrees.")
    parser.add_argument("input_file", help="Path to the .txt file")
    
    # Adding the --timing flag
    parser.add_argument("--timing", action="store_true", help="Split CH1 in half to create CH1 and CH2. Ignores raw file CH2.")
    
    args = parser.parse_args()
    parse_tct_ascii_to_root(args.input_file, is_timing=args.timing)

# Example: python3 TCTScripts/convertTXTtoRoot.py data/CNM/W4/C18/TCT/2026_08_13_10_48_10_CNM_18399_W4_C18.txt --timing