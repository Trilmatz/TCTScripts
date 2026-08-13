import uproot
import awkward as ak
import numpy as np
from pathlib import Path

def parse_tct_ascii_to_root(input_file):
    input_path = Path(input_file)
    output_file = input_path.with_name(f"{input_path.stem}_waveforms.root")
    
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_file}")

    print(f"Reading TCT ASCII data from: {input_file}")

    # ==========================================
    # 0. CONFIGURATION
    # ==========================================
    num_channels = 2  

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
    
    tokens_per_event = 9 + (record_length * num_channels)
    total_events = len(tokens) // tokens_per_event
    
    print(f"Identified tokens for {total_events} intact events with {num_channels} channels. Processing...")

    # FIX 1: Force the time axis to be 32-bit floats
    time_axis = (np.arange(record_length) * sampling_period).astype(np.float32)
    
    ch_data = {f"ch{i+1}": [] for i in range(num_channels)}
    time_data = []
    event_data = []
    x_data = []  
    y_data = []  

    idx = 0
    event_num = 0
    
    while idx + tokens_per_event <= len(tokens):
        x_data.append(float(tokens[idx + 3]))
        y_data.append(float(tokens[idx + 4]))

        for i in range(num_channels):
            start = idx + 9 + (i * record_length)
            end = start + record_length
            
            # FIX 2: Convert the text tokens directly into a 32-bit NumPy array
            waveform = np.array(tokens[start:end], dtype=np.float32)
            ch_data[f"ch{i+1}"].append(waveform)
        
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
        "time": ak.Array(time_data)
    }
    
    # FIX 3: Explicitly define the branches as "var * float32" instead of float64
    branch_types = {
        "event": np.uint64,
        "x": np.float32,
        "y": np.float32,
        "time": "var * float32"
    }

    for i in range(num_channels):
        ch_name = f"ch{i+1}"
        output_dict[ch_name] = ak.Array(ch_data[ch_name])
        branch_types[ch_name] = "var * float32"

    with uproot.recreate(output_file) as outfile:
        outfile.mktree("waves", branch_types)
        outfile["waves"].extend(output_dict)
        
    print(f"Data successfully written to {output_file} as a standard TTree named 'waves'.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert Multi-Channel TCT ASCII files to ROOT TTrees.")
    parser.add_argument("input_file", help="Path to the .txt file")
    args = parser.parse_args()
    parse_tct_ascii_to_root(args.input_file)

    # Example: python3 TCTScripts/convertTXTtoRoot.py data/CNM/W4/C18/TCT/2026_08_13_10_48_10_CNM_18399_W4_C18.txt