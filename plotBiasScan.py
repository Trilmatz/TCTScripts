import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse

# 1. Open the explicitly named ROOT file
parser = argparse.ArgumentParser(description="Plot waveforms of bias scan from TCT ROOT file.")
parser.add_argument("input_file", help="Path to the input .root file")
args = parser.parse_args()

# 1. Open the ROOT file
file_name = args.input_file
print(f"Loading data from: {file_name}")
root_file = uproot.open(file_name)
tree = root_file["ch0"]

volt_data = -1000 * ak.to_numpy(tree["volt"].array())
time_data = ak.to_numpy(tree["time"].array())
i_max = np.argmax(volt_data, axis=1)[-1]
time_data = time_data - time_data[0,i_max]

bias_data = np.abs(ak.to_numpy(tree["Vbias"].array()))

cmap = plt.get_cmap('viridis_r')
norm = plt.Normalize(vmin=np.min(bias_data), vmax=np.max(bias_data))

plt.figure(figsize=(8, 6))

for i, bias in enumerate(bias_data):
    color = cmap(norm(bias))
    plt.plot(time_data[i], volt_data[i], color=color, label=f"{int(bias)}V")

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([]) # Required for matplotlib < 3.1
# cbar = plt.colorbar(sm, ax=plt.gca())
# cbar.set_label('Bias Voltage [V]', rotation=270, labelpad=15)

plt.xlabel('Time [ns]')
plt.ylabel('Amplitude [mV]')
plt.legend(loc="upper right")
plt.xlim(-2, 6)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"../plots/CNM_W4_H21/TCT/bias_scan.pdf")
plt.show()


# python3 TCTScripts/plotBiasScan.py data/CNM/W4/C18/TCT/2026_08_12_15_54_40_CNM_18399_W4_C18.txt.root