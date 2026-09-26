import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from uncertainties import unumpy, ufloat_fromstr

def plot_charges(prefix, temperature=20):
    Q_pin = 0.5
    for sensor in ["C18", "H21"]:
        filepath=f"../plots/{prefix}_{sensor}/charges.csv"
        charges_df = pd.read_csv(filepath, index_col=0)
        plt.plot(charges_df.index, charges_df["1"] / Q_pin, 'o', label=f"Beta {prefix}_{sensor}")

    filepath=f"../plots/{prefix}_H21/TCT/stats_2026_09_26_14_13_44_CNM_18399_W4_H21_charges.csv"
    tct_charges_df = pd.read_csv(filepath, index_col=0)

    columns = tct_charges_df.columns
    for column in columns:
        tct_charges_df[column] = tct_charges_df[column].map(ufloat_fromstr)
        plt.plot(tct_charges_df.index, unumpy.nominal_values(tct_charges_df[column]) / Q_pin, 's', label=f"TCT {prefix}_H21")


    plt.ylim(0, 90)
    plt.xlabel("Bias [V]")
    plt.ylabel(fr"Gain")
    plt.legend(title=rf"$T = {temperature}\,$°C")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../plots/gain_beta_TCT_CNM_W4.pdf")
    plt.close()


if __name__ == "__main__":
    plot_charges("CNM_W4", temperature=20)