import pickle
import matplotlib.pyplot as plt
import numpy as np
import json

def save_to_pickle(name, c):
    with open(name, 'wb') as f:
        pickle.dump(c, f)


def load_from_pickle(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def plot_multiple_data(dicts, xlabels, legend, dst, type_plot="bar"):
    """
    Use this function for plotting multiple distributions
    :param dicts: list of dictionaries containing the distributions to plot
    :param xlabels: labels to put on the x-axis
    :param legend: List of values to display in the legend
    :param dst: destination file where the plot will be saved
    :return:
    """
    width = 1/len(dicts)
    dcs = []
    x = np.arange(len(xlabels))
    for el in dicts:
        c = list(el.values())
        dc = {v: c.count(v) for v in xlabels}
        dcs.append(dc)
    if type_plot == "line":
        for i in range(len(dcs)):
            plt.plot(x, list(dcs[i].values()), label=legend[i])
    elif type_plot == "bar":
        if len(dicts) == 2:
            width = .35
            plt.bar(x - width/2, list(dcs[0].values()), width, label=legend[0])
            plt.bar(x + width/2, list(dcs[1].values()), width, label=legend[1])
        elif len(dicts) == 3:
            plt.bar(x - width, list(dcs[0].values()), width, label=legend[0])
            plt.bar(x, list(dcs[1].values()), width, label=legend[1])
            plt.bar(x + width, list(dcs[2].values()), width, label=legend[2])
    plt.xticks(x, xlabels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig("{}.pdf".format(dst))
    plt.show()


if __name__ == "__main__":
    type_plot = "line"
    xlabels = ["far left", "left", "center", "right", "far right", "non political", "unknown"]
    dst = "synthetic_dataset/stance/plots/comparison_mistral_1500_{}".format(type_plot)
    legend = ["P1", "P2"]

    with open("synthetic_dataset/stance/affiliation_dicts/mistral_no_american_politics/affiliations_1500_mistral.json", 'r') as f:
        d_1 = json.load(f)
    with open("synthetic_dataset/stance/affiliation_dicts/affiliations_1500_mistral_newlabel.json", 'r') as f:
        d_2 = json.load(f)
    with open("synthetic_dataset/stance/affiliation_dicts/affiliations_4000_local-model_newlabel.json", 'r') as f:
        d_llama = json.load(f)

    dicts = [d_1, d_2]

    plot_multiple_data(dicts, xlabels=xlabels, legend=legend, dst=dst, type_plot=type_plot)
