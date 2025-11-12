import pickle
import matplotlib.pyplot as plt
import numpy as np
import json
import os
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
            width = .2
            plt.bar(x - width, list(dcs[0].values()), width, label=legend[0])
            plt.bar(x, list(dcs[1].values()), width, label=legend[1])
            plt.bar(x + width, list(dcs[2].values()), width, label=legend[2])
    plt.xticks(x, xlabels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig("{}.pdf".format(dst))
    plt.show()


def plot_different_distributions(d1, d2, political_leanings, legend):
    x = np.arange(len(political_leanings))
    width = .35
    c1 = list(d1.values())
    c2 = list(d2.values())
    dc1 = {v: c1.count(v) for v in political_leanings}
    dc2 = {v: c2.count(v) for v in political_leanings}
    lc1 = np.array(list(dc1.values()))
    lc2 = np.array(list(dc2.values()))

    percentages1 = lc1 / sum(lc1) * 100
    percentages2 = lc2 / sum(lc2) * 100
    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, list(percentages1), width=width, label=legend[0])
    bars2 = ax.bar(x + width/2, list(percentages2), width=width, label=legend[1])

    for bar, pct in zip(bars1, percentages1):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # x position
            height,  # y position (top of bar)
            f'{pct:.1f}%',  # formatted percentage
            ha='center', va='bottom', fontsize=8, fontweight='bold'
        )
    for bar, pct in zip(bars2, percentages2):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # x position
            height,  # y position (top of bar)
            f'{pct:.1f}%',  # formatted percentage
            ha='center', va='bottom', fontsize=7, fontweight='bold'
        )
    plt.xticks(x, political_leanings, rotation=45, ha='right')
    plt.yticks([])

    plt.legend()
    os.makedirs('plots', exist_ok=True)
    plt.tight_layout()
    plt.savefig('plots/posts_vs_bios.svg')
    plt.show()


def plot_n_distributions(ld, political_leanings, legend, width, dst):
    x = np.arange(len(political_leanings))
    cl = []
    dcl = []
    lc = []

    percentages = []
    for l in ld:
        lv = list(l.values())
        dc = {v: lv.count(v) for v in political_leanings}
        dc_array = np.array(list(dc.values()))
        cl.append(lv)
        dcl.append(dc)
        lc.append(dc_array)
        percentages.append((dc_array / sum(dc_array)) * 100)

    fig, ax = plt.subplots()
    for i in range(len(percentages)):
        offset = (i - len(percentages) / 2 + 0.5) * (width / len(percentages))
        bars = ax.bar(x + offset, list(percentages[i]), width=width/len(percentages), label=legend[i])
        for bar, pct in zip(bars, percentages[i]):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / len(ld),  # x position
                height,  # y position (top of bar)
                f'{pct:.1f}%',  # formatted percentage
                ha='center', va='bottom', fontsize=8, fontweight='bold'
            )
    plt.xticks(x, political_leanings, rotation=45, ha='right')
    plt.yticks([])
    plt.legend()
    if not dst.endswith(".svg"):
        dst += ".svg"
    plt.tight_layout()
    plt.savefig(dst)
    plt.show()


if __name__ == "__main__":
    type_plot = "line"
    dim = 1500
    bios_src = "synthetic_dataset/bios/afl.json"
    mistral_src = "synthetic_dataset/stance/affiliation_dicts/on_gab_dataset/affiliations_complete_mistral.json"
    llama_src = "synthetic_dataset/stance/affiliation_dicts/on_gab_dataset/affiliations_complete_llama.json"
    gpt_src = "synthetic_dataset/stance/affiliation_dicts/on_gab_dataset/affiliations_complete_gpt.json"
    xlabels = ["far-left", "left", "center", "right", "far-right", "non-political", "unknown"]

    dst = "synthetic_dataset/stance/plots/bios_vs_gab.svg"
    legend = ["Bios (Mistral)", "Mistral", "GPT", "Llama"]

    ld = []
    for s in [bios_src, mistral_src, llama_src, gpt_src]:
        with open(s, "r") as f:
            ld.append(json.load(f))

    plot_n_distributions(ld=ld, political_leanings=xlabels, legend=legend, dst=dst, width=.8)

