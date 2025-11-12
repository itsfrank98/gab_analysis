import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from openpyxl.styles import Font
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from numpy import array


def create_dataframes(df, afl, leanings, model_name, dim, dst_dir, id_field="account_id"):
    for k in list(afl.keys()):
        if k.__contains__("duplicate"):
            afl.pop(k)
    lean_posts = []
    list_of_dfs = []
    for lean in leanings:
        users_belonging_to_lean = [k for k in list(afl.keys()) if afl[k] == lean]
        lean_posts.append((lean, users_belonging_to_lean))

    df = df.drop(columns=[k for k in df.columns if k not in [id_field, "posts_count", "content"]])
    df[id_field] = df[id_field].astype(str)
    for lean, x in lean_posts:
        df_lean = df[df[id_field].isin(x)]
        list_of_dfs.append((lean, df_lean))

    with pd.ExcelWriter(f"{dst_dir}/{model_name}_{dim}.xlsx") as writer:
        for lean, d in list_of_dfs:
            d.to_excel(writer, sheet_name=lean, index=False)

        for sheet in writer.sheets:
            worksheet = writer.sheets[sheet]
            # Set font size for all cells
            for row in worksheet.iter_rows():
                for cell in row:
                    cell.font = Font(size=14)

def plot_confusion_matrix(y_true, y_pred, model):
    print(precision_recall_fscore_support(y_true, y_pred, average="micro"))
    mat = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=sorted(set(y_true + y_pred)))
    plt.figure(figsize=(6, 4))
    ax = plt.subplot()
    sn.heatmap(mat, annot=True, cmap="CMRmap", linewidths=0.5, cbar=False, fmt="d", ax=ax,
               xticklabels=sorted(set(y_true + y_pred)), yticklabels=sorted(set(y_true + y_pred)))
    #'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    os.makedirs("matrices", exist_ok=True)
    plt.savefig(f"matrices/matrix_{model}.png")
    plt.show()