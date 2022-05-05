import argparse
import os
import os.path as osp

from plotly.subplots import make_subplots
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go

from utils.file_utils import load_pickle


def nameonFig(basename):
    return (
        basename.replace(
            "MotionSet_autoencoder-mse-reduced-7x7_2058-", "MotionSet"
        )
        .replace("MotionSet", "")
        .replace("external_", "")
        .replace(".json", "")
        .replace("autoencoder-mse", "")
        .replace("-224-", "")
        .replace("-mse-", "")
        .replace("autoencoder-", "")
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for the video frame averaging tool."
    )
    parser.add_argument(
        "stats_dir",
        type=str,
        help="Directory containing pre-computed scores.",
    )

    return parser.parse_args().stats_dir


def main(label, stats_dir):
    _alldata = os.listdir(stats_dir)

    alldataDict = [load_pickle(osp.join(stats_dir, i)) for i in _alldata]
    titles = [nameonFig(i[label]["basename"]) for i in alldataDict]

    figImg = make_subplots(
        rows=2, cols=len(alldataDict), subplot_titles=titles
    )
    figStats = make_subplots(
        rows=2, cols=len(alldataDict), subplot_titles=titles
    )
    figLine = make_subplots(rows=1, cols=1)
    for idx, dataDict in enumerate(alldataDict):

        # Img figure
        xt = dataDict[label]["labels"]
        # print("idx: ", idx)
        figImg.add_trace(
            px.imshow(
                np.around(dataDict[label]["CM"], 2), text_auto=True, x=xt, y=xt
            ).data[0],
            row=1,
            col=1 + idx,
        )
        basenameCM = osp.basename(dataDict[label]["basename"])
        figImg.update_xaxes(
            row=1, col=1 + idx, showgrid=False, scaleanchor="y", scaleratio=1
        )
        figImg.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
            row=1,
            col=1 + idx,
            autorange="reversed",
            showgrid=False,
        )

        figImg.add_trace(
            px.imshow(
                np.around(dataDict[label]["Dist"], 2),
                text_auto=True,
                x=xt,
                y=xt,
            ).data[0],
            row=2,
            col=1 + idx,
        )

        figImg.update_xaxes(
            row=2, col=1 + idx, showgrid=False, scaleanchor="y", scaleratio=1
        )
        figImg.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
            row=2,
            col=1 + idx,
            autorange="reversed",
            showgrid=False,
        )

        statdict = dataDict[label]["allstats"]

        # stats figure
        figStats.add_trace(
            go.Bar(
                x=list(statdict.keys()),
                y=list(statdict.values()),
                text=np.around(list(statdict.values()), 2),
            ),
            row=1,
            col=1 + idx,
        )
        figStats.update_yaxes(range=[0, 1], row=1, col=1 + idx)

        group_labels = ["inlier", "outlier"]
        hist_data = [dataDict[label]["inlier"], dataDict[label]["outlier"]]

        figff = ff.create_distplot(
            hist_data,
            group_labels,
            bin_size=(np.max(dataDict[label]["outlier"]) / 20.0),
        )

        figStats.add_trace(
            go.Histogram(figff["data"][0]),
            row=2,
            col=1 + idx,
        )

        figStats.add_trace(
            go.Histogram(figff["data"][1]),
            row=2,
            col=1 + idx,
        )

        figStats.add_trace(
            go.Scatter(figff["data"][2]),
            row=2,
            col=1 + idx,
        )

        figStats.add_trace(
            go.Scatter(figff["data"][3]),
            row=2,
            col=1 + idx,
        )
        figLine.add_trace(
            go.Scatter(
                x=list(statdict.keys()),
                y=list(statdict.values()),
                name=nameonFig(basenameCM),
            ),
            row=1,
            col=1,
        )

    figImg.show()
    figStats.show()
    figLine.show()


if __name__ == "__main__":
    stats_dir = parse_args()
    main("label", stats_dir)
    main("puredir", stats_dir)
    main("withDir", stats_dir)
