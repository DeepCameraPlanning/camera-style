import argparse
import os.path as osp

from enum import Enum, auto
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

from utils.diverse_utils import divide
from utils.file_utils import save_pickle

STYLE_LABELS = [
    "boom",
    "cras",
    "pan_",
    "pull",
    "push",
    "roll",
    "stat",
    "tilt",
    "truc",
    "zoom",
]
STYLE_LABELS_DIRECTION = [
    "boom_up",
    "boom_down",
    "cras_in",
    "pan__left",
    "pan__right",
    "pull_out",
    "push_in",
    "roll_anti",
    "roll_clock",
    "stat_none",
    "tilt_up",
    "tilt_down",
    "truc_left",
    "truc_right",
    "zoom_in",
    "zoom_out",
]
DIRECTION_LABELS = [
    "in",
    "out",
    "left",
    "right",
    "up",
    "down",
    "anti",
    "clock",
    "none",
]


class mode(Enum):
    LABEL = auto()
    DIRECTION = auto()
    LABEL_DIRECTION = auto()
    DEFAULT = auto()


class statsSim:
    dictData = {}
    lutfname = {}
    linlier = []
    loutlier = []
    cmode = mode.DEFAULT

    def __init__(self) -> None:
        self.dictData.clear()
        self.lutfname.clear()
        self.linlier = []
        self.loutlier = []
        pass

    def convertStyleName(self, styleName, fname, lut):
        if self.cmode == mode.LABEL_DIRECTION:
            return styleName + "_" + lut[fname]  # all same
        elif self.cmode == mode.LABEL:
            return styleName
        elif self.cmode == mode.DIRECTION:
            return lut[fname]

    def add(
        self,
        styleName: str,
        foundStyle: str,
        fname: str,
        fdfname: str,
        isN: int,
        dist: float,
    ):
        cvrtName = self.convertStyleName(styleName, fname, self.lutfname)
        # print(cvrtName)
        if cvrtName in self.dictData:
            self.dictData[cvrtName]["isN"].append(isN)
            self.dictData[cvrtName]["foundstyle"].append(
                self.convertStyleName(foundStyle, fdfname, self.lutfname)
            )
            self.dictData[cvrtName]["dist"].append(dist)
        else:
            self.dictData[cvrtName] = {
                "isN": [isN],
                "foundstyle": [
                    self.convertStyleName(foundStyle, fdfname, self.lutfname)
                ],
                "dist": [dist],
            }

    def stats(self, clabels, isnormalize=True):
        lactual = []
        lpred = []
        # Computing Dist/Quant
        dim = len(clabels)
        mDist = np.zeros((dim, dim))
        mQuant = np.zeros((dim, dim))

        for key in self.dictData.keys():
            lfdStyle = self.dictData[key]["foundstyle"]
            ldist = self.dictData[key]["dist"]
            lisN = self.dictData[key]["isN"]
            idxkey = clabels.index(key)
            # print("act: ", key)
            for idx, elem in enumerate(lfdStyle):
                if lisN[idx]:
                    lactual.append(key)
                    lpred.append(elem)
                mDist[idxkey][clabels.index(elem)] += ldist[idx]
                mQuant[idxkey][clabels.index(elem)] += 1

                if key == elem:
                    self.linlier.append(ldist[idx])
                else:
                    self.loutlier.append(ldist[idx])
                # print("pred: ", elem)
        allstats = {}
        cmpType = ["micro", "macro", "weighted"]
        allstats["acc"] = accuracy_score(lactual, lpred)

        for cidx, ctype in enumerate(cmpType):
            prc = precision_score(
                lactual, lpred, average=ctype, zero_division=0
            )
            rec = recall_score(lactual, lpred, average=ctype, zero_division=0)
            f1 = f1_score(lactual, lpred, average=ctype, zero_division=0)
            allstats[ctype + "_prec"] = prc
            allstats[ctype + "_rec"] = rec
            allstats[ctype + "_f1"] = f1

        # Computing ConfMat
        cm_real = confusion_matrix(lactual, lpred)
        distM = divide(mDist, mQuant)

        if isnormalize:
            cm_real = (
                cm_real.astype("float") / cm_real.sum(axis=1)[:, np.newaxis]
            )
            distM = minmax_scale(distM)

            return cm_real, distM, allstats


def LoadLUTDir(dir):
    lut = pd.read_csv(dir, delimiter=";", header=None)
    return dict(zip(lut[0], lut[1]))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for the video frame averaging tool."
    )
    parser.add_argument(
        "MSfeatures",
        type=str,
        help="The features as pickle files of the motionset",
    )
    parser.add_argument(
        "outputDir",
        type=str,
        help="output dir",
    )
    parser.add_argument(
        "lutDir",
        type=str,
        help="lut of fname to type e.g: up down etc. ",
    )
    parser.add_argument(
        "--topN",
        "-t",
        default=3,
        type=int,
        help="top N similar when computing confusion matrix",
    )
    parser.add_argument(
        "--print",
        "-p",
        action="store_true",
        help="Wether to print scores or not",
    )
    parser.add_argument(
        "--save",
        "-s",
        action="store_true",
        help="Wether to save scores or not",
    )

    return parser.parse_args()


def main(cmode, args):
    stats = statsSim()
    stats.cmode = cmode
    stats.lutfname = LoadLUTDir(args.lutDir)
    with open(args.MSfeatures, "rb") as f:
        fMS = pickle.load(f)
    basenamepk = osp.splitext(osp.basename(args.MSfeatures))[0]

    for idxMS in tqdm(range(len(fMS))):
        msKey = list(fMS)[idxMS]
        msVal = list(fMS.values())[idxMS]
        styleName = msKey.rpartition("/")[0]
        typename = styleName[:4]

        lDist = []
        for idxS, x in enumerate(fMS.values()):
            lDist.append(
                np.linalg.norm(x.flatten().numpy() - msVal.flatten().numpy())
            )
        lSimDisim = [np.argsort(lDist)[:]]
        for lTopNArg in lSimDisim:
            for ctr, sidx in enumerate(lTopNArg[1:]):
                foundKeys = list(fMS.keys())[sidx]
                foundstyleName = foundKeys.rpartition("/")[0]
                fdtypename = foundstyleName[:4]
                if ctr < args.topN:
                    stats.add(
                        typename,
                        fdtypename,
                        styleName,
                        foundstyleName,
                        True,
                        lDist[sidx],
                    )
                else:
                    stats.add(
                        typename,
                        fdtypename,
                        styleName,
                        foundstyleName,
                        False,
                        lDist[sidx],
                    )

    if cmode == mode.LABEL_DIRECTION:
        xt = STYLE_LABELS_DIRECTION
    elif cmode == mode.DIRECTION:
        xt = DIRECTION_LABELS
    elif cmode == mode.LABEL:
        xt = STYLE_LABELS

    ConfMat, DistMat, allstats = stats.stats(xt)

    outputdict = {}
    outputdict["CM"] = ConfMat
    outputdict["Dist"] = DistMat
    outputdict["inlier"] = np.array(stats.linlier)
    outputdict["outlier"] = np.array(stats.loutlier)
    outputdict["labels"] = xt
    outputdict["allstats"] = allstats
    outputdict["basename"] = basenamepk

    return outputdict


if __name__ == "__main__":
    args = parse_args()

    allDict = {}
    allDict["label"] = main(mode.LABEL, args)
    allDict["puredir"] = main(mode.DIRECTION, args)
    allDict["withDir"] = main(mode.LABEL_DIRECTION, args)

    if args.print:
        for stat_mode in allDict:
            print(f"{stat_mode}:")
            for name, value in allDict[stat_mode]["allstats"].items():
                print(f"    + {name:13} = {value:.2f}")
            print()

    if args.save:
        stats_path = osp.join(
            args.outputDir, allDict["label"]["basename"] + ".pk"
        )
        save_pickle(allDict, stats_path)
        print(f"stats saved at {stats_path}")
