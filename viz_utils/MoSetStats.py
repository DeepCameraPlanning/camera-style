import os
import os.path
# import matplotlib.pyplot as plt
import pandas as pd
import sys
import re
import difflib
import pickle
import numpy as np
import heapq
import random
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import plotly.express as px
import plotly
from sklearn.preprocessing import minmax_scale
import plotly.graph_objects as go
import plotly.figure_factory as ff
from enum import Enum, auto


# modes
# XYZRS=False
# DIRECTION=False
# ONLYDIR = False

class mode(Enum):
  LABEL           = auto()
  DIRECTION       = auto()
  LABEL_DIRECTION = auto()
  DEFAULT         = auto()
    
LUTdir = "/Users/triocrossing/Scripts/SimilarFinding/directions.csv"

styleLabels    = ["boom","cras","pan_","pull","push","roll","stat","tilt","truc","zoom"] 
styleLabelsDir = ["boom_up","boom_down","cras_in","pan__left","pan__right","pull_out","push_in","roll_anti","roll_clock","stat_none","tilt_up","tilt_down","truc_left","truc_right","zoom_in","zoom_out"] 
dirLabels = ["in", "out", "left", "right", "up", "down", "anti", "clock", "none"]
    
class statsSim:
    dictData = {}
    lutfname = {}
    linlier  = []
    loutlier = []
    cmode    = mode.DEFAULT
    def __init__(self) -> None:
        self.dictData.clear()
        self.lutfname.clear()
        self.linlier=[]
        self.loutlier=[]
        pass
    
    def convertStyleName(self, styleName, fname, lut):  
      if self.cmode == mode.LABEL_DIRECTION:
        return styleName+"_"+lut[fname] # all same
      elif self.cmode == mode.LABEL:
        return styleName
      elif self.cmode == mode.DIRECTION:          
        return lut[fname]
    
    def add(self, styleName:str, foundStyle:str, fname:str, fdfname:str, isN:int, dist:float):
      cvrtName = self.convertStyleName(styleName, fname, self.lutfname)
      # print(cvrtName)
      if cvrtName in self.dictData:
        self.dictData[cvrtName]["isN"].append(isN)
        self.dictData[cvrtName]["foundstyle"].append(self.convertStyleName(foundStyle, fdfname, self.lutfname))
        self.dictData[cvrtName]["dist"].append(dist)
      else:
        self.dictData[cvrtName] = {"isN":[isN], "foundstyle":[self.convertStyleName(foundStyle, fdfname, self.lutfname)], "dist":[dist]}

    def stats(self, clabels, isnormalize = True):
      lactual = []
      lpred   = []
      # Computing Dist/Quant
      dim = len(clabels)
      mDist = np.zeros((dim, dim))
      mQuant = np.zeros((dim, dim))

      for key in self.dictData.keys():
        lfdStyle=self.dictData[key]["foundstyle"]
        ldist = self.dictData[key]["dist"]
        lisN = self.dictData[key]["isN"]
        idxkey = clabels.index(key)
        # print("act: ", key)
        for idx, elem in enumerate(lfdStyle):
          if(lisN[idx]):
            lactual.append(key)
            lpred.append(elem)
          mDist[idxkey][clabels.index(elem)]+= ldist[idx]
          mQuant[idxkey][clabels.index(elem)]+=1
          
          if(key==elem):
            self.linlier.append(ldist[idx])
          else:
            self.loutlier.append(ldist[idx])
          # print("pred: ", elem)
      allstats = {}
      cmpType = ["micro", "macro", "weighted"]
      allstats["acc"] = accuracy_score(lactual, lpred)

      for cidx, ctype in enumerate(cmpType): 
        prc = precision_score(lactual, lpred, average=ctype, zero_division=0)
        rec = recall_score(lactual,    lpred, average=ctype, zero_division=0)
        f1  = f1_score(lactual,        lpred, average=ctype, zero_division=0)
        allstats[ctype+"_prec"]=prc
        allstats[ctype+"_rec"] =rec
        allstats[ctype+"_f1"]  =f1
        
      # Computing ConfMat
      cm_real = confusion_matrix(lactual, lpred)
      distM = mDist/mQuant
      
      if isnormalize:
        cm_real = cm_real.astype('float') / cm_real.sum(axis=1)[:, np.newaxis]
        distM = minmax_scale(distM)

        return cm_real, distM, allstats
      
def LoadLUTDir(dir):
    lut = pd.read_csv(dir, delimiter=';', header=None)
    return dict(zip(lut[0], lut[1]))
          
def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for the video frame averaging tool.")

    parser.add_argument(
        "--MSfeatures",
        type=str,
        help="The features as pickle files of the motionset",
    )
    
    parser.add_argument(
        "--outputDir",
        default="/Users/triocrossing/INRIA/UnityProjects/analysis/resMotionSetSimAuto",
        type=str,
        help="output dir",
    )
    
    parser.add_argument(
        "--topN",
        default=3,
        type=int,
        help="top N similar when computing confusion matrix",
    )
    
    parser.add_argument(
        "--lutDir",
        default="/Users/triocrossing/Scripts/SimilarFinding/directions.csv",
        type=str,
        help="lut of fname to type e.g: up down etc. ",
    )
    
    return parser.parse_args()

def main(cmode):
    args = parse_args()
    stats = statsSim()
    stats.cmode = cmode
    stats.lutfname = LoadLUTDir(args.lutDir)
    print("loading pk ..")
    with open(args.MSfeatures, 'rb') as f:
        fMS = pickle.load(f)
    print("pk loaded ..")
    basenamepk = os.path.splitext(os.path.basename(args.MSfeatures))[0]
    print("pk name: ", basenamepk)
    
    for idxMS in tqdm(range(len(fMS))):
      msKey = list(fMS)[idxMS]
      msVal = list(fMS.values())[idxMS]
      styleName = msKey.rpartition('/')[0]
      typename = styleName[:4]

      lDist=[]
      for idxS, x in enumerate(fMS.values()):
        lDist.append(np.linalg.norm(x.flatten().numpy()- msVal.flatten().numpy()))      
      lSimDisim = [np.argsort(lDist)[:]]
      for lTopNArg in lSimDisim:
        for ctr, sidx in enumerate(lTopNArg[1:]):
          foundKeys = list(fMS.keys())[sidx]
          foundstyleName = foundKeys.rpartition('/')[0]
          fdtypename = foundstyleName[:4]
          if ctr<args.topN:
            stats.add(typename,fdtypename,styleName,foundstyleName,True,lDist[sidx])
          else:
            stats.add(typename,fdtypename,styleName,foundstyleName,False,lDist[sidx])
          
    print("current mode = ", cmode)
    if cmode == mode.LABEL_DIRECTION:
      xt = styleLabelsDir
    elif cmode == mode.DIRECTION:
      xt = dirLabels
    elif cmode == mode.LABEL: 
      xt = styleLabels
    
    basefoldreDir = args.outputDir+"/"+basenamepk
    if not os.path.exists(basefoldreDir):
      os.system('mkdir -p {dir}'.format(dir=basefoldreDir))
  
    ConfMat, DistMat, allstats=stats.stats(xt)
    
    fig = px.imshow(np.around(ConfMat,2), text_auto=True, x=xt, y=xt)
    # fig.show()
    fig = px.imshow(np.around(DistMat,2), text_auto=True, x=xt, y=xt)
    # fig.show()
    
    print("mean in: ", np.mean(stats.linlier))
    print("mean out: ", np.mean(stats.loutlier))
    
    group_labels = ['inlier', 'outlier']
    hist_data = [np.array(stats.linlier), np.array(stats.loutlier)]
    
    fig = ff.create_distplot(hist_data, group_labels,
                            show_curve=True)
    fig.update_layout(barmode='stack')
    # fig.show()
    
    outputdict = {}
    outputdict["CM"]      =ConfMat
    outputdict["Dist"]    =DistMat
    outputdict["inlier"]  =np.array(stats.linlier)
    outputdict["outlier"] =np.array(stats.loutlier)
    outputdict["labels"]  =xt
    outputdict["allstats"]=allstats
    outputdict["basename"]=basenamepk
    
    return outputdict, basefoldreDir

if __name__ == "__main__":
    allDict = {}
    allDict["label"]  , _             = main(mode.LABEL)
    allDict["puredir"], _             = main(mode.DIRECTION)
    allDict["withDir"], basefoldreDir = main(mode.LABEL_DIRECTION)

    with open(basefoldreDir+'/vizdata.pkl', 'wb') as f:
      pickle.dump(allDict, f)
