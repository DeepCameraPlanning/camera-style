import os
import os.path
import sys
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import glob
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff


def nameonFig(basename):
    return basename.replace("MotionSet_autoencoder-mse-reduced-7x7_2058-","MotionSet").replace("MotionSet","").replace("external_","").replace(".json","").replace("autoencoder-mse","").replace("-224-","").replace("-mse-","").replace("autoencoder-","")

def loadPkl(fname):
  with open(fname, 'rb') as f:
    loaded_dict = pickle.load(f)
    return loaded_dict
def main(label):
  args = sys.argv[1:]
  
  largs = len(args)
  print("received: ", largs)
  print(args)

  _alldata = glob.glob(args[0]+"/*/vizdata.pkl")
  

  alldataDict = [ loadPkl(i) for i in _alldata]
  titles = [ nameonFig(i[label]["basename"]) for i in alldataDict]
 
  figImg = make_subplots(
    rows=2, cols=len(alldataDict), subplot_titles=titles)
  figStats = make_subplots(
    rows=2, cols=len(alldataDict), subplot_titles=titles)
  figLine = make_subplots(
    rows=1, cols=1)
  for idx, dataDict in enumerate(alldataDict):

    # Img figure
    xt = dataDict[label]["labels"]
    # print("idx: ", idx)
    figImg.add_trace(px.imshow(np.around(dataDict[label]["CM"],2), text_auto=True, x=xt, y=xt).data[0], row=1, col=1+idx)
    basenameCM = os.path.basename(dataDict[label]["basename"])
    figImg.update_xaxes(row=1, col=1+idx, showgrid=False, scaleanchor = "y",
    scaleratio = 1)
    figImg.update_yaxes( scaleanchor = "x",
    scaleratio = 1, row=1, col=1+idx, autorange="reversed",showgrid=False)
    
    figImg.add_trace(px.imshow(np.around(dataDict[label]["Dist"],2), text_auto=True, x=xt, y=xt).data[0], row=2, col=1+idx)

    figImg.update_xaxes(row=2, col=1+idx, showgrid=False, scaleanchor = "y",
    scaleratio = 1)
    figImg.update_yaxes( scaleanchor = "x",
    scaleratio = 1, row=2, col=1+idx, autorange="reversed",showgrid=False)
    
    statdict = dataDict[label]["allstats"]

    # stats figure
    figStats.add_trace(go.Bar(x=list(statdict.keys()), y=list(statdict.values()), text=np.around(list(statdict.values()),2)), row=1, col=1+idx)
    figStats.update_yaxes(range=[0, 1], row=1, col=1+idx)
    
    group_labels = ['inlier', 'outlier']
    hist_data = [dataDict[label]["inlier"], dataDict[label]["outlier"]]
    
    figff = ff.create_distplot(hist_data, group_labels, bin_size=(np.max(dataDict[label]["outlier"])/20.0))

    figStats.add_trace(go.Histogram(figff['data'][0]
                              ), row=2, col=1+idx,)

    figStats.add_trace(go.Histogram(figff['data'][1]
                              ), row=2, col=1+idx,)

    figStats.add_trace(go.Scatter(figff['data'][2]
                            ), row=2, col=1+idx,)

    figStats.add_trace(go.Scatter(figff['data'][3]
                            ), row=2, col=1+idx,)
    figLine.add_trace(go.Scatter(x=list(statdict.keys()), y=list(statdict.values()), name=nameonFig(basenameCM)), row=1, col=1)
    
  figImg.show()
  figStats.show()
  figLine.show()
    
if __name__ == "__main__":
    main("label")
    main("puredir")
    main("withDir")
    


  
  
  


