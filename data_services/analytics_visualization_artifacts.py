"""
     Functions to make some plots about the model results
"""
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, Dict
from . import FilterPipeline

import pandas as pd
import mlflow
import matplotlib.pyplot as pl
import matplotlib as mat
import os
import numpy as np
import seaborn as sn
import logging

def visualization_caller(
    data: pd.DataFrame,
    pipeline_for_filter : FilterPipeline,
    model_params: Dict[str, Any],
    global_params: Dict[str, Any],
    active_run_id: str,
) -> None:
    """
    Function to call analytics visualization functions as requested
    in the parameter files

    Args:

    data : dataframe containing the data that will be plotted
    (model needs to have been trained and predictions made)

    pipeline_for_filter : one filtered pipeline (main or baseline) 
    to allow data to be filtered before to be plotted
    
    model_params : parameters of the model, need to include
    a new section 'visual' to describe the desired plot,
    for now 'visual' contains 4 sub-sections :
      'plots' : describing the type of plots (need to
      be defined in the dict_of_funcs dictionary)
      'datasets' : for now list of data.group to run on
      ('test' or/and 'train')
      'estimates' : list of names of the prediction to run on
      (model_params['name'] or/and 'predicted_baseline')
      'kwargs' : dictionary of kwargs to be fed to the plots
           this sub-section is optional, the code will assume {}
           if it is not populated

    global_params : global parameters to get the name of the airport 
    
    active_run_id : MLFlow run ID to be able to log the plots
    as artifacts


    Returns :

    None


    """
    dict_of_funcs = {
        "residual_histogram": residual_histogram,
        "estimate_vs_truth": estimate_vs_truth,
    }
    
    airp = global_params['airport_icao']
    visual_requested = model_params['visual']['plots']
    dataset_requested = model_params['visual']['datasets']
    with mlflow.start_run(run_id=active_run_id) :
        for visual_i in visual_requested :
            if (visual_i in dict_of_funcs.keys()) :
                for dataseti in dataset_requested :
                    kwarg_vis = extract_kwargs(model_params['visual'],visual_i)
                    vis=dict_of_funcs[visual_i](
                        data,
                        pipeline_for_filter,
                        model_params,
                        dataseti,
                        airp,
                        kwarg_vis
                    )
                    for visi in vis :
                        mlflow.log_artifact(visi)
                        os.remove(visi) # Clean-up file since it's in the store

    return True


def extract_kwargs(model_visual, plot_name):
    """
    """
    out_dict = {}
    if ('kwargs' in model_visual.keys()) :
        if (plot_name in model_visual['kwargs'].keys()):
            out_dict = model_visual['kwargs'][plot_name]
    return out_dict
        

def residual_histogram(
        data: pd.DataFrame,
        pipeline_for_filter : FilterPipeline,
        model_params: Dict[str, Any],
        data_group: str,
        airp : str,
        kwargs_hist : Dict[str,Any] = {}
        )-> str:
    """
    Plot a histogram of the residuals of each estimate 

    If STBO undelayed taxi time is available and requested, output a second histogram
    with STBO taxi time as the truth data

    """
    estimates = model_params['visual']['estimates']
    target_val = [model_params['target']]
    prefxs = ['']
    # Use baseline model  although filter should be the same in taxi-out
    # Check if STBO is also available and add as a target if so
    if ('label' in model_params) :
        stbo_target = 'undelayed_departure_{}_transit_time'.format(model_params['label'])
        if stbo_target in data.columns:
            target_val.append(stbo_target)
            prefxs.append('STBO ')
    
    feats = model_params['features']
    group_sel = data.group == data_group
    if ("unimpeded_AMA" in data.keys()) :
        group_sel = group_sel & data.unimpeded_AMA
        
    sel_x, _, sel_y, _ = pipeline_for_filter.filter(data.loc[:,feats],
                                                    data.loc[:,target_val[0]])
    sel = sel_x & sel_y & group_sel
    
    y_trues = data.loc[sel,target_val]

    outname = airp+'_'+data_group+'_'+'residual_histogram.png'

    pl.ioff()

    outnames = []
    for y_true, prefx in zip([x[1] for x in y_trues.iteritems()],prefxs) :
        pl.clf()
        for estimatei in estimates :
            y_est  = data.loc[sel,estimatei]
            est_mae,est_rmse = np.nanmean(np.abs(y_true-y_est)),np.nanmean((y_true-y_est)**2)**0.5
            if (estimatei == estimates[0]) :
                resmin,resmax = np.nanpercentile(y_est-y_true,[0.1,99.9])
                range_value = np.abs(resmax-resmin)
                resmin,resmax = resmin-range_value/10.0,resmax+range_value/10.0
            hist_args = {'bins':41,'alpha':0.5,'label':estimatei+', MAE: {:2.1f} , RMSE: {:2.1f}'.format(est_mae,est_rmse),'range':[resmin,resmax]}
            hist_args.update(kwargs_hist)
            pl.hist(y_est-y_true,**hist_args)
        pl.title(prefx+airp+' - '+data_group+' set - residuals')
        pl.xlabel('y_est - y_true')
        pl.ylabel('Num. of flights')
        pl.legend(loc=2)
        pl.savefig(prefx.replace(' ','_')+outname,bbox_inches='tight')
        outnames.append(prefx.replace(' ','_')+outname)
        pl.close()

    return outnames





def estimate_vs_truth(
        data: pd.DataFrame,
        pipeline_for_filter : FilterPipeline,
        model_params: Dict[str, Any],
        data_group : str,
        airp : str,
        kwargs_plot : Dict[str,Any] = {}
        )->Any:
    """
    Plot an estimate vs truth for each estimate

    If STBO undelayed taxi time is available and requested, output a second plot
    with STBO taxi time as the truth data

    """
    estimates = model_params['visual']['estimates']
    target_val = [model_params['target']]
    prefxs = ['']

    # Check if STBO is also available and add as a target if so
    if ('label' in model_params) :
        stbo_target = 'undelayed_departure_{}_transit_time'.format(model_params['label'])
        if stbo_target in data.columns:
            target_val.append(stbo_target)
            prefxs.append('STBO ')


    feats = model_params['features']
    group_sel = data.group == data_group
    if ("unimpeded_AMA" in data.keys()):
        group_sel = group_sel & data.unimpeded_AMA
        
    sel_x, _, sel_y, _ = pipeline_for_filter.filter(data.loc[:,feats],
                                                    data.loc[:,target_val[0]])
    sel = sel_x & sel_y & group_sel

    y_trues = data.loc[sel,target_val]

    outname = airp+'_'+data_group+'_'+'estimate_vs_truth.png'
    
    shapes = ['o','s','d']
   
    pl.ioff()
    outnames = []
    for y_true, prefx in zip([x[1] for x in y_trues.iteritems()],prefxs) :
        pl.clf()
        yall = y_true.copy() # to find a good range
        for i,estimatei in enumerate(estimates) :
            y_est  = data.loc[sel,estimatei]
            yall = np.concatenate((yall,y_est))
            plot_args = {'alpha':0.5,'label':estimatei}
            plot_args.update(kwargs_plot)
            pl.plot(y_true,y_est,shapes[i],**plot_args)
        pl.title(prefx+airp+' - '+data_group+'  set')
        pl.legend()
        pl.xlabel('y_true [secs]')
        pl.ylabel('y_est [secs]')
        ymin,ymax = np.nanpercentile(yall,[0.1,99.9])
        range_value = np.abs(ymax-ymin)
        ymin,ymax = ymin-range_value/5.0,ymax+range_value/5.0
        pl.plot(np.arange(ymin,ymax),np.arange(ymin,ymax),'k--')
        pl.axis([ymin,ymax,ymin,ymax])
        pl.savefig(prefx.replace(' ','_')+outname,bbox_inches='tight')
        outnames.append(prefx.replace(' ','_')+outname)
        pl.close()
    
    return outnames

        
def make_cluster_plot(gateinfo_sel, runways, fileprefix='') :
    """
       Make a N-D plot of the gate to runways taxi time with marks for gate clusters
       Used in gate_cluster_encoder.py

    """

    log = logging.getLogger(__name__)
    nrunways = len(runways)
    gate_df = gateinfo_sel[gateinfo_sel.clustered == nrunways].copy() # Only the clustered, ie with all the necessary runways
    
    if (nrunways < 2) :
        log.info('Gate Clustering : plot not created, only {} runways'
                 .format(nrunways))
    if (nrunways > 4) :
        runways = runways[0:4]
        nrunways = 4
        log.info('Gate Clustering : too many runways ({}), selecting only those {} runways'
                 .format(nrunways,runways))

    # Try to order the runways in a "clever" way, ie try to have opposite flow in x,y
    # then the rest
    if (nrunways > 2) :
        # look at first and last character
        runways_m = pd.DataFrame({'head':runways.str[0].values,'pos':runways.str[-1].values})
        # set the list of index
        runindx = set(range(nrunways))
        # calculate correlation btw runways
        num_vec = pd.get_dummies(runways_m).values
        corr_m = num_vec.dot(num_vec.T)
        # choose runways with least in common for x,y pair
        # correction to have only 1 of the pair (it could several at the minimum value)
        xyindx = np.array(np.where(corr_m == corr_m.min()))[:,0]
        othidx = list(runindx - set(xyindx))
        # reorder everything :
        runways = runways[xyindx].append(runways[othidx])
        
    filename = fileprefix+'_'+"_".join(runways)+'_'+'gate_runway_taxi_clusters.png'
    # Python matplotlib color progression :
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',\
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # Add extra random colors if the number of clusters is larger :
    nextra_col = gate_df['labels'].max()+1-len(colors)
    if (nextra_col > 0) :
        extra_cols = ['#'+hex(x)[2:].zfill(6) for x in np.random.choice(256,size=[nextra_col,3]).dot([1,2**8,2**16])]
        colors = colors + extra_cols
    
    pl.ioff()
    pl.clf()
    # 4D version
    if (nrunways == 4) :
        sn.relplot(data=gate_df,
                   x=runways[0], y=runways[1],
                   size=runways[2], hue=runways[3],
                   palette='viridis'
                   )
    elif (nrunways == 3) :
        # 3D version
        sn.relplot(data=gate_df,
                   x=runways[0], y=runways[1],
                   size=runways[2]
                   )
    else :
        #2D version
        sn.relplot(data=gate_df,
                   x=runways[0], y=runways[1],
                   )

        
    ax = pl.gca()
    rad = gate_df[runways[0:2]].std().mean()/10.0
    for _,gatei in gate_df.iterrows():
        ax.add_artist(mat.patches.Circle((gatei[runways[0]],gatei[runways[1]]),
                                         color=colors[int(gatei["labels"])],radius=rad,fill=False))

    pl.draw()
    pl.savefig(filename,bbox_inches='tight')
    pl.close()

    return filename, colors
