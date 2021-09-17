"""
    Encoder class that extend scikit-learn encoder
    Requires a pre-defined mapping from stands to stand clusters.

"""

from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

import pandas as pd
import re
import logging
import mlflow
import os
import numpy as np

from data_services.analytics_visualization_artifacts import make_cluster_plot



class GateClusterEncoder(BaseEstimator, TransformerMixin):
    """
        Encoder for Gates, the algorithm first clusters the gates by taxi time defined
        in the fit y input btw the gates and runways 

        model_params needed :
        gate_cluster :
          to_transform :
          scaler :
          clust :
          nrunways :
          nclusters :

         DataFrame needs columns : 'departure_runways_actual', to_transform ('arrival_stand_actual' 
         or 'departure_stand_actual')



    """
    def __init__(self,
                 to_transform = 'departure_stand_actual',
                 scaler = '',
                 clust = 'hierch',
                 nrunways = 3,
                 nclusters = 5,
                 active_run_id : Any = None):

        self.to_transform = to_transform
        self.scaler = scaler
        self.clust = clust
        self.nrunways = nrunways
        self.nclusters = nclusters
        self.active_run_id = active_run_id

        
    def fit(
            self,
            data,
            y
    ):

        print("fitting pars :\n")
        print(self)
        print("end fit pars ####\n")
        if (type(self.to_transform) != str) :
            raise(TypeError("GateClusterEncoder expects a string in the to_transform\n"+\
                            "field of gate_cluster parameter but got : {}".format(self.to_transform)))
        
        if (self.to_transform not in ['arrival_stand_actual','departure_stand_actual']) :
            raise(TypeError("GateClusterEncoder expects 1 column dataframe with specific column name\n"+\
                            "Given DataFrame with columns : {}".format(self.to_transform)))
                
        colnames = []
        nrunways = int(self.nrunways)
        nclusters = int(self.nclusters)
        keytrans = self.to_transform

        scaler_dict = {'minmax':MinMaxScaler(),
                       'standard':StandardScaler(),
                       '':FunctionTransformer(validate=False)}

        clust_dict = {'hierch': AgglomerativeClustering,
                      'kmean': KMeans}

        log = logging.getLogger(__name__)

        # Add back the target that should be in the y, hopefully order is consistent
        data_copy = data.copy()
        data_copy['gate_cluster_target'] = y

        # Filter on data is going to be the same as for the full-pipeline
        # Given it will be one of the process in the FilterPipeline
        
        sumary=data_copy.groupby(['departure_runway_actual',keytrans])\
            ['gate_cluster_target'].describe().sort_values('count')
        high_ct_flights = sumary[sumary['count'] >= 10]
        high_ct_med = high_ct_flights['50%'].reset_index()
        runways_ct = high_ct_med['departure_runway_actual'].value_counts()

        gateinfo=pd.pivot(high_ct_med,index=keytrans,columns='departure_runway_actual',values='50%')
        selrwy = runways_ct[0:nrunways].index
        gateinfo['terminal'] = gateinfo.index.str.extract('^(?P<terminal>[A-Z])[0-9].*').values
        # Give a name to no/undefined terminal
        gateinfo['terminal'] = gateinfo['terminal'].fillna('-99')

        if (self.scaler not in scaler_dict.keys()):
            raise(TypeError("GateClusterEncoder expects a specific name for the scaler"+
                            "But Given the following name : {}".format(self.scaler)))
        
        if (self.clust not in clust_dict.keys()):
            raise(TypeError("GateClusterEncoder expects a specific name for the clustering algorithm"+
                            "But Given the following name : {}".format(self.clust)))

        sel_full = gateinfo[selrwy].notnull().all(axis=1)
        gateinfo_sel = gateinfo.loc[sel_full,selrwy].copy()

        flight_cover= data_copy[keytrans].isin(gateinfo_sel.index).sum()/len(data_copy[keytrans])*100.0
        gate_cover = len(gateinfo_sel)/len(data_copy[keytrans].unique())*100.0

        log.info('Gate Clustering : {:.1f}% of flights and {:.1f}% of gates covered'\
                 .format(flight_cover,gate_cover))

        Xgates = scaler_dict[self.scaler].fit_transform(gateinfo_sel)
        
        clusal = clust_dict[self.clust](nclusters)
        labels = clusal.fit_predict(Xgates)

        gateinfo_sel['labels'] = labels
        gateinfo_sel['clustered'] = nrunways
        gateinfo_sel['terminal'] = gateinfo.loc[sel_full, 'terminal']
        
        ############################################################################
        ##### Fill Gates that don't have the required runways in the fitting data
        needlabel = set(gateinfo.index) - set(gateinfo_sel.index)
        if (len(needlabel) > 0) :
            gate_unknown = gateinfo.loc[needlabel, :]
            gate_unknown['clustered'] = gate_unknown[selrwy].notnull().sum(axis=1)

            def closest_gate_in_terminal(row):
                sel = row.notnull() & row.index.isin(selrwy)
                dist = ((gateinfo_sel.loc[:,sel[sel].index]-row[sel])**2).sum(axis=1)
                dist = dist[gateinfo_sel['terminal'] == row['terminal']]
                mindistgates = dist.index[dist == dist.min()]
                return mindistgates

            def closest_gate(row):
                sel = row.notnull() & row.index.isin(selrwy)
                dist = ((gateinfo_sel.loc[:,sel[sel].index]-row[sel])**2).sum(axis=1)
                mindistgates = dist.index[dist == dist.min()]
                return mindistgates

            # find closest gate in terminal
            gate_unknown['closest_gate'] = gate_unknown.apply(closest_gate_in_terminal, axis=1)
            no_closest_gate_found  = (gate_unknown['closest_gate'].apply(len) == 0)
            if (no_closest_gate_found.sum() > 0) :
                # if not found, find closest gate
                gate_unknown.loc[no_closest_gate_found, 'closest_gate'] = \
                    gate_unknown[no_closest_gate_found].apply(closest_gate, axis=1)
            no_closest_gate_found  = (gate_unknown['closest_gate'].apply(len) == 0)
            if (no_closest_gate_found.sum() > 0) :
                # if not found, drop the gate, it will be an unseen gate in the transformer
                gate_unknown = gate_unknown.drop(no_closest_gate_found)
            
            # return most frequent label :
            gate_unknown['labels'] = gate_unknown['closest_gate'].apply(lambda val: gateinfo_sel['labels'][val].mode()[0])
            gate_unknown=gate_unknown.drop('closest_gate',axis=1)

            gateinfo_sel = pd.concat([gateinfo_sel,gate_unknown])
        ############################################################################ 

        ##### One hot encode the cluster number
        self.df_gates_encoded = pd.get_dummies(gateinfo_sel, prefix='cluster', sparse=False, columns=['labels'])
        self.colnames = self.df_gates_encoded.keys()
        
        ############## Set a default cluster (most common), for gates in the prediction data
        ############## but not in the training set
        most_freq_clus = gateinfo_sel['labels'].mode()[0]
        one_gate_most_freq = gateinfo_sel.query('labels == '+str(most_freq_clus)).index[0]
        self.default_cluster = self.df_gates_encoded.loc[one_gate_most_freq].copy()
        self.default_cluster['clustered'] = 0 # to show that gates using default were not in the fit

        
        ## log some results into MLFlow server if active_run_id is defined
        if (self.active_run_id != None) :
            with mlflow.start_run(run_id=self.active_run_id):
                # Commented out : because no more gufi here
                #deparr=data_copy.reset_index()['gufi'].str.extract(r'.*\.(?P<dep>[A-Z]+)\.(?P<arr>[A-Z]+)\..*')
                #ndeparr = arrdep['dep'].value_counts()[0],arrdep['arr'].value_counts()[0]
                #airp = deparr['dep'].mode()[0] if (ndeparr[0] >= ndeparr[1]) else deparr['arr'].mode()[0]
                fileprefix = keytrans+'_nclus{}_{}_{}'.format(nclusters,self.clust,self.scaler)
                vis, self.colors = make_cluster_plot(gateinfo_sel, selrwy, fileprefix)
                mlflow.log_artifact(vis)
                os.remove(vis)

        
        return self

    
    def transform(
            self,
            data,
    ) -> pd.DataFrame:

        log = logging.getLogger(__name__)
        
        key_to_merge = self.to_transform
        # Can only include the encoding, not the additional information
        keys_to_include = self.df_gates_encoded.keys()[self.df_gates_encoded.keys().str.startswith('cluster_')]
        data_out = pd.merge(data, self.df_gates_encoded[keys_to_include].reset_index(), sort=False, on=key_to_merge, how='left')
        # fill missing cluster number with default values ie most common :
        unknown_gate = data_out[keys_to_include].isna().any(axis=1)
        if (unknown_gate.sum() > 0) :
            # First look if some gates are similar up to an additional letter
            similar_gate_extracted = data_out[key_to_merge].str.extract(r'([A-Z][0-9])+[A-Z]+').iloc[:,0]
            similar_gates=unknown_gate & similar_gate_extracted.isin(self.df_gates_encoded.index)
            if (similar_gates.sum() > 0) :
                data_out.loc[similar_gates, keys_to_include] = self.df_gates_encoded.loc[similar_gate_extracted[similar_gates], keys_to_include].values
                unknown_gate = data_out[keys_to_include].isna().any(axis=1)
            # Second look for gates @ -4,-2,+2,+4
            found_clusters, changed = self._replace_to_neighbor_gate(
                data_out.loc[unknown_gate, key_to_merge], self.df_gates_encoded[keys_to_include])
            unknown_gate[changed[changed].index] = False
            if (changed.sum() > 0) :
                data_out.loc[changed[changed].index, keys_to_include] = found_clusters
            # Third  look for same terminal
            found_clusters, changed = self._replace_to_same_terminal(
                data_out.loc[unknown_gate, key_to_merge], self.df_gates_encoded[keys_to_include])
            unknown_gate[changed[changed].index] = False
            if (changed.sum() > 0) :
                data_out.loc[changed[changed].index, keys_to_include] = found_clusters
            # last choice, fill with default            
            data_out.loc[unknown_gate, keys_to_include] = self.default_cluster[keys_to_include].values
            log.info('Gate Clustering transform : {:.1f}% of flights with unknown gate in the fit'
                     .format(unknown_gate.sum()/len(data_out)*100.0))

        # Needs to drop the original column to avoid double fitting  & errors (because these are not numbers)
        data_out = data_out.drop(key_to_merge, axis=1)
        
        return data_out

    def get_feature_names(self):
            return self.colnames

        
    def _from_cluster_to_array(self, cluster_df, ncols) :
        nrows = len(cluster_df)
        array_out = np.zeros(nrows*ncols)
        array_out[cluster_df.values + np.arange(0,nrows*ncols,ncols)] = 1.0
        array_out = array_out.reshape(nrows, ncols)
        return array_out

    def _replace_to_neighbor_gate(self, gate_value, gate_code) :
        gate_code_cols = gate_code.columns
        num_neigh = [-4,-2,2,4] # look around current gate, preserve oddness
        neighbor_gate_extracted = gate_value.str.extract(r'(?P<terminal>[A-Z])(?P<gate>[0-9]+).*')
        neighbor_gate_extracted = neighbor_gate_extracted.fillna('-99')
        for col in gate_code_cols :
            neighbor_gate_extracted[col]=0.0
        for neighbor in num_neigh :
            new_gate = neighbor_gate_extracted.terminal+\
                (neighbor_gate_extracted.gate.astype(int)+neighbor).astype(str)
            neighbor_gate_extracted.loc[:, gate_code_cols] += gate_code.reindex(new_gate, fill_value=0.0).values
        changed = neighbor_gate_extracted[gate_code_cols].max(axis=1) != 0
        if (changed.sum() > 0) :
            new_cluster = neighbor_gate_extracted.loc[changed, gate_code_cols].idxmax(axis=1).str[-1:].astype(int)
            array_out = self._from_cluster_to_array(new_cluster, len(gate_code_cols))
        else :
            array_out = []
        return array_out, changed

    def _replace_to_same_terminal(self, gate_value, gate_code) :
        gate_code_cols = gate_code.columns
        gate_code_index = gate_code.index.name
        gate_code = gate_code.reset_index().join(\
            gate_code.reset_index()[gate_code_index].str.extract(r'(?P<terminal>[A-Z])(?P<gate>[0-9]+).*'))
        gate_code = gate_code.set_index(gate_code_index)
        # if no set of (terminal, gate) is defined : exit with no proposed change
        if ((gate_code['terminal'].isna() | gate_code['gate'].isna()).all()) :
            return [], pd.Series(data=False,index=gate_value.index)
        cluster_of_terminal = gate_code.groupby('terminal').mean().idxmax(axis=1).str[-1:].astype(int)
        terminal_search_str = ",".join(cluster_of_terminal.index)
        terminal_gate_extracted = gate_value.str.extract(r'(?P<terminal>['+terminal_search_str+'])(?P<gate>[0-9]+).*')
        non_null_terminal = terminal_gate_extracted['terminal'].notnull()
        new_cluster = cluster_of_terminal[terminal_gate_extracted['terminal'][non_null_terminal]]
        array_out = self._from_cluster_to_array(new_cluster, len(gate_code_cols))
        return array_out, non_null_terminal
    


    
