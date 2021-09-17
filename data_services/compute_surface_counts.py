#!/usr/bin/env python

"""
Functions for computing surface counts

The surface count computations in this file are designed to match the surface
count computations done by the live NAS-MODEL system.

Input into the surface model computations are as follows for the OOOI times:
    DEPARTURES:
        Actual Out Time: departure_stand_actual_time
        Actual Spot Time: departure_movement_area_actual_time
        Actual Off Time: departure_runway_actual_time
    ARRIVALS:
        Actual On Time: arrival_runway_actual_time
        Actual Spot Time: arrival_movement_area_actual_time
        Actual In Time: arrival_stand_actual_time

"""
import pandas as pd
import numpy as np
from data_services import data_services_utils as utils

def compute_arrival_departure_count(
    data: pd.DataFrame
) -> pd.DataFrame:
    '''
    Rollup function to compute entire set of surface counts

    Parameters
    ----------
    data : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    DataFrame with surface counts

    '''
    
    data0 = compute_arrival_count_in_ramp_at_landing(data)
    data1 = compute_arrival_count_in_ama_at_landing(data0)
    data2 = compute_arrival_count_on_surface_at_landing(data1)
    
    data3 = compute_departure_count_in_ramp_at_landing(data2)
    data4 = compute_departure_count_in_ama_at_landing(data3)
    data5 = compute_departure_count_on_surface_at_landing(data4)
    
    data6 = compute_total_count_on_surface_at_landing(data5)
    
    return data6

def compute_total_count_on_surface_at_landing(
    data: pd.DataFrame
) -> pd.DataFrame:
    
    # Only interested in counts @ landing, so grab just the arrivals
    dat_arrivals = data[['gufi','arrival_runway_actual_time']]\
        [data['isarrival'] & data['arrival_runway_actual_time'].notnull()].copy().\
        sort_values(by='arrival_runway_actual_time').\
        reset_index(drop=True)
            
    departure_count = compute_departure_count_timeseries(data,
                                               'total',
                                               use_original_times=True,
                                               timeout_flights=True,
                                               infer_flight_times=True)
    
    # Merge just the arrival flights on to the counts, using merge_asof
    dat0 = pd.merge_asof(
        dat_arrivals,
        departure_count,
        left_on='arrival_runway_actual_time',
        right_on='timestamp')
    
    arrival_count = compute_arrival_count_timeseries(data,
                                               'total',
                                               use_original_times=True,
                                               timeout_flights=True,
                                               infer_flight_times=True)
    
    # Merge just the arrival flights on to the counts, using merge_asof
    dat1 = pd.merge_asof(
        dat0,
        arrival_count,
        left_on='arrival_runway_actual_time',
        right_on='timestamp')
    
    # Add arrival and departure counts
    dat1['total_flights_on_surface'] = (dat1['total_arrivals_on_surface'] +
                                        dat1['total_departures_on_surface'])
        
    # Merge back to original dataframe
    dat2 = data.merge(
        dat1[["gufi","total_flights_on_surface"]],
        on='gufi',
        how='left')
    
    return dat2

def compute_arrival_count_in_ama_at_landing(
    data: pd.DataFrame
) -> pd.DataFrame:
    
    # Calculate Arrival Counts in AMA @ Landing
    dat_arrivals = data[['gufi','arrival_runway_actual_time']]\
        [data['isarrival'] & data['arrival_runway_actual_time'].notnull()].copy().\
        sort_values(by='arrival_runway_actual_time').\
        reset_index(drop=True)
            
    arr_ama_count = compute_arrival_count_timeseries(data,
                                                     'ama',
                                                     use_original_times=True,
                                                     timeout_flights=True,
                                                     infer_flight_times=True)
    
    # Merge just the arrival flights on to the counts, using merge_asof
    dat0 = pd.merge_asof(
        dat_arrivals,
        arr_ama_count,
        left_on='arrival_runway_actual_time',
        right_on='timestamp')
        
    # Merge back to original dataframe
    dat1 = data.merge(
        dat0[["gufi","arr_runway_AMA_count"]],
        on='gufi',
        how='left')
    
    return dat1

def compute_arrival_count_in_ramp_at_landing(
    data: pd.DataFrame
) -> pd.DataFrame:
    
    # Calculate Arrival Counts in AMA @ Landing
    dat_arrivals = data[['gufi','arrival_runway_actual_time']]\
        [data['isarrival'] & data['arrival_runway_actual_time'].notnull()].copy().\
        sort_values(by='arrival_runway_actual_time').\
        reset_index(drop=True)
            
    arr_ramp_count = compute_arrival_count_timeseries(data,
                                                      'ramp',
                                                      use_original_times=True,
                                                      timeout_flights=True,
                                                      infer_flight_times=True)
    
    # Merge just the arrival flights on to the counts, using merge_asof
    dat0 = pd.merge_asof(
        dat_arrivals,
        arr_ramp_count,
        left_on='arrival_runway_actual_time',
        right_on='timestamp')
        
    # Merge back to original dataframe
    dat1 = data.merge(
        dat0[["gufi","AMA_gate_count"]],
        on='gufi',
        how='left')
    
    return dat1

def compute_arrival_count_on_surface_at_landing(
    data: pd.DataFrame
) -> pd.DataFrame:
    
    # Calculate Arrival Counts in AMA @ Landing
    dat_arrivals = data[['gufi','arrival_runway_actual_time']]\
        [data['isarrival'] & data['arrival_runway_actual_time'].notnull()].copy().\
        sort_values(by='arrival_runway_actual_time').\
        reset_index(drop=True)
            
    arr_count = compute_arrival_count_timeseries(data,
                                                 'total',
                                                 use_original_times=True,
                                                 timeout_flights=True,
                                                 infer_flight_times=True)
    
    # Merge just the arrival flights on to the counts, using merge_asof
    dat0 = pd.merge_asof(
        dat_arrivals,
        arr_count,
        left_on='arrival_runway_actual_time',
        right_on='timestamp')
        
    # Merge back to original dataframe
    dat1 = data.merge(
        dat0[["gufi","total_arrivals_on_surface"]],
        on='gufi',
        how='left')
    
    return dat1

def compute_departure_count_in_ama_at_landing(
    data: pd.DataFrame
) -> pd.DataFrame:
    
    # Only interested in counts @ landing, so grab just the arrivals
    dat_arrivals = data[['gufi','arrival_runway_actual_time']]\
        [data['isarrival'] & data['arrival_runway_actual_time'].notnull()].copy().\
        sort_values(by='arrival_runway_actual_time').\
        reset_index(drop=True)
            
    count = compute_departure_count_timeseries(data,
                                               'ama',
                                               use_original_times=True,
                                               timeout_flights=True,
                                               infer_flight_times=True)
    
    # Merge just the arrival flights on to the counts, using merge_asof
    dat0 = pd.merge_asof(
        dat_arrivals,
        count,
        left_on='arrival_runway_actual_time',
        right_on='timestamp')
        
    # Merge back to original dataframe
    dat1 = data.merge(
        dat0[["gufi","dep_AMA_runway_count"]],
        on='gufi',
        how='left')
    
    return dat1

def compute_departure_count_in_ramp_at_landing(
    data: pd.DataFrame
) -> pd.DataFrame:
    
    # Only interested in counts @ landing, so grab just the arrivals
    dat_arrivals = data[['gufi','arrival_runway_actual_time']]\
        [data['isarrival'] & data['arrival_runway_actual_time'].notnull()].copy().\
        sort_values(by='arrival_runway_actual_time').\
        reset_index(drop=True)
            
    count = compute_departure_count_timeseries(data,
                                               'ramp',
                                               use_original_times=True,
                                               timeout_flights=True,
                                               infer_flight_times=True)
    
    # Merge just the arrival flights on to the counts, using merge_asof
    dat0 = pd.merge_asof(
        dat_arrivals,
        count,
        left_on='arrival_runway_actual_time',
        right_on='timestamp')
        
    # Merge back to original dataframe
    dat1 = data.merge(
        dat0[["gufi","dep_stand_AMA_count"]],
        on='gufi',
        how='left')
    
    return dat1

def compute_departure_count_on_surface_at_landing(
    data: pd.DataFrame
) -> pd.DataFrame:
    
    # Only interested in counts @ landing, so grab just the arrivals
    dat_arrivals = data[['gufi','arrival_runway_actual_time']]\
        [data['isarrival'] & data['arrival_runway_actual_time'].notnull()].copy().\
        sort_values(by='arrival_runway_actual_time').\
        reset_index(drop=True)
            
    count = compute_departure_count_timeseries(data,
                                               'total',
                                               use_original_times=True,
                                               timeout_flights=True,
                                               infer_flight_times=True)
    
    # Merge just the arrival flights on to the counts, using merge_asof
    dat0 = pd.merge_asof(
        dat_arrivals,
        count,
        left_on='arrival_runway_actual_time',
        right_on='timestamp')
        
    # Merge back to original dataframe
    dat1 = data.merge(
        dat0[["gufi","total_departures_on_surface"]],
        on='gufi',
        how='left')
    
    return dat1

def compute_arrival_count_timeseries(
    data0: pd.DataFrame,
    surface_location: str,
    use_original_times: np.bool = True,
    timeout_flights: np.bool = True,
    infer_flight_times: np.bool = True
) -> pd.DataFrame:
    
    # Prepare the data for count computations - mimics what the live system
    # would be doing (timeout flights, fill flight times, etc.)
    data = prep_data_for_count_computations(data0,
                                            use_original_times,
                                            timeout_flights,
                                            infer_flight_times)

    # Grab only the arrival flights
    dat_arrivals = data[data['isarrival']].copy().reset_index(drop=True)
    
    # Compute timeseries count at specified location
    if surface_location=='ama':
        # Compute 
        count_ts = compute_timeseries_count(
            dat_arrivals['arrival_runway_actual_time'],
            dat_arrivals['arrival_movement_area_actual_time'])
        
        count_ts = count_ts.rename(columns={'count':'arr_runway_AMA_count'})
    elif surface_location=='ramp':
        # Compute 
        count_ts = compute_timeseries_count(
            dat_arrivals['arrival_movement_area_actual_time'],
            dat_arrivals['arrival_stand_actual_time'])
        
        count_ts = count_ts.rename(columns={'count':'AMA_gate_count'})
    elif surface_location=='total':
        # Compute 
        count_ts = compute_timeseries_count(
            dat_arrivals['arrival_runway_actual_time'],
            dat_arrivals['arrival_stand_actual_time'])
        
        count_ts = count_ts.rename(columns={'count':'total_arrivals_on_surface'})
    else:
        raise ValueError('Unknown surface location --> must specify [ama, ramp, total]')
        
    return count_ts

def compute_departure_count_timeseries(
    data0: pd.DataFrame,
    surface_location: str,
    use_original_times: np.bool = True,
    timeout_flights: np.bool = True,
    infer_flight_times: np.bool = True
) -> pd.DataFrame:
    
    # Prepare the data for count computations - mimics what the live system
    # would be doing (timeout flights, fill flight times, etc.)
    data = prep_data_for_count_computations(data0,
                                            use_original_times,
                                            timeout_flights,
                                            infer_flight_times)

    # Grab only the departure flights
    dat_departures = data[data['isdeparture']].copy().reset_index(drop=True)
    
    # Compute timeseries count at specified location
    if surface_location=='ama':
        # Compute 
        count_ts = compute_timeseries_count(
            dat_departures['departure_movement_area_actual_time'],
            dat_departures['departure_runway_actual_time'])
        
        count_ts = count_ts.rename(columns={'count':'dep_AMA_runway_count'})
    elif surface_location=='ramp':
        # Compute 
        count_ts = compute_timeseries_count(
            dat_departures['departure_stand_actual_time'],
            dat_departures['departure_movement_area_actual_time'])
        
        count_ts = count_ts.rename(columns={'count':'dep_stand_AMA_count'})
    elif surface_location=='total':
        # Compute 
        count_ts = compute_timeseries_count(
            dat_departures['departure_stand_actual_time'],
            dat_departures['departure_runway_actual_time'])
        
        count_ts = count_ts.rename(columns={'count':'total_departures_on_surface'})
    else:
        raise ValueError('Unknown surface location --> must specify [ama, ramp, total]')
        
    return count_ts

def prep_data_for_count_computations(
    data: pd.DataFrame,
    use_original_times: np.bool,
    timeout_flights: np.bool,
    infer_flight_times: np.bool,
) -> pd.DataFrame:
    '''
    These changes are ONLY for computing the surface counts, and we don't
    want them to make their way back into the original dataset.
    
    They mimic the live system and how it would handle the data as input to
    computing the surface counts
    '''
    
    # Make a copy of the dataframe
    data0 = data.copy()
    
    # If we coalesced the __actual__stand__ times fields, we need to grab
    # the original times and use those here for the surface counts, as that
    # will be what is used in the live system
    if use_original_times:
        if 'arrival_stand_actual_time_orig' in data0.columns:
            data0['arrival_stand_actual_time'] = data0['arrival_stand_actual_time_orig']
            
        if 'departure_stand_actual_time_orig' in data0.columns:
            data0['departure_stand_actual_time'] = data0['departure_stand_actual_time_orig']
    
    if timeout_flights:
        data0 = utils.add_flight_timeout(data0)
        
    if infer_flight_times:
        data0 = utils.fill_flight_times_for_surface_counts(data0)
            
    return data0

def compute_timeseries_count(
    entry_time: pd.Series,
    exit_time: pd.Series,
    ) -> pd.DataFrame():
    
    # Generate DataFrame for entry_time
    x = pd.DataFrame()
    x['timestamp'] = entry_time
    x['increment'] = 1

    # Generate DataFrame for exit time
    y = pd.DataFrame()
    y['timestamp'] = exit_time
    y['increment'] = -1
    
    # Concatenate DataFrames
    dat = pd.concat([x,y]).reset_index(drop=True)
    
    # Drop instances where event time is null
    dat = dat.drop(dat.index[pd.isna(dat['timestamp'])]).reset_index(drop=True)
    
    # Sort DataFrame on event time
    dat = dat.sort_values(by='timestamp').reset_index(drop=True)
    
    # Calculate count as cumsum()
    dat['count'] = dat['increment'].cumsum()
    
    # Drop Duplicates - we may have some instances where the event times are
    # identical, and we only care about the last one
    dat = dat.drop_duplicates(subset=['timestamp'],keep='last').reset_index(drop=True)
    
    return dat[['timestamp','count']]

def compute_arrival_departure_count_deprecated(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute a time series representation of expected counts, given frequently-
        updated predictions of some future event.
    """    
    
    # Make a copy of the dataframe
    data0 = data.copy()
    
    # If we coalesced the __actual__stand__ times fields, we need to grab
    # the original times and use those here for the surface counts, as that
    # will be what is used in the live system
    if 'arrival_stand_actual_time_orig' in data0.columns:
        data0['arrival_stand_actual_time'] = data0['arrival_stand_actual_time_orig']
        
    if 'departure_stand_actual_time_orig' in data0.columns:
        data0['departure_stand_actual_time'] = data0['departure_stand_actual_time_orig']
    
    data0 = utils.add_flight_timeout(data0)
    data0 = utils.fill_flight_times_for_surface_counts(data0)

    arrival_merged_times_runways_df = data0[data0['isarrival']==True]
    arrival_merged_times_runways_df = arrival_merged_times_runways_df[['gufi','arrival_runway_actual_time','arrival_movement_area_actual_time','arrival_stand_actual_time']]

    arrival_merged_times_runways_df =arrival_merged_times_runways_df.sort_values(by=['arrival_stand_actual_time'], ascending=[1])

    # arrival_runway_count
    arrival_runway_count = arrival_merged_times_runways_df.set_index('arrival_runway_actual_time').groupby( pd.Grouper(freq='1min')).count()
    arrival_runway_count = arrival_runway_count.rename(columns={"gufi": "arr_runway_count"})

    # arrival_spot_count
    arrival_spot_count = arrival_merged_times_runways_df.set_index('arrival_movement_area_actual_time').groupby(pd.Grouper(freq='1min')).count()
    arrival_spot_count = arrival_spot_count.rename(columns={"gufi": "arr_spot_count"})
    arrival_spot_count = arrival_spot_count.drop(columns=["arrival_stand_actual_time", "arrival_runway_actual_time"])

    # arrival_stand_count
    arrival_stand_count = arrival_merged_times_runways_df.set_index('arrival_stand_actual_time').groupby(pd.Grouper(freq='1min')).count()
    arrival_stand_count = arrival_stand_count.rename(columns={"gufi": "arr_stand_count"})
    arrival_stand_count = arrival_stand_count.drop(columns=["arrival_movement_area_actual_time", "arrival_runway_actual_time"])

    # merge arrival counts
    arrival_total1 = arrival_runway_count.join(arrival_spot_count, lsuffix='_left')
    arrival_total_counts = arrival_total1.join(arrival_stand_count)

    #remove na values
    arrival_total_counts = arrival_total_counts.fillna(0)

    #calculate AMA and gate count
    arrival_total_counts['runway_AMA_count'] = arrival_total_counts['arr_runway_count']-arrival_total_counts["arr_spot_count"]
    arrival_total_counts["AMA_gate_count"] = arrival_total_counts["arr_spot_count"] - arrival_total_counts["arr_stand_count"]
    arrival_total_counts["arr_runway_AMA_count"] = arrival_total_counts["runway_AMA_count"].cumsum()
    arrival_total_counts["arr_AMA_gate_count"] = arrival_total_counts["AMA_gate_count"].cumsum()
    arrival_total_counts = arrival_total_counts.rename_axis('date').reset_index()
    arrival_total_counts['total_arrivals_on_surface'] = arrival_total_counts["arr_runway_AMA_count"] + arrival_total_counts["arr_AMA_gate_count"]
    arrival_total_counts = arrival_total_counts[['date','arr_runway_AMA_count','AMA_gate_count','total_arrivals_on_surface']]

    ############Departure_count###########################################################4
    departure_merged_times_runways_df = data0[data0['isdeparture']==True]
    departure_merged_times_runways_df = departure_merged_times_runways_df[['gufi', 'departure_runway_actual_time', 'departure_movement_area_actual_time', 'departure_stand_actual_time']]

    departure_merged_times_runways_df = departure_merged_times_runways_df.sort_values(by=['departure_stand_actual_time'],ascending=[1])

    departure_stand_count = departure_merged_times_runways_df.set_index('departure_stand_actual_time').groupby(pd.Grouper(freq='1min')).count()
    departure_stand_count = departure_stand_count.rename(columns={"gufi": "dep_stand_count"})
    departure_stand_count = departure_stand_count.drop(columns=["departure_runway_actual_time", "departure_movement_area_actual_time"])

    # departure_spot_calculation
    departure_merged_times_runways_df = departure_merged_times_runways_df.sort_values(by=['departure_movement_area_actual_time'],ascending=[1])
    departure_spot_count = departure_merged_times_runways_df.set_index('departure_movement_area_actual_time').groupby(pd.Grouper(freq='1min')).count()
    departure_spot_count = departure_spot_count.rename(columns={"gufi": "dep_spot_count"})
    departure_spot_count = departure_spot_count.drop(columns=["departure_stand_actual_time", "departure_runway_actual_time"])

    # departure_runway_calculation
    departure_merged_times_runways_df = departure_merged_times_runways_df.sort_values(by=['departure_runway_actual_time'],ascending=[1])
    departure_runway_count = departure_merged_times_runways_df.set_index('departure_runway_actual_time').groupby(pd.Grouper(freq='1min')).count()
    departure_runway_count = departure_runway_count.rename(columns={"gufi": "dep_runway_count"})
    departure_runway_count = departure_runway_count.drop(columns=["departure_stand_actual_time","departure_movement_area_actual_time"])

    # joining the departure counts
    departure_total_counts = departure_stand_count.join(departure_spot_count,lsuffix='_left')
    departure_total_counts = departure_total_counts.join(departure_runway_count)
    
    # remove na values
    departure_total_counts = departure_total_counts.fillna(0)

    # Total departure on surface is the difference between the departures on runway and the departures in gate
    # calculate the amount of departure between stand and dep_spot and then between spot and runway
    departure_total_counts["dep_stand_AMA_count"] = departure_total_counts["dep_stand_count"] - departure_total_counts["dep_spot_count"]
    departure_total_counts["dep_AMA_runway_count"] = departure_total_counts["dep_spot_count"] - departure_total_counts["dep_runway_count"]
    departure_total_counts["dep_stand_AMA_count"] = departure_total_counts["dep_stand_AMA_count"].cumsum()
    departure_total_counts["dep_AMA_runway_count"] = departure_total_counts["dep_AMA_runway_count"].cumsum()
    departure_total_counts = departure_total_counts.rename_axis('date').reset_index()
    departure_total_counts["total_departures_on_surface"] = departure_total_counts["dep_stand_AMA_count"] + departure_total_counts["dep_AMA_runway_count"]
    departure_total_counts = departure_total_counts[['date', 'dep_AMA_runway_count', 'dep_stand_AMA_count', 'total_departures_on_surface']]

    #Total_flights_on_surface
    total_flights_count = pd.merge(arrival_total_counts,departure_total_counts, on = ['date'])
    total_flights_count['total_flights_on_surface']= total_flights_count['total_arrivals_on_surface'] + total_flights_count['total_departures_on_surface']

    total_flights_count['date'] = total_flights_count["date"].astype('datetime64[ns]')
        
    # Merge back to data
    data['date'] = data['arrival_runway_actual_time'].dt.floor('T')            
    df = pd.merge(data,
                  total_flights_count,
                  on='date',
                  how='left')    
    df = df.drop(columns=['date'])  
            
    return df
