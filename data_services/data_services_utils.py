import pandas as pd

ARRIVAL_RAMP_DURATION_MINUTES = 10
ARRIVAL_AMA_DURATION_MINUTES = 10
DEPARTURE_AMA_DURATION_MINUTES = 10
DEPARTURE_RAMP_DURATION_MINUTES = 10

def add_flight_timeout(
    data: pd.DataFrame,
    logger = None,
) -> pd.DataFrame:
    '''
    Flight Time-Outs:
    If the final time (In for arrivals, Off for departures) is not received
    within a pre-configured time frame, the system will automatically time out
    the flight and assign a time to it. The time out values occur at the time
    specified below from the LAST received actual time for the flight.
    
    if isArrival:
        if arrival_stand_actual_time is null:
            if arrival_movement_area_time is null:
                arrival_stand_actual_time = arrival_runway_actual_time + 
                    ARRIVAL_AMA_DURATION_MINUTES + 
                    ARRIVAL_RAMP_DURATION_MINUTES
            else:
                arrival_stand_actual_time = arrival_movement_area_actual_time +
                    ARRIVAL_RAMP_DURATION_MINUTES
    elif isDeparture:
        if departure_runway_actual_time is null:
            if departure_movement_area_actual_time is null:
                departure_runway_actual_time = departure_stand_actual_time +
                    DEPARTURE_RAMP_DURATION_MINUTES +
                    DEPARTURE_AMA_DURATION_MINUTES
            else:
                departure_runway_actual_time = departure_movement_area_actual_time +
                    DEPARTURE_AMA_DURATION_MINUTES
            
    
    Analysis was conducted to find the average AMA and Ramp
    taxi times at KDFW, KCLT. These averages will be used as the timeout
    values (rounded up to the nearest 5 minutes for simplicity)
    
    ARRIVAL_RAMP_DURATION_MINUTES = 10
    ARRIVAL_AMA_DURATION_MINUTES = 10
    DEPARTURE_AMA_DURATION_MINUTES = 10
    DEPARTURE_RAMP_DURATION_MINUTES = 10
    '''
        
    # If arrival flight has no arrival_stand_actual_time and no
    # arrival_movement_area_actual_time, then set the arrival_stand_actual_time
    # to be the arrival_runway_actual_time + 
    # ARRIVAL_AMA_DURATION_MINUTES + ARRIVAL_RAMP_DURATION_MINUTES
    arr_timeout_full = ((data['isarrival'])&
                   (pd.isna(data['arrival_stand_actual_time']))& 
                   (pd.isna(data['arrival_movement_area_actual_time'])))
    if sum(arr_timeout_full) > 0:
        data.loc[arr_timeout_full,'arrival_stand_actual_time'] = (
            data.loc[arr_timeout_full,'arrival_runway_actual_time'] + 
            pd.Timedelta(minutes=ARRIVAL_AMA_DURATION_MINUTES) + 
            pd.Timedelta(minutes=ARRIVAL_RAMP_DURATION_MINUTES))
        
    # If arrival flight has no arrival_stand_actual_time but has a valid
    # arrival_movement_area_actual_time, then set the arrival_stand_actual_time
    # to be the arrival_movement_area_actual_time + ARRIVAL_RAMP_DURATION_MINUTES
    arr_timeout_ramp = ((data['isarrival'])&
                   (pd.isna(data['arrival_stand_actual_time']))& 
                   (pd.notna(data['arrival_movement_area_actual_time'])))
    if sum(arr_timeout_ramp) > 0:
        data.loc[arr_timeout_ramp,'arrival_stand_actual_time'] = (
            data.loc[arr_timeout_ramp,'arrival_movement_area_actual_time'] + 
            pd.Timedelta(minutes=ARRIVAL_RAMP_DURATION_MINUTES))
        

    # If departure flight has no departure_runway_actual_time but has a valid
    # departure_movement_area_actual_time, then set the 
    # departure_runway_actual_time to be the departure_movement_area_actual_time +
    # DEPARTURE_AMA_DURATION_MINUTES
    dep_timeout_ama = ((data['isdeparture'])&
                   (pd.isna(data['departure_runway_actual_time']))&
                   (pd.notna(data['departure_movement_area_actual_time'])))
    if sum(dep_timeout_ama) > 0:
        data.loc[dep_timeout_ama,'departure_runway_actual_time'] = (
            data.loc[dep_timeout_ama,'departure_movement_area_actual_time'] +             
            pd.Timedelta(minutes=DEPARTURE_AMA_DURATION_MINUTES))
        
    # If departure flight has no departure_runway_actual_time and no
    # departure_movement_area_actual_time, then set the 
    # departure_runway_actual_time to be the departure_stand_area_actual_time +
    # DEPARTURE_RAMP_DURATION_MINUTES + DEPARTURE_AMA_DURATION_MINUTES
    dep_timeout_full = ((data['isdeparture'])&
                   (pd.isna(data['departure_runway_actual_time']))&
                   (pd.isna(data['departure_movement_area_actual_time'])))
    if sum(dep_timeout_full) > 0:
        data.loc[dep_timeout_full,'departure_runway_actual_time'] = (
            data.loc[dep_timeout_full,'departure_stand_actual_time'] +
            pd.Timedelta(minutes=DEPARTURE_RAMP_DURATION_MINUTES) +
            pd.Timedelta(minutes=DEPARTURE_AMA_DURATION_MINUTES))
        
    return data

def fill_flight_times_for_surface_counts(
    data: pd.DataFrame,
) -> pd.DataFrame:
    
    # Set missing departure_movement_area_actual_time to 
    # departure_runway_actual_time
    data['departure_movement_area_actual_time'] = (
        data['departure_movement_area_actual_time'].combine_first(
        data['departure_runway_actual_time']))
        
    # Set missing departure_stand_actual_time to either 
    # departure_movement_area_actual_time or departure_runway_actual_time
    data['departure_stand_actual_time'] = (
        data['departure_stand_actual_time'].combine_first(
        data['departure_movement_area_actual_time']).combine_first(
        data['departure_runway_actual_time']))
    
    # Set missing arrival_runway_actual_time to either 
    # arrival_movement_area_actual_time or arrival_stand_actual_time
    data['arrival_runway_actual_time'] = (
        data['arrival_runway_actual_time'].combine_first(
        data['arrival_movement_area_actual_time']).combine_first(
        data['arrival_stand_actual_time']))
                
    # Set missing arrival_movement_area_actual_time to
    # arrival_stand_actual_time
    data['arrival_movement_area_actual_time'] = (
        data['arrival_movement_area_actual_time'].combine_first(
        data['arrival_stand_actual_time']))
            
    return data
