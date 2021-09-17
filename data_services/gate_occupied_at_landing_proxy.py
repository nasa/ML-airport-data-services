# -*- coding: utf-8 -*-
"""
Compute gate occupied at landing

"""
import pandas as pd
from data_services import data_services_utils as utils

last_arrival_to_gate = {}
last_departure_from_gate = {}
departure_stand_times_per_gate = {}

def compute_gate_occupied_at_landing_proxy(
    data0: pd.DataFrame,
) -> pd.DataFrame:
    
    # Make a copy of the dataframe
    data = data0.copy()
    
    # If we coalesced the __actual__stand__ times fields, we need to grab
    # the original times and use those here for the surface counts, as that
    # will be what is used in the live system
    if 'arrival_stand_actual_time_orig' in data.columns:        
        data['arrival_stand_actual_time'] = data['arrival_stand_actual_time_orig']
        
    if 'departure_stand_actual_time_orig' in data.columns:
        data['departure_stand_actual_time'] = data['departure_stand_actual_time_orig']
            
    data = utils.add_flight_timeout(data)
    data = utils.fill_flight_times_for_surface_counts(data)
    
    #arrival_stand
    arrival_stand_runway_times = data[data['isarrival']==True]
    arrival_stand_runway_times = arrival_stand_runway_times[['gufi','arrival_runway_actual_time','arrival_movement_area_actual_time','arrival_stand_actual_time','arrival_stand_actual']]
    arrival_stand_runway_times=arrival_stand_runway_times.sort_values(by=['arrival_stand_actual_time'], ascending=[1])

    #departure_stand
    departure_stand_times = data[data['isdeparture']==True]
    departure_stand_times= departure_stand_times[["gufi","departure_stand_actual_time", "departure_stand_actual"]]
    departure_stand_times=departure_stand_times.sort_values(by=['departure_stand_actual_time'], ascending=[1])

    arrival_stand_runway_times= arrival_stand_runway_times.reset_index(drop=True)
    departure_stand_times = departure_stand_times.reset_index(drop=True)

    departure_gates=departure_stand_times['departure_stand_actual'].drop_duplicates().to_list()

    for departure_gate in departure_gates:
        departure_stand_times_per_gate[departure_gate]=departure_stand_times[
            departure_stand_times['departure_stand_actual'] == departure_gate].reset_index()

    arrival_stand_runway_times['gate_occupied_at_landing_proxy']=False

    # for each landing flight,
    # check last arrivals and last departures to gate.

    counter = 0
    for i, g in arrival_stand_runway_times.iterrows():
        flight_runway_time=g['arrival_runway_actual_time']
        flight_gate=g['arrival_stand_actual']
        flight_stand_time=g['arrival_stand_actual_time']

        recent_arrival_time= get_and_update_last_arrival_to_gate(flight_gate, flight_stand_time)
        recent_departure_time= get_and_update_last_departure_from_gate(flight_runway_time, flight_gate)
        # check  recent arrival and departure time, if arrival time is more recent than departure on the gate, this returns to true
        # in gate occupied at landing, otherwise set it to false
        if recent_arrival_time is not None and recent_departure_time is not None:
            arrival_stand_runway_times.at[i, 'gate_occupied_at_landing_proxy']=recent_arrival_time >= recent_departure_time

        counter+= 1
    
    # Merge back into dataframe
    arrival_stand_runway_times = arrival_stand_runway_times[['gufi','gate_occupied_at_landing_proxy']]
    
    # merging back into the original dataframe (data0)
    data_out = pd.merge(data0,
                        arrival_stand_runway_times,
                        on='gufi',
                        how='left')
    
    return data_out

def get_next_departure_from_gate(flight_gate, start_index):
    try:
        return departure_stand_times_per_gate[flight_gate].loc[start_index, "departure_stand_actual_time"]
    except Exception as e:
        pass

# create a cache for the latest arrival for each gate.
def get_and_update_last_arrival_to_gate(flight_gate, next_timestamp):
    previous_timestamp=last_arrival_to_gate.get(flight_gate)
    last_arrival_to_gate[flight_gate]=next_timestamp
    return previous_timestamp


# for each gate.
# get and cache the last two timestamps with the index  and check their values againest the flight landing time,

def get_and_update_last_departure_from_gate(flight_runway_time, flight_gate):
    if flight_gate in last_departure_from_gate:
        next_timestamp=last_departure_from_gate[flight_gate]['next_timestamp']
        if next_timestamp is None:
            return last_departure_from_gate[flight_gate]['previous_timestamp']
        c=0
        while next_timestamp is not None and next_timestamp < flight_runway_time:
            c+=1
            last_departure_from_gate[flight_gate]['previous_timestamp']=last_departure_from_gate[flight_gate][
                'next_timestamp']
            current_index=last_departure_from_gate[flight_gate]['next_index']
            next_timestamp=get_next_departure_from_gate(flight_gate, current_index + 1)
            last_departure_from_gate[flight_gate]['next_timestamp']=next_timestamp
            last_departure_from_gate[flight_gate]['next_index']=current_index + 1
        return last_departure_from_gate[flight_gate]['previous_timestamp']
    else:
        previous_timestamp=get_next_departure_from_gate(flight_gate, 0)

        if previous_timestamp is None:
            return None
        next_timestamp=get_next_departure_from_gate(flight_gate, 1)
        last_departure_from_gate[flight_gate]={}
        last_departure_from_gate[flight_gate]['previous_timestamp']=previous_timestamp
        last_departure_from_gate[flight_gate]['next_timestamp']=next_timestamp
        last_departure_from_gate[flight_gate]['next_index']=1
        return get_and_update_last_departure_from_gate(flight_runway_time, flight_gate)

    return None