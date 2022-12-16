import pyart
import numpy as np
import copy
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage import median_filter

def main(radar = None):
    """
    This function is necessry to evaluate many of the proposed products, such as VIL, ET, VILD, MESH and POSH.
    It sorts all azimuths in an incresing order and fields following the resorted order. It also sorts the elevation angles of each sweep if the are not constant as with the DECEA radars.
    If the azimuthal resolution is not constant between sweeps, it deletes the extra ray in case of the DECEA radars,
    or interpolates the fields at higher tilts to maintain the azimuth array shape homogeneous.
    
    Parameters
    __________
    
    radar: Radar 
        Radar object used
  
    Returns
    __________
    
    Radar object with sorted azimuths, elevations and fields.
    """
    
    # get the lowest elevation angle and range gates
    sweep_min = np.min(radar.sweep_number['data'])
    ranges = radar.range['data']
    
    # start iterating at each sweep
    for sweep in radar.sweep_number['data']:
        # get sweep slices
        sweep_slice = radar.get_slice(sweep)
        
        # get the radar azimuths of each slice
        azi = radar.azimuth['data'][sweep_slice]
        
        # if the array has one extra azimuth, delete the last one
        if azi.shape[0] == 361:
            azi = np.delete(azi, [-1], axis=0)
        
        # get the radar elevation angle of each slice
        ele = radar.elevation['data'][sweep_slice]
        
        # if the array has one extra elevation as in the azimuth array, delete the last one
        if ele.shape[0] == 361:
            ele = np.delete(ele, [-1], axis=0)
        
        # sort the azimuths and elevations in an increasing order
        az_idx = np.argsort(azi)
        az_sorted = azi[az_idx]
        ele_sorted = ele[az_idx]
        
        # save the lowest tilt azimuth array
        if sweep == sweep_min:
                azimuth_final = azi
        
        # if higher tilt azimuths have a lower azimuthal resolution, perform the operations needed to maintain an homogeneous shape between arrays
        if az_sorted.shape[0] != azimuth_final.shape[0]:
            # interpolate between azimuths using the shape of the first sweep
            az_sorted = np.linspace(az_sorted[0], az_sorted[-1], azimuth_final.shape[0])
            # expand the elevation array to have the same shape as the azimuths
            ele_sorted = np.resize(ele_sorted, azimuth_final.shape[0])
            # create an azimuth index array
            az_idx = np.arange(0, azimuth_final.shape[0], 1)
        
        # if the base sweep, copy the azimuths, elevations and indexes creating a new axis allowing the vertical stacking of the array
        if sweep == sweep_min:
            az_final = copy.deepcopy(az_sorted[np.newaxis, :])
            ele_final = copy.deepcopy(ele_sorted[np.newaxis, :])
            idx_final = copy.deepcopy(az_idx[np.newaxis, :])
        # stack the arrays at higher tilts
        else:
            az_final = np.concatenate([az_final, az_sorted[np.newaxis, :]])
            ele_final = np.concatenate([ele_final, ele_sorted[np.newaxis, :]])
            idx_final = np.concatenate([idx_final, az_idx[np.newaxis, :]])
            
    # iterate through each field and sort their arrays according to the azimuth array
    for fields in list(radar.fields.keys()):
        # iterate over each sweep
        for sweep in radar.sweep_number['data']:
            # get the gate's x and y coordenates, and evaluate their range
            x, y, _ = radar.get_gate_x_y_z(sweep)
            r = np.sqrt(x**2 + y**2)
            
            # get the field slice for every sweep
            field = radar.get_field(sweep, fields)
            field_type = field.dtype
            
            # similarly, if there's one extra azimuth for the sweep, delete the last ray of the range and field's arrays.
            if field.shape[0] == 361:
                r = np.delete(r, [-1], axis=0)
                field = np.delete(field, [-1], axis=0)
            
            # get the sweep slice
            sweep_slice = radar.get_slice(sweep)
            
            # get the unambiguous range of each sweep
            max_range = radar.instrument_parameters['unambiguous_range']['data'][sweep_slice][0]
            
            # get the sorted azimuth arrays for the sweep
            azi = radar.azimuth['data'][sweep_slice]
            if azi.shape[0] == 361:
                azi = np.delete(azi, [-1], axis=0)
            
            az_idx = np.argsort(azi)
            azi_sorted = azi[az_idx]
            field_sorted = field[az_idx]
            
            # interpolate the range and fields if the shape of azimuth axis is smaller than at the lowest sweep.
            if field.shape[0] != az_final[0].shape[0]:
                range_interpolator = interp1d(azi_sorted, r, axis = 0)
                r = range_interpolator(az_final[sweep])
                field_interpolator = interp2d(ranges, azi_sorted, field_sorted)
                field_interp = field_interpolator(ranges, az_final[sweep])
                field_sorted = np.round(field_interp, 3)
             
            # mask out areas of the interpolation where the range exceeds the unambiguous range
            field_sorted = np.ma.masked_where(r >= max_range, field_sorted)
            
            # mask previously masked gates that where filled during the interpolation
            field_sorted = np.ma.masked_equal(field_sorted, field_sorted.min())
            # if the base sweep, copy the the array creating a new axis allowing the vertical stacking of the array
            if sweep == sweep_min:
                field_final = field_sorted[np.newaxis, :]
                
            # stack the arrays at higher tilts
            else:
                field_final = np.ma.concatenate([field_final, field_sorted[np.newaxis, :]])
                
        # add the sorted and interpolated field arrays to the radar object
        radar.fields[fields]['data'] = np.ma.vstack(field_final).astype(field_type)
        radar.fields[fields]['data'] = np.ma.masked_equal(radar.fields[fields]['data'], radar.fields[fields]['data'].min())
    
    # overwrite the azimuth, elevation and number of rays arrays
    radar.azimuth['data'] = az_final.flatten().astype(field_type)
    radar.elevation['data'] = ele_final.flatten().astype(field_type)
    radar.nrays = radar.azimuth['data'].shape[0]
    
    # overwrite the ray indexes arrays
    radar.sweep_start_ray_index['data'] = np.arange(0, radar.azimuth['data'].shape[0], az_final[0].shape[0], dtype = np.int32)
    radar.sweep_end_ray_index['data'] = np.arange(radar.sweep_start_ray_index['data'][1] - 1, radar.azimuth['data'].shape[0], az_final[0].shape[0], np.int32)
    
    return radar
