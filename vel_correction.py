import pyart
import numpy as np
from scipy.ndimage import median_filter

#speed of light constant
c = 3e8 

def dual_nyquist(radar):
    """
    This function evaluates de staggered PRT Nyquist velocity as in Zrnic and Mahapatra (1985)
    
    Parameters
    __________
    
    radar: Radar 
        Radar object used
  
    Returns
    __________
    
    Radar object with correct Nyquist velocity.
    """
    
    #get radar wavelenghth
    lamb = c/radar.instrument_parameters['frequency']['data']
    
    #evaluate the extended Nyquist interval and add it to the radar object
    prt_ratio = radar.instrument_parameters['prt_ratio']['data'][0]
    nyquist = lamb/(4*radar.instrument_parameters['prt']['data'][0]*((1/prt_ratio)-1))
    radar.instrument_parameters['nyquist_velocity']['data'] = np.resize(nyquist, radar.nrays)
    
def staggered_to_dual(radar):
    """
    This function modifies some of the instrument parameters to make it look like the radar is operating in dual-PRF mode
    
    Parameters
    __________
    
    radar: Radar 
        Radar object used
  
    Returns
    __________
    
    Radar object with correct dummy dual-PRF instrument parameters.
    """
    
    #evaluate the correct Nyquist velocity and add it to the radar object
    dual_nyquist(radar)
    
    #check for sweeps that use the staggered PRT method and change them to dual-PRF mode
    for n, prt_mode in enumerate(radar.instrument_parameters['prt_mode']['data']):
        if prt_mode =='staggered':
            radar.instrument_parameters['prt_mode']['data'][n] = b'dual'
        if prt_mode =='fixed':
            radar.instrument_parameters['prt_mode']['data'][n] = b'fixed'
    radar.instrument_parameters['prt_mode']['data'] = radar.instrument_parameters['prt_mode']['data'].astype('|S5')
    
    #invert the PRT ratio to look like the PRF ratio
    radar.instrument_parameters['prt_ratio']['data'] = 1 / radar.instrument_parameters['prt_ratio']['data']
    
    #add a PRF flag dictionary
    radar.instrument_parameters['prf_flag'] = {'units': 'unitless',
                                           'comments': 'PRF used to collect ray. 0 for high PRF, 1 for low PRF.',
                                           'meta_group': 'instrument_parameters',
                                           'long_name': 'PRF flag',
                                           'data':np.resize(([0, 1]), radar.nrays)}
    
def smooth_vel(radar, vel_name = 'vcor_cmean'):
    """
    This function applies a 3x3 median filter moving window in order to correct remaining speckles of adequately corrected velocity at the expense of smoothing some extremes.
    
    Parameters
    __________
    
    radar: Radar 
        Radar object used
        
    vel_name: str
        velocity field name to be corrected
  
    Returns
    __________
    
    adds the smoothed velocity field to the radar object.
    """
    
    #smooth the velocity field
    vel_smooth = median_filter(radar.fields[vel_name]['data'], 3)
    
    #mask undesirable gates
    smooth_data_ma = np.ma.masked_where(np.ma.getmask(radar.fields[vel_name]['data']), vel_smooth)
    smooth_data_ma = np.ma.masked_equal(smooth_data_ma, smooth_data_ma.min())
    
    #add field to radar object
    radar.add_field_like('vcor_cmean', 'vcor_cmean_smooth', 
                         smooth_data_ma, replace_existing = True)