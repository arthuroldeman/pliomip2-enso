import numpy as np
import xarray as xr
import mtspec
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_process import ArmaProcess

import eofs
from eofs.xarray import Eof

import warnings

warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA', FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA', FutureWarning)
warnings.filterwarnings('ignore','SerializationWarning')

def MonthConverter(X):

    V = (1.0/(X)/12.0)

    return ["%.0f" % z for z in V]

def trim_axs(axs, N):
    """little helper to massage the axs list to have correct length..."""
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]

def find_nearest(array, value, index=True, nearest=True):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if index == True and nearest == True:
        return idx, array[idx]
    elif index == True and nearest == False:
        return idx
    elif index == False and nearest == True:
        return array[idx]

def TP_variability(model = 'CCSM4', run = 'E280', latbound = 23):
    
    # Calculating the tropical Pacific SST s.d.
    
    ## latitude and longitude bounds, equatorial Pacific
    minlat = -1*latbound;  maxlat = latbound;
    minlon = 140; maxlon = 280;
    
    ## Open file 
    file = f'models/{model}/{model}_{run}.SST.timeseries_no_ann_cycle.nc'
    ds   = xr.open_dataset(file)
    
    ## Select the correct data (different names per model)
    if model == 'CCSM4' or model == 'CESM1.2' or model == 'CESM2':
        ds = ds.rename({'TS':'tos'})
    elif model == 'EC-Earth3.3' or model == 'IPSLCM5A' or model == 'IPSLCM5A2' or model == 'NorESM-L' or model == 'NorESM1-F':
        ds = ds.rename({'sst':'tos'})
    elif model == 'HadCM3':
        ds = ds.rename({'temp':'tos'})
        
    ## Select the correct lat, lon name
    if model == 'CCSM4-UoT':
        ds = ds.rename({'lat':'latitude'})
        ds = ds.rename({'lon':'longitude'})
    
    if model == 'IPSLCM5A' or model == 'IPSLCM5A2' or model == 'HadGEM3':  
        ds = ds.rename({'time_counter':'time'})
    assert 'time' in ds.dims      
        
    # 2D standard deviation of SSTs
    TPsd = ds.tos.sel(latitude = slice(minlat, maxlat)).sel(longitude = slice(minlon, maxlon)).std(dim='time');
    
    return TPsd 
    
def Zonal_SST_gradient(model = 'CCSM4', run = 'E280', latbound = 5, latmean = True):
    
    # Calculating the tropical / equatorial Pacific zonal SST gradient (merid. mean)
    # from monthly mean files (climatology)
    
    ## latitude and longitude bounds, equatorial Pacific
    minlat = -1*latbound;  maxlat = latbound;
    minlon = 140; maxlon = 280;
    
    ## Open file 
    file = f'climatology/{model}/{run}.SST.mean_month.nc'
    ds   = xr.open_dataset(file)
    
    ## Select the correct data (different names per model)
    if model == 'CCSM4' or model == 'CESM1.2' or model == 'CESM2':
        ds = ds.rename({'TS':'tos'})
    elif model == 'EC-Earth3.3' or model == 'IPSLCM5A' or model == 'IPSLCM5A2' or model == 'NorESM-L' or model == 'NorESM1-F':
        ds = ds.rename({'sst':'tos'})
    elif model == 'HadCM3':
        ds = ds.rename({'temp':'tos'})
        
    ## Select the correct lat, lon name
    if model == 'CCSM4-UoT':
        ds = ds.rename({'lat':'latitude'})
        ds = ds.rename({'lon':'longitude'})
    
    ## select correct time axis
    if model == 'HadGEM3' or model == 'NorESM-L' or model == 'NorESM1-F':
        ds = ds.rename({'time':'month'})
        
    # annual mean sst
    zonal_grad = ds.tos.mean(dim='month').sel(latitude = slice(minlat, maxlat)).sel(longitude = slice(minlon, maxlon))
        
    if latmean == True:
        zonal_grad = zonal_grad.mean(dim='latitude')
    
    return zonal_grad
        
def Nino34_timeseries(model = 'CCSM4', run = 'E280', trend = None):

    ## Calculating the Nino3.4 index time series from regridded PlioMIP2 SST data
    ##
    ## Note: at the moment NO moving average, NO detrending, NO area weighting.
    
    ## latitude and longitude bounds of Nino3.4 region
    minlat = -5;  maxlat = 5;
    minlon = 190; maxlon = 240;
    
    ## Open file 
    file = f'models/{model}/{model}_{run}.SST.timeseries_no_ann_cycle.nc'
    ds   = xr.open_dataset(file)
        
    ## rename tos, lat, etc
    if model == 'CCSM4' or model == 'CESM1.2' or model == 'CESM2':
        ds = ds.rename({'TS':'tos'})
    elif model == 'EC-Earth3.3' or model == 'IPSLCM5A' or model == 'IPSLCM5A2' or model == 'NorESM-L' or model == 'NorESM1-F':
        ds = ds.rename({'sst':'tos'})
    elif model == 'HadCM3':
        ds = ds.rename({'temp':'tos'})
        
    if model == 'CCSM4-UoT':
        ds = ds.rename({'lat':'latitude'})
        ds = ds.rename({'lon':'longitude'})    
    
    ## Calculate the mean SST in the Nino3.4 region
    Nino34 = ds.tos.where(ds.latitude>minlat).where(ds.latitude<maxlat).where(ds.longitude>minlon).where(ds.longitude<maxlon).mean(dim='latitude').mean(dim='longitude')
        
    Nino34 = Nino34[:1200] #make sure to get the last 1200months; fe CESM1.2 has 1 more year data
    
    # do trend removal and moving average here
    if trend == "linear":
        t = range(len(Nino34))
        p = np.polyfit(t, Nino34, 1)
        l = t*p[0] + p[1]
        Nino34 = Nino34 - l
    
    ## Output is a time series in xr.DataArray form
    return Nino34

def Nino_timeseries(model = 'CCSM4', run = 'E280', mode = 'Nino34', trend = None):

    ## Calculating the Nino3 / 4 / 3.4 / 1+2 index time series from regridded PlioMIP2 SST data
    ##
    ## Note: at the moment NO moving average, NO detrending, NO area weighting.
    
    ## latitude and longitude bounds 
    minlat = -5;  maxlat = 5;
    if mode == 'Nino4':
        minlon = 160; maxlon = 210;
    elif mode == 'Nino34':
        minlon = 190; maxlon = 240;
    elif mode == 'Nino3':
        minlon = 210; maxlon = 270;
    elif mode == 'Nino12':
        minlat = -10;  maxlat = 0;
        minlon = 270; maxlon = 280;
    
    ## Open file 
    file = f'models/{model}/{model}_{run}.SST.timeseries_no_ann_cycle.nc'
    ds   = xr.open_dataset(file)
    
    ## rename tos, lat, etc
    if model == 'CCSM4' or model == 'CESM1.2' or model == 'CESM2':
        ds = ds.rename({'TS':'tos'})
    elif model == 'EC-Earth3.3' or model == 'IPSLCM5A' or model == 'IPSLCM5A2' or model == 'NorESM-L' or model == 'NorESM1-F':
        ds = ds.rename({'sst':'tos'})
    elif model == 'HadCM3':
        ds = ds.rename({'temp':'tos'})
        
    if model == 'CCSM4-UoT':
        ds = ds.rename({'lat':'latitude'})
        ds = ds.rename({'lon':'longitude'}) 
    
    Nino = ds.tos.where(ds.latitude>minlat).where(ds.latitude<maxlat).where(ds.longitude>minlon).where(ds.longitude<maxlon).mean(dim='latitude').mean(dim='longitude')        
    Nino = Nino[:1200] #make sure to get the last 1200months; fe CESM1.2 has 1 more year data
    
    # do trend removal and moving average here
    if trend == "linear":
        t = range(len(Nino))
        p = np.polyfit(t, Nino, 1)
        l = t*p[0] + p[1]
        Nino = Nino - l
    
    ## Output is a time series in xr.DataArray form
    return Nino

def FFT_spectrum(time_series, normalise = True, scale = True, trend = None):

    # do trend removal and moving average here
    if trend == "linear":
        t = range(len(time_series))
        p = np.polyfit(t, time_series, 1)
        l = t*p[0] + p[1]
        time_series = time_series - l

    # Normalise
    if normalise == True:
        time_series == (time_series - np.mean(time_series))/np.std(time_series)

    # Get the Fourier Spectrum of original time series
    freq_series = np.fft.fft(time_series) #Take fourier spectrum
    freq_series = ((np.real(freq_series)**2.0) + (np.imag(freq_series)**2.0)) #Determine power law (absolute value)
    freq        = np.fft.fftfreq(len(time_series))
    freq_series_original = freq_series[1:freq.argmax()] #Restrict to f = 0.5
    freq        = freq[1:freq.argmax()] #Restrict to f = 0.5
    
    # Scale power spectrum on sum (if True)
    if scale == True:
        freq_series_original = freq_series_original / np.sum(freq_series_original)
    
    return freq, freq_series_original

def MT_spectrum(time_series, normalise = True, scale = True, trend = None):
    
    # do trend removal and moving average here
    if trend == "linear":
        t = range(len(time_series))
        p = np.polyfit(t, time_series, 1)
        l = t*p[0] + p[1]
        time_series = time_series - l
    
    # Normalise
    if normalise == True:
        time_series = (time_series - np.mean(time_series))/np.std(time_series)
        
    spec, freq = mtspec.mtspec(
                data=time_series, delta=1., time_bandwidth=2,
                number_of_tapers=3, statistics=False)
    
    spec = spec[1:freq.argmax()]
    freq = freq[1:freq.argmax()]
    
    # Scale power spectrum on sum (if True)
    if scale == True:
        spec = spec / np.sum(spec)
    
    return freq, spec

def Confidence_intervals(time_series, normalise=True, scale = True, trend = None, mode = 'FFT', N_surrogates = 1000):
    
    # do trend removal and moving average here
    if trend == "linear":
        t = range(len(time_series))
        p = np.polyfit(t, time_series, 1)
        l = t*p[0] + p[1]
        time_series = time_series - l

    if type(time_series) == xr.core.dataarray.DataArray:
        time_series = time_series.values
    
    # normalise
    if normalise == True:
        time_series = (time_series - np.mean(time_series))/np.std(time_series)

    #First determine the auto-correlation for each time series
    N = 250
    auto_lag  = np.arange(N)
    auto_corr = np.zeros(len(auto_lag))

    for lag_i in range(len(auto_lag)):
        #Determine the auto-correlation
        auto_corr[lag_i] = np.corrcoef(time_series[0:len(time_series)-lag_i], time_series[lag_i:])[0][1]

    #Determine the e-folding time and keep the minimum lag
    e_1 =  np.where(auto_corr < 1.0/np.e)[0][0]

    #Determine the first coefficient for the AR(1) process
    #The auto-lag(1) is equivalent to -1.0/(a - 1)
    a   = -1.0/(e_1) + 1.0 

    #Determine the variance in the time series and the last coefficient for the AR(1) process
    var = np.var(time_series)
#     b   = np.sqrt((1.0 - a**2.0) * var)
    b   = np.sqrt(var)
    
    # generate Monte Carlo sample 
    mc_series = mc_ar1_ARMA(phi=a, std=b, n=len(time_series), N=N_surrogates)
    
    for surrogate_i in range(N_surrogates):
        #Generating surrogate spectra
        
        # select spectrum type
        if mode == 'FFT':
            freq, spec = FFT_spectrum(mc_series[surrogate_i,:], normalise=False, scale=scale)
        elif mode == 'MT':
            freq, spec = MT_spectrum(mc_series[surrogate_i,:], normalise=False, scale=scale)
            
        # create array for spectra    
        if surrogate_i==0: 
            surrogate_spec = np.ma.masked_all((N_surrogates, len(freq)))
        
        # allocate surrogate spectrum
        surrogate_spec[surrogate_i] = spec

    CI_90 = np.percentile(surrogate_spec, 90, axis = 0)
    CI_95 = np.percentile(surrogate_spec, 95, axis = 0)
    CI_99 = np.percentile(surrogate_spec, 99, axis = 0)
        
    return CI_90, CI_95, CI_99

def mc_ar1_ARMA(phi, std, n, N=1000):
    """ Monte-Carlo AR(1) processes
    input:
    phi .. (estimated) lag-1 autocorrelation
    std .. (estimated) standard deviation of noise
    n   .. length of original time series
    N   .. number of MC simulations 
    """
    AR_object = ArmaProcess(np.array([1, -phi]), np.array([1]), nobs=n)
    mc = AR_object.generate_sample(nsample=(N,n), scale=std, axis=1, burnin=1000)
    
    return mc

def EOF_SST_analysis(model = 'CCSM4', run = 'E280', latbound = 23, weights=None, n=1):
    
    """ Empirical Orthogonal Function analysis of SST(t,x,y) field;  """

    ## latitude and longitude bounds of tropical Pacific
    minlat = -latbound;  maxlat = latbound;
    minlon = 140; maxlon = 280;
    
    ## Open file 
    file = f'models/{model}/{model}_{run}.SST.timeseries_no_ann_cycle.nc'
    ds   = xr.open_dataset(file)
        
    ## rename tos, lat, etc
    if model == 'CCSM4' or model == 'CESM1.2' or model == 'CESM2':
        ds = ds.rename({'TS':'tos'})
    elif model == 'EC-Earth3.3' or model == 'IPSLCM5A' or model == 'IPSLCM5A2' or model == 'NorESM-L' or model == 'NorESM1-F':
        ds = ds.rename({'sst':'tos'})
    elif model == 'HadCM3':
        ds = ds.rename({'temp':'tos'})
        
    if model == 'CCSM4-UoT':
        ds = ds.rename({'lat':'latitude'})
        ds = ds.rename({'lon':'longitude'})
        
    if model == 'IPSLCM5A' or model == 'IPSLCM5A2' or model == 'HadGEM3':  
        ds = ds.rename({'time_counter':'time'})
    assert 'time' in ds.dims      
    
    assert type(ds.tos)==xr.core.dataarray.DataArray
    if weights!=None:
        assert type(weights)==xr.core.dataarray.DataArray
        assert np.shape(ds.tos[0,:,:])==np.shape(weights)

    ## Select SSTs in tropical pacific
    ds = ds.sortby(ds.latitude)
    sst_TP = ds.tos.sel(latitude = slice(minlat, maxlat)).sel(longitude = slice(minlon, maxlon))
        
    # Retrieve the leading EOF, expressed as the covariance between the leading PC
    # time series and the input xa anomalies at each grid point.
    solver = Eof(sst_TP, weights=weights, center=True)
    eofs = solver.eofsAsCovariance(neofs=n)
    pcs  = solver.pcs(npcs=n, pcscaling=1)
#    eigs = solver.eigenvalues(neigs=n)
    varF = solver.varianceFraction(neigs=n)
    data = xr.merge([eofs, pcs, varF])
     
    return data