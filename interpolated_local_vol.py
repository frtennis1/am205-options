import numpy as np
import scipy.interpolate as interp

class InterpolatedLocalVol:
    
    def __init__(self, point_estimates, dates, strikes, uses_dates=False):
        self.uses_dates = uses_dates
        self.point_estimates = point_estimates
        if uses_dates:
            self.raw_dates = np.array(dates)
            self.dates = np.array([d.toordinal() for d in dates])
        else:
            self.raw_dates = np.array(dates)
            self.dates = np.array(dates)
            
        self.strikes = np.array(strikes)
        self.f = interp.interp2d(self.strikes, self.dates, point_estimates, bounds_error=False)
    
    def __call__(self, raw_dates, strikes):
        if self.uses_dates:
            try:
                dates = np.array([d.toordinal() for d in raw_dates])
            except:
                dates = raw_dates.toordinal()
        else:
            dates = raw_dates
        
#         print(strikes)
#         print(dates)
        return self.f(strikes, dates)