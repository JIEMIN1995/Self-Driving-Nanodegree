import numpy as np
import collections

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, buffer_len=10):
        # was the line detected in the last iteration
        self.detected = False
        
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]

        # list of polynomial coefficients of the last N iterations
        self.recent_poly = collections.deque(maxlen = buffer_len)
        self.recent_meter = collections.deque(maxlen = 2*buffer_len)        

        # polynomial coefficients averaged over the last n iterations
        self.average_fit = None
        
        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30/700 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        
    def clearBuffer(self):
        self.recent_poly.clear()
        
    def measure(self):
        y_eval = 719
        
        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(self.ally*self.ym_per_pix, self.allx*self.xm_per_pix, 2)
        
        # Calculate the new radii of curvature
        curverad = ((1 + (2*fit_cr[0]*y_eval*self.ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        
        # Now our radius of curvature is in meters
        return curverad
        
    def update(self, nonzerox, nonzeroy, lane_inds):
        self.detected = True
        # Update the current polynomial fit and x,y pixel values 
        # refering to the left, right line pixels extracted
        self.allx = nonzerox[lane_inds]
        self.ally = nonzeroy[lane_inds]

        # Fit a second order polynomial to each
        self.current_fit = np.polyfit(self.ally, self.allx, 2)
        
        # store the current fit in the recent polynomial coefficients
        self.recent_poly.append(self.current_fit)
        
        # compute the polynomial coefficients averaged over last n iterations
        self.average_fit = np.mean(self.recent_poly, 0)