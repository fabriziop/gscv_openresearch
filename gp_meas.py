#!/usr/bin/env python3
# .+
# .context    : SDR common view time trasfer
# .title      : plot client and server data
# .kind       : python script
# .author     : Fabrizio Pollastri <f.pollastri@inrim.it>
# .site       : Torino - Italy
# .creation   : 30-Nov-2023
# .copyright  : (c) 2023 Fabrizio Pollastri
# .license    : all right reserved
# .description
# 
# REQUIRES activated pyvenv where is installed the allatools module.
# .-


import allantools as at
import numpy as np
import scipy.signal as sg
import scipy.stats as st
import matplotlib.pyplot as pl
import os, sys


PEAK_TO_PEAK_GRADE1 = 0.95
PEAK_TO_PEAK_GRADE2 = 0.99
PEAK_TO_PEAK_GRADE3 = 0.999
MOVING_AVERAGE_WINDOW = 120


# program identification
__script__ = os.path.basename(__file__)
__version__ = '0.1.0'
__author__ = 'Fabrizio Pollastri <f.pollastri@inrim.it>'

# function
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

## init option parsing
import argparse as ap
oparser = ap.ArgumentParser(prog=__script__,description=__version__,
    epilog=__author__)
#oparser.add_argument("infname",type=ap.FileType('r'))
oparser.add_argument("-b","--bias",type=float,required=True,
    help="subtract bias from marker phase offset")
oparser.add_argument("-c","--clock",action="store_true",
    help="show also clock frequency correction")
oparser.add_argument("-q","--qmin",type=float,default=0.,
    help="remove data with marker quality below given threshold")
oparser.add_argument("-s","--skip",type=int,default=[100], nargs="+",
    help="skip samples [FROM [TO]] ")
oparser.add_argument("infname",type=str,
    help="input data file name")
oargs = oparser.parse_args()
if oargs.skip == []:
    oargs.skip = [100]

# read in data file
#timestamps, marker_offsets, signal_levels, qualities = \
data = np.loadtxt(oargs.infname,delimiter=',',unpack=True)

# if required skip initial N samples and cut exceeding tail
print(oargs.skip)
start = int(oargs.skip[0])
if len(oargs.skip) > 1:
    end = int(oargs.skip[1])
else:
    end = -1

# unpack data
clock_adjusts = None
if len(data) == 4:
    timestamps, marker_offsets, signal_levels, qualities = data
elif len(data) == 5:
    timestamps, marker_offsets, signal_levels, qualities, clock_adjusts = data
elif len(data) == 7:
    timestamps, marker_offsets, signal_levels, qualities, clock_adjusts, \
    corr_phase, corr_phase_dev = data
else:
    raise("error: unknown format of input data")
if not oargs.clock:
    clock_adjusts = None

# selected time slice only
print(start,end)
timestamps = timestamps[start:end]
marker_offsets = marker_offsets[start:end]
signal_levels = signal_levels[start:end]
qualities = qualities[start:end]
if type(clock_adjusts) == np.ndarray:
    clock_adjusts = clock_adjusts[start:end]

# time origin to first timestamp
timestamps -= timestamps[0]

# marker offset to ns unit with zero bias
marker_offsets -= oargs.bias
marker_offsets *= 100

# remove invalid markers: quality < qmin
valid = qualities > oargs.qmin
timestamps = timestamps[valid]
marker_offsets = marker_offsets[valid]
signal_levels = signal_levels[valid]
qualities = qualities[valid]
if type(clock_adjusts) == np.ndarray:
    clock_adjusts = clock_adjusts[valid]

# phase off peak to peak
mo_sorted = marker_offsets[np.argsort(marker_offsets)]
outliers_part = int((1. - PEAK_TO_PEAK_GRADE1) * len(mo_sorted) / 2)
pp1 = int(mo_sorted[-outliers_part-1] - mo_sorted[outliers_part])
outliers_part = int((1. - PEAK_TO_PEAK_GRADE2) * len(mo_sorted) / 2)
pp2 = int(mo_sorted[-outliers_part-1] - mo_sorted[outliers_part])
outliers_part = int((1. - PEAK_TO_PEAK_GRADE3) * len(mo_sorted) / 2)
pp3 = int(mo_sorted[-outliers_part-1] - mo_sorted[outliers_part])

# print out some statistics
print("\n==== Marker phase offset stats ===")
stats = st.describe(marker_offsets)
print(stats)
print("stddev=",np.sqrt(stats.variance),sep="")
print("peak to peak at %f%% = %d" % (PEAK_TO_PEAK_GRADE1,pp1))
print("peak to peak at %f%% = %d" % (PEAK_TO_PEAK_GRADE2,pp2))
print("peak to peak at %f%% = %d" % (PEAK_TO_PEAK_GRADE3,pp3))

print("\n==== Marker phase offset moving average stats ===")
marker_offsets_mavg = moving_average(marker_offsets,MOVING_AVERAGE_WINDOW)
stats = st.describe(marker_offsets_mavg)
print(stats)
print("stddev=",np.sqrt(stats.variance),sep="")

print("\n==== Signal level stats ===")
stats = st.describe(signal_levels)
print(stats)
print("stddev=",np.sqrt(stats.variance),sep="")

print("\n==== Marker quality ===")
stats = st.describe(qualities)
print(stats)
print("stddev=",np.sqrt(stats.variance),"\n",sep="")

if type(clock_adjusts) == np.ndarray:
    print("\n==== Clock frequency adjustment ===")
    stats = st.describe(clock_adjusts)
    print(stats)
    print("stddev=",np.sqrt(stats.variance),"\n",sep="")

# Initialise the subplot function using number of rows and columns
plot_num = 3
if type(clock_adjusts) == np.ndarray:
    plot_num += 1
pl.rcParams.update({'font.size':16})
figure, axis = pl.subplots(plot_num)
#figure.suptitle(oargs.infname)
#pl.tight_layout()
  
# plot marker offset
axis[0].plot(timestamps,marker_offsets,color="red",label="offset") 
axis[0].plot(timestamps[MOVING_AVERAGE_WINDOW-1:],marker_offsets_mavg*10,color="orange",label="mean 120s x10") 
axis[0].set_title("Marker phase offset (ns)") 
axis[0].legend(fontsize=16) 
axis[0].grid() 
  
# plot signal power (sqrt) level
axis[1].plot(timestamps,signal_levels/2.,color="green") 
axis[1].set_title("Signal power (sqrt, range 0-1)") 
axis[1].grid() 
  
# plot marker quality
axis[2].plot(timestamps,qualities,color="blue") 
axis[2].set_title("Marker quality (range 0-1)") 
axis[2].grid() 
   
# clock frequency adjustment
if type(clock_adjusts) == np.ndarray:
    axis[3].plot(timestamps,clock_adjusts,color="magenta") 
    axis[3].set_title("Clock frequency adjustment (Hz)") 
    axis[3].grid() 

pl.subplots_adjust(bottom=0.05,left=0.05,right=0.95,top=0.95,hspace=0.4)

# plot allan variance
pl.figure(2)
pl.title("allan deviation")
taus, adevs, errors, ns = at.oadev(marker_offsets,taus=range(200))
pl.grid(which="both")
pl.minorticks_on()
pl.semilogy(taus,adevs,color='blue')
pl.xlabel('tau (s)')
pl.ylabel('tdev allan deviation (ns)')
 
# Combine all the operations and display 
pl.show() 
