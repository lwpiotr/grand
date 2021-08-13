#! /usr/bin/env python
from grand.simulation import Antenna, ShowerEvent, TabulatedAntennaModel
#from grand.simulation import ShowerEvent, TabulatedAntennaModel
#from grand.simulation.antenna .generic import compute_voltage
import numpy as np
from matplotlib import pyplot as plt


import os
grand_astropy = True
try:
    if os.environ['GRAND_ASTROPY']=="0":
        grand_astropy=False
except:
    pass

if grand_astropy:
    from grand import ECEF, LTP
    import astropy.units as u
    from astropy.coordinates import BaseRepresentation, CartesianRepresentation
    
# Load the radio shower simulation data
showerdir = '../../tests/simulation/data/zhaires'
shower = ShowerEvent.load(showerdir)
if shower.frame is None:
    shower.localize(39.5 * u.deg, 90.5 * u.deg) # Coreas showers have no
                                                # localization info. This must
                                                # be set manually

print("Shower frame=",shower.frame)
print("Zenith (Zhaires?!) =",shower.zenith)
print("Azimuth (Zhaires?!) =",shower.azimuth)
print("Xmax=",shower.maximum)
#shower.core=CartesianRepresentation(0, 0, 2900, unit='m')
print("Core=",shower.core)


# Define an antenna model
#
# A tabulated model of the Butterfly antenna is used. Note that a single EW
# arm is assumed here for the sake of simplicity

antenna_model = TabulatedAntennaModel.load('./HorizonAntenna_EWarm_leff_loaded.npy')

# Loop over electric fields and compute the corresponding voltages
for antenna_index, field in shower.fields.items():
    # Compute the antenna local frame
    #
    # The antenna is placed within the shower frame. It is oriented along the
    # local magnetic North by using an ENU/LTP frame (x: East, y: North, z: Upward)
    if grand_astropy:
        # LWP: This just adds (x,y,z) from field.electric.r to shower.frame, but field.electric.r does not seem to be used later
        antenna_location = shower.frame.realize_frame(field.electric.r)
    	# LWP: antenna_location and antenna_frame are different. I am not sure where it comes from
        antenna_frame = LTP(location=antenna_location, orientation='NWU',magnetic=True, obstime=shower.frame.obstime)     
        print(antenna_index,"Antenna pos=",antenna_location)        
    else:
        # There are things in antenna_location, like separate (x,y,z) (what is it?) and declination, that are lost when creating the frame with astropy, so I guess not needed
        #antenna_frame = np.array(list(shower.frame.location.value))
        antenna_frame = np.array(list(shower.frame))

    #exit()

    antenna = Antenna(model=antenna_model, frame=antenna_frame)


    # Compute the voltage on the antenna
    #
    # The electric field is assumed to be a plane-wave originating from the
    # shower axis at the depth of maximum development. Note that the direction
    # of observation and the electric field components are provided in the
    # shower frame. This is indicated by the `frame` named argument.
    print(shower.maximum, field.electric.r)
    if grand_astropy:
        direction = shower.maximum - field.electric.r
    else:
        # LWP: convert to km
        direction = (np.array(shower.maximum) - np.array(field.electric.r))/1000
    print("Direction to Xmax = ",direction)
#    exit()
    #print(antenna_frame.realize_frame(direction))
    # LWP: Here field.electric.E seems to be already in the cartesian representation
    if grand_astropy:
        Exyz = field.electric.E.represent_as(CartesianRepresentation)
    else:
        Exyz = field.electric.E

    print(Exyz)
    #exit()
    field.voltage = antenna.compute_voltage(direction, field.electric,frame=shower.frame)
    print("computed voltage", field.voltage)
#    exit()

    plt.figure()
    plt.subplot(211)
    if grand_astropy:
        plt.plot(field.electric.t,Exyz.x,label='Ex')
        plt.plot(field.electric.t,Exyz.y,label='Ey')
        plt.plot(field.electric.t,Exyz.z,label='Ez')
        plt.xlabel('Time ('+str(field.electric.t.unit)+')')
        plt.ylabel('Efield ('+str(Exyz.x.unit)+')')
        plt.legend(loc='best')
        plt.subplot(212)
        print("V", field.voltage.V)
        plt.plot(field.voltage.t,field.voltage.V,label='V$_{EW}$')
        plt.xlabel('Time ('+str(field.voltage.t.unit)+')')
        plt.ylabel('Voltage ('+str(field.voltage.V.unit)+')')
        
    else:
        plt.plot(field.electric.t,Exyz[:,0],label='Ex')
        plt.plot(field.electric.t,Exyz[:,1],label='Ey')
        plt.plot(field.electric.t,Exyz[:,2],label='Ez')
        plt.xlabel('Time (ns)')
        plt.ylabel('Efield (uV / m)')
        plt.legend(loc='best')
        plt.subplot(212)
        print(field.voltage.V.shape)
        plt.plot(field.voltage.t,field.voltage.V,label='V$_{EW}$')
        plt.xlabel('Time (ns)')
        plt.ylabel('Voltage (uV)')
        
    plt.legend(loc='best')
    plt.show()
