from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
# LWP: added Any for later use
from typing import cast, Optional, Union, Any
import numpy
# LWP moved the frames to the astropy only case
from ... import io#, ECEF, LTP

import os
grand_astropy = True
try:
    if os.environ['GRAND_ASTROPY']=="0":
        grand_astropy=False
except:
    pass


if grand_astropy:
    from ... import ECEF, LTP
    from astropy.coordinates import BaseRepresentation, CartesianRepresentation
    import astropy.units as u

__all__ = ['Antenna', 'AntennaModel', 'ElectricField', 'MissingFrameError',
           'Voltage']


_logger = getLogger(__name__)


@dataclass
class ElectricField:
    if grand_astropy:
        t: u.Quantity
        E: BaseRepresentation
        r: Union[BaseRepresentation, None] = None    
        type0 = Union[ECEF, LTP, None]
    else:
        t: Any
        E: Any
        r: Any
        type0 = Any
    
    frame: type0 = None
        
    @classmethod
    def load(cls, node: io.DataNode):
        _logger.debug(f'Loading E-field from {node.filename}:{node.path}')

        t = node.read('t', dtype='f8')
        E = node.read('E', dtype='f8')

        try:
            r = node.read('r', dtype='f8')
        except KeyError:
            r = None

        try:
            frame = node.read('frame')
        except KeyError:
            frame = None

        return cls(t, E, r, frame)

    def dump(self, node: io.DataNode):
        _logger.debug(f'Dumping E-field to {node.filename}:{node.path}')

        node.write('t', self.t, unit='ns', dtype='f4')
        node.write('E', self.E, unit='uV/m', dtype='f4')

        if self.r is not None:
            node.write('r', self.r, unit='m', dtype='f4')

        if self.frame is not None:
            node.write('frame', self.frame)


@dataclass
class Voltage:
    if grand_astropy:
        t: u.Quantity
        V: u.Quantity
    else:
        t: Any
        V: Any
    
    @classmethod
    def load(cls, node: io.DataNode):
        _logger.debug(f'Loading voltage from {node.filename}:{node.path}')
        t = node.read('t', dtype='f8')
        V = node.read('V', dtype='f8')
        return cls(t, V)

    def dump(self, node: io.DataNode):
        _logger.debug(f'Dumping E-field to {node.filename}:{node.path}')
        node.write('t', self.t, unit='ns', dtype='f4')
        node.write('V', self.V, unit='uV', dtype='f4')


class AntennaModel:
    def effective_length(self, direction: BaseRepresentation,
        frequency: u.Quantity) -> CartesianRepresentation:
        pass


class MissingFrameError(ValueError):
    pass


@dataclass
class Antenna:
    model: AntennaModel
    # LWP: shower-event.py does not test the Antenna now, so not sure if these astropy removals would work
    if grand_astropy:
        type0 = Union[ECEF, LTP, None]
        type1 = Union[ECEF, LTP, BaseRepresentation]
        type2 = Union[ECEF, LTP, None]
    else:
        type0 = Any
        type1 = Any
        type2 = Any

    frame: type0 = None        

    def compute_voltage(self, direction: type1,
            field: ElectricField, frame: type2=None)          \
            -> Voltage:
        # Uniformise the inputs
        if self.frame is None:
            antenna_frame = None
            if (frame is not None) or                                          \
               (not isinstance(field.E, BaseRepresentation)) or                \
               (not isinstance(direction, BaseRepresentation)):
                raise MissingFrameError('missing antenna frame')
            else:
                E_frame, dir_frame = None, None
                E = field.E
        else:
            if grand_astropy:
                antenna_frame = cast(Union[ECEF, LTP], self.frame)
            else:
                antenna_frame = self.frame
            frame_required = False

            if field.frame is None:
                E_frame, frame_required = frame, True
            else:
                E_frame = field.frame

            if grand_astropy:
                if isinstance(direction, BaseRepresentation):
                    dir_frame, frame_required = frame, True
                else:
                    dir_frame = direction
            else:
                dir_frame = direction

            if frame_required and (frame is None):
                raise MissingFrameError('missing frame')

        # Compute the voltage
        def rfft(q):
            if grand_astropy:
                return numpy.fft.rfft(q.value) * q.unit
            else:
                return numpy.fft.rfft(q)

        def irfft(q):
            if grand_astropy:
                return numpy.fft.irfft(q.value) * q.unit
            else:
                return numpy.fft.irfft(q)

        def fftfreq(n, t):
            if grand_astropy:
                dt = (t[1] - t[0]).to_value('s')
                return numpy.fft.fftfreq(n, dt) * u.Hz
            else:
                # LWP: ToDo: Now they are ns, so I convert to s here, but probably should be converted earlier
                dt = (t[1] - t[0])*1e-9
                return numpy.fft.fftfreq(n, dt)
	
#        print("E not repr", field.E)
        # LWP: Seems unnecessary at least in the tested case - field.E already in cartesian
        if grand_astropy:
            E = field.E.represent_as(CartesianRepresentation)
            Ex = rfft(E.x)
            Ey = rfft(E.y)
            Ez = rfft(E.z)            
        else:
            E = field.E
            Ex = rfft(E[:,0])
            Ey = rfft(E[:,1])
            Ez = rfft(E[:,2])            
            
#        print("E cart", E)
#        exit()
	

        f = fftfreq(Ex.size, field.t)

#        print("dir post1", direction)
#        exit()


        if dir_frame is not None:
            # Change the direction to the antenna frame
            # LWP: ToDo: move parts outside of this if?
            # LWP: this joins direction and dir_frame - not needed if we get rid of astropy
            if grand_astropy:
                if isinstance(direction, BaseRepresentation):
                    print("dir pre", direction, dir_frame, antenna_frame)
                    direction = dir_frame.realize_frame(direction)
                direction = direction.transform_to(antenna_frame) # Now compute direction vector data in antenna frame
                print("dir1", direction)
                direction = direction.data
                print("dir2", direction)                
            else:
                E = field.E
                # Bases are in meters
                #base_diff = numpy.array([dir_frame.location.x.value-antenna_frame.location.x.value, dir_frame.location.y.value-antenna_frame.location.y.value, dir_frame.location.z.value-antenna_frame.location.z.value])/1000
                # LWP: The frames are NWU and ENU orientation, yet their values are as similar as they had the same orientation...
                print(antenna_frame, dir_frame)
                #print(base_diff)
                # LWP: ToDo: Need to check the orientations of frames and perhaps reposition the x,y,z
                #direction = numpy.array([direction.x.value,direction.y.value,direction.z.value])+base_diff
                # LWP: Valentin: The fact that direction gets offset was a bug. The direction should only be rotated actually since it is a vector, not a point.
                # But both antenna_frame and dir_frame are NWU, so I guess no rotation needed either
                # direction = numpy.array([direction.x.value,direction.y.value,direction.z.value])
                direction = numpy.array(direction)
                print(direction)
                #exit()
                
#            print("dir post1", direction)
#            exit()
            


        Leff:CartesianRepresentation
        Leff = self.model.effective_length(direction, f)
        if antenna_frame is not None:
            if grand_astropy:
                # Change the effective length to the E-field frame
                print("Leff first", Leff)
#                exit()
                tmp = antenna_frame.realize_frame(Leff)
                print("tmp1", tmp)
                tmp = tmp.transform_to(E_frame)
                print("tmp2", tmp)            
                # Transorm from antenna_frame to E_frame is a translation.
                # The Leff vector norm should therefeore not be modified, but it is because it is defined as a point in astropy coordinates
                #print(Leff,"\n", tmp.cartesian)
                print("aE", antenna_frame, E_frame)
                #print(Leff.shape, Leff)
                
                #exit()            
                Leff = tmp.cartesian
            else:
                print("Leff first", Leff)
#                exit()
                # LWP: ToDo: no idea if it is right, since I don't understand the astropy results above
                print(antenna_frame[0])
                print(antenna_frame[0], antenna_frame[1], antenna_frame[2])
#                base_diff = numpy.array([E_frame.location.x.value-antenna_frame[0], E_frame.location.y.value-antenna_frame[1], E_frame.location.z.value-antenna_frame[2]])
                base_diff = E_frame-antenna_frame
                print("sh", Leff.shape, Leff)
                print(base_diff, antenna_frame, E_frame)
                #exit()
                Leff+=base_diff

        print("Leff", Leff)
#        exit()

        # Here we have to do an ugly patch for Leff values to be correct
        if grand_astropy:
            V = irfft(Ex * (Leff.x  - Leff.x[0]) + Ey * (Leff.y - Leff.y[0]) + Ez * (Leff.z - Leff.z[0]))
        else:
            print(Ex.shape, Leff.shape)
            Ex*(Leff[:,0]  - Leff[0,0])
            V = irfft(Ex * (Leff[:,0]  - Leff[0,0]) + Ey * (Leff[:,1] - Leff[0,1]) + Ez * (Leff[:,2] - Leff[0,2]))            
        t = field.t
        t = t[:V.size]

        return Voltage(t=t, V=V)
