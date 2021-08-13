from __future__ import annotations

from datetime import datetime
from logging import getLogger
from pathlib import Path
import re
from typing import Any, Dict, Optional

import h5py
import numpy

from .generic import CollectionEntry, FieldsCollection, ShowerEvent
from ..antenna import ElectricField
from ..pdg import ParticleCode


import os
grand_astropy = True
try:
    if os.environ['GRAND_ASTROPY']=="0":
        grand_astropy=False
except:
    pass

if grand_astropy:
    from ...tools.coordinates import ECEF, LTP
    from astropy.coordinates import BaseCoordinateFrame, CartesianRepresentation,  \
                                    PhysicsSphericalRepresentation
    import astropy.units as u


__all__ = ['InvalidAntennaName', 'ZhairesShower']


logger = getLogger(__name__)


class InvalidAntennaName(ValueError):
    pass


class ZhairesShower(ShowerEvent):
    @classmethod
    def _check_dir(cls, path: Path) -> bool:
        try:
            info_file = path.glob('*.sry').__next__()
        except StopIteration:
            return False
        return True

    @classmethod
    def _from_dir(cls, path: Path) -> ZhairesShower:   
        if not path.exists():
            raise FileNotFoundError(path)

        positions = {}
        ant_file = path / 'antpos.dat'
        if ant_file.exists():
            pattern = re.compile('A([0-9]+)$')
            with ant_file.open() as f:
                for line in f:
                    if not line: continue
                    words = line.split()

                    match = pattern.search(words[1])
                    if match is None:
                        raise InvalidAntennaName(words[1])
                    antenna = int(match.group(1))

                    if grand_astropy:
                        positions[antenna] = CartesianRepresentation(
                            x = float(words[2]) * u.m,
                            y = float(words[3]) * u.m,
                            z = 0 * u.m
                            #z = float(words[4]) * u.m
                        )
                    else:
                        positions[antenna] = (float(words[2]), float(words[3]),0)
                    
                    #print("### Warning: Forcing antenna height = 0m")

        fields: Optional[FieldsCollection] = None
        raw_fields = {}
        for field_path  in path.glob('a*.trace'):
            antenna = int(field_path.name[1:].split('.', 1)[0])
            logger.debug(f'Loading trace for antenna {antenna}')
            data = numpy.loadtxt(field_path)
            if grand_astropy:
                uVm = u.uV / u.m
                t  = data[:,0] * u.ns
                Ex = data[:,1] * uVm
                Ey = data[:,2] * uVm
                Ez = data[:,3] * uVm
                electric = ElectricField(
                    t,
                    CartesianRepresentation(Ex, Ey, Ez),
                    positions[antenna]
                )
#                print(electric)
                
            else:
                #LWP: ToDo: change to seconds? They are ns now
                t  = data[:,0]
                Ex = data[:,1]
                Ey = data[:,2]
                Ez = data[:,3]
                electric = ElectricField(t, numpy.stack([Ex, Ey, Ez], axis=1), positions[antenna])
#                print(electric)
#            exit()                
                            
            raw_fields[antenna] = CollectionEntry(electric)

        if raw_fields:
            fields = FieldsCollection()
            for key in sorted(raw_fields.keys()):
                fields[key] = raw_fields[key]

        inp: Dict[str, Any] = {}
        if grand_astropy:
            inp['core'] = CartesianRepresentation(0, 0, 0, unit='m')
        else:
            inp['core'] = (0,0,0)
        print("inp", inp)
#        exit()
        
        try:
            sry_path = path.glob('*.sry').__next__()
        except StopIteration:
            raise FileNotFoundError(path / '*.sry')
        else:
            def parse_primary(string: str) -> ParticleCode:
                return {
                    'Proton': ParticleCode.PROTON,
                    'Iron': ParticleCode.IRON
                }[string.strip()]

            def parse_quantity(string: str) -> u.Quantity:
                words = string.split()
                if grand_astropy:
                    return float(words[0]) * u.Unit(words[1])
                else:
                    return float(words[0])                

            def parse_frame_location(string: str) -> BaseCoordinateFrame:
                lat, lon = string.split('Long:')
                lat = parse_quantity(lat[:-2])
                lon = parse_quantity(lon[:-3])
                if grand_astropy:                    
                    return ECEF(lat, lon, 0 * u.m, representation_type='geodetic')
                else:
                    return (float(lat), float(lon), 0)

            def parse_date(string: str) -> datetime:
                return datetime.strptime(string.strip(), '%d/%b/%Y')

            def parse_frame_direction(string: str) -> BaseCoordinateFrame:
                inp['_origin'] = inp['frame']

                string = string.strip()
                if string == 'Local magnetic north':
                    return 'NWU'
                else:
                    raise NotImplementedError(string)

            def parse_geomagnet_intensity(string: str) -> u.Quantity:
                if grand_astropy:
                    return float(string.split()[0]) << u.uT
                else:
                    return float(string.split()[0])                

            def parse_geomagnet_angles(string: str) -> CartesianRepresentation:
                intensity = inp['geomagnet']
                inclination, _, _, declination, _ = string.split()
                if grand_astropy:
                    theta = (90 + float(inclination)) << u.deg
                    inp['_declination'] = float(declination) << u.deg
                    spherical = PhysicsSphericalRepresentation(theta=theta,
                        phi=0 << u.deg, r=intensity)
                    return spherical.represent_as(CartesianRepresentation)
                else:
                    # LWP: ToDo: Should be changed to radians! But that requires checking the code that uses it later, if it accepts radians
                    theta = (90 + float(inclination))
                    inp['_declination'] = float(declination)
                    return (intensity*numpy.cos(0)*numpy.sin(numpy.deg2rad(theta)),intensity*numpy.sin(0)*numpy.sin(numpy.deg2rad(theta)),intensity*numpy.cos(numpy.deg2rad(theta)))
                

            def parse_maximum(string: str) -> CartesianRepresentation:
                _, _, *xyz = string.split()
                x, y, z = map(float, xyz)
                ## Here we have to express the Xmax height in the correct LTP defined at ground level (eg z(antenna) = 0)
                ## Hence we have to correct z_Xmax by "GroundAltitude" read from ZHaires input file
                ## Dirty hack by OMH for now
                try:
                    inp_file = path.glob('*.inp').__next__()
                    print("### zhaires.py: reading groundaltitude from. inp file.")
                    with open(inp_file) as f:
                      for line in f:
                        if 'GroundAltitude' in line:
                            ground_alt = float(line.split()[1])
                except StopIteration:
                    raise FileNotFoundError(path / '*.inp')
                    
                if grand_astropy:
                    return CartesianRepresentation(x * u.km, y * u.km, (z-ground_alt/1000) * u.km)
                else:
                    # LWP: Change to meters, for compatibility with field.electric.r in shower-event.py
                    return (x*1000, y*1000, (z-ground_alt/1000)*1000)

            converters = (
                ('(Lat', 'frame', parse_frame_location),
                ('Date', '_obstime', parse_date),
                ('Primary particle', 'primary', parse_primary),
                ('Primary energy', 'energy', parse_quantity),
                ('Primary zenith angle', 'zenith', parse_quantity),
                ('Primary azimuth angle', 'azimuth', parse_quantity),
                ('Zero azimuth direction', 'frame', parse_frame_direction),
                ('Geomagnetic field: Intensity:', 'geomagnet',
                    parse_geomagnet_intensity),
                ('I:', 'geomagnet', parse_geomagnet_angles),
                ('Location of max.(Km)', 'maximum', parse_maximum)
            )

            i = 0
            tag, k, convert = converters[i]
            with sry_path.open() as f:
                for line in f:
                    start = line.find(tag)
                    if start < 0: continue

                    inp[k] = convert(line[start+len(tag)+1:])
                    print("inp", k, inp[k], type(inp[k]))
                    i = i + 1
                    try:
                        tag, k, convert = converters[i]
                    except IndexError:
                        break

        origin = inp.pop('_origin')
        declination = inp.pop('_declination')
        obstime = inp.pop('_obstime')
        orientation = inp['frame']
        if grand_astropy:
        	inp['frame'] = LTP(location=origin, orientation=orientation,
                           declination=declination, obstime=obstime)
        else:
            inp['frame'] = origin

        return cls(fields=fields, **inp)


    @classmethod
    def _from_datafile(cls, path: Path) -> ZhairesShower:  
        with h5py.File(path, 'r') as fd:
            if not 'RunInfo.__table_column_meta__' in fd['/']:
                return super()._from_datafile(path)

            for name in fd['/'].keys():
                if not name.startswith('RunInfo'):
                    break

            event = fd[f'{name}/EventInfo']
            antennas = fd[f'{name}/AntennaInfo']
            traces = fd[f'{name}/AntennaTraces']

            fields = FieldsCollection()

            pattern = re.compile('([0-9]+)$')
            for tag, x, y, z, *_ in antennas:
                tag = tag.decode()
                antenna = int(pattern.search(tag)[1])
                r = CartesianRepresentation(
                    float(x), float(y), float(z), unit=u.m)
                tmp = traces[f'{tag}/efield'][:]
                efield = tmp.view('f4').reshape(tmp.shape + (-1,))
                t = numpy.asarray(efield[:,0], 'f8') << u.ns
                Ex = numpy.asarray(efield[:,1], 'f8') << u.uV / u.m
                Ey = numpy.asarray(efield[:,2], 'f8') << u.uV / u.m
                Ez = numpy.asarray(efield[:,3], 'f8') << u.uV / u.m
                E = CartesianRepresentation(Ex, Ey, Ez, copy=False),

                fields[antenna] = CollectionEntry(
                    electric=ElectricField(t = t, E = E, r = r))

            primary = {
                'Fe^56'  : ParticleCode.IRON,
                'Gamma'  : ParticleCode.GAMMA,
                'Proton' : ParticleCode.PROTON
            }[event[0, 'Primary'].decode()]

            geomagnet = PhysicsSphericalRepresentation(
                theta = float(90 + event[0, 'BFieldIncl']) << u.deg,
                phi = 0 << u.deg,
                r = float(event[0, 'BField']) << u.uT)

            try:
                latitude = event[0, 'Latitude'] << u.deg
                longitude = event[0, 'Longitude'] << u.deg
                declination = event[0, 'BFieldDecl'] << u.deg
                obstime = datetime.strptime(event[0, 'Date'].strip(),
                                            '%d/%b/%Y')
            except ValueError:
                frame = None
            else:
                if grand_astropy:
                    origin = ECEF(latitude, longitude, 0 * u.m, representation_type='geodetic')                
                    frame = LTP(location=origin, orientation='NWU',
                            declination=declination, obstime=obstime)
                else:
                    origin = (latitude, longitude, 0)
                    frame = origin

            return cls(
                energy = float(event[0, 'Energy']) << u.EeV,
                zenith = (180 - float(event[0, 'Zenith'])) << u.deg,
                azimuth = -float(event[0, 'Azimuth']) << u.deg,
                primary = primary,

                frame = frame,
                core = CartesianRepresentation(0, 0, 0, unit='m'),
                geomagnet = geomagnet.represent_as(CartesianRepresentation),
                maximum = CartesianRepresentation(*event[0, 'XmaxPosition'],
                                                  unit='m'),

                fields = fields
            )
