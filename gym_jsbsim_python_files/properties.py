import math
import collections
from gym_jsbsim import utils


class BoundedProperty(collections.namedtuple('BoundedProperty', ['name', 'description', 'min', 'max'])):
    def get_legal_name(self):
        return utils.AttributeFormatter.translate(self.name)


class Property(collections.namedtuple('Property', ['name', 'description'])):
    def get_legal_name(self):
        return utils.AttributeFormatter.translate(self.name)


# position and attitude
altitude_sl_ft = BoundedProperty('position/h-sl-ft', 'altitude above mean sea level [ft]', -1400, 85000)
altitude_agl_ft = Property('position/h-agl-ft', 'altitude above ground level [ft]')
pitch_rad = BoundedProperty('attitude/pitch-rad', 'pitch [rad]', -0.5 * math.pi, 0.5 * math.pi)
roll_rad = BoundedProperty('attitude/roll-rad', 'roll [rad]', -math.pi, math.pi)
heading_deg = BoundedProperty('attitude/psi-deg', 'heading [deg]', 0, 360)
sideslip_deg = BoundedProperty('aero/beta-deg', 'sideslip [deg]', -180, +180)
lat_geod_deg = BoundedProperty('position/lat-geod-deg', 'geocentric latitude [deg]', -90, 90)
lng_geoc_deg = BoundedProperty('position/long-gc-deg', 'geodesic longitude [deg]', -180, 180)
dist_travel_m = Property('position/distance-from-start-mag-mt', 'distance travelled from starting position [m]')

# velocities
u_fps = BoundedProperty('velocities/u-fps', 'body frame x-axis velocity [ft/s]', -2200, 2200)
v_fps = BoundedProperty('velocities/v-fps', 'body frame y-axis velocity [ft/s]', -2200, 2200)
w_fps = BoundedProperty('velocities/w-fps', 'body frame z-axis velocity [ft/s]', -2200, 2200)
v_north_fps = BoundedProperty('velocities/v-north-fps', 'velocity true north [ft/s]', float('-inf'), float('+inf'))
v_east_fps = BoundedProperty('velocities/v-east-fps', 'velocity east [ft/s]', float('-inf'), float('+inf'))
v_down_fps = BoundedProperty('velocities/v-down-fps', 'velocity downwards [ft/s]', float('-inf'), float('+inf'))
p_radps = BoundedProperty('velocities/p-rad_sec', 'roll rate [rad/s]', -2 * math.pi, 2 * math.pi)
q_radps = BoundedProperty('velocities/q-rad_sec', 'pitch rate [rad/s]', -2 * math.pi, 2 * math.pi)
r_radps = BoundedProperty('velocities/r-rad_sec', 'yaw rate [rad/s]', -2 * math.pi, 2 * math.pi)
altitude_rate_fps = Property('velocities/h-dot-fps', 'Rate of altitude change [ft/s]')

# controls state
aileron_left = BoundedProperty('fcs/left-aileron-pos-norm', 'left aileron position, normalised', -1, 1)
aileron_right = BoundedProperty('fcs/right-aileron-pos-norm', 'right aileron position, normalised', -1, 1)
elevator = BoundedProperty('fcs/elevator-pos-norm', 'elevator position, normalised', -1, 1)
rudder = BoundedProperty('fcs/rudder-pos-norm', 'rudder position, normalised', -1, 1)
throttle = BoundedProperty('fcs/throttle-pos-norm', 'throttle position, normalised', 0, 1)
gear = BoundedProperty('gear/gear-pos-norm', 'landing gear position, normalised', 0, 1)
flaps = BoundedProperty('fcs/flap-pos-norm', 'flap position, normalized', 0, 1)


# --- Engine Parameters ---
engine_running = Property('propulsion/engine/set-running', 'engine running (0/1 bool)')
all_engine_running = Property('propulsion/set-running', 'set engine running (-1 for all engines)')
engine_thrust_lbs = Property('propulsion/engine/thrust-lbs', 'engine thrust [lb]')
engine_rpm = Property('propulsion/engine/rpm', 'engine RPM')
fuel_flow_lbs_sec = Property('propulsion/engine/fuel-flow-lbs_sec', 'fuel flow [lbs/sec]')
propeller_rpm = Property('propulsion/engine/prop-rpm', 'propeller RPM')
engine_ignition = Property('propulsion/engine/ignition', 'Engine ignition switch (0/1)')
engine_magnetos = Property('propulsion/magneto_cmd', 'Engine magnetos switch (0 = off, 3 = both)')
engine_starter = Property('propulsion/starter_cmd', 'Engine starter (0/1)')
engine_cutoff = Property('propulsion/engine/cutoff_cmd', 'Engine fuel cutoff (0 = ON, 1 = OFF)')

# --- Fuel System ---
fuel_tank_left = Property('propulsion/tank[0]/contents-lbs', 'Fuel left tank [lbs]')
fuel_tank_right = Property('propulsion/tank[1]/contents-lbs', 'Fuel right tank [lbs]')
fuel_total = Property('propulsion/tank/total-fuel-lbs', 'Total fuel on board [lbs]')

# --- Electrical System ---
battery_on = Property('electrical/battery-on', 'Battery switch state (0/1)')
alternator_on = Property('electrical/alternator-on', 'Alternator switch state (0/1)')
bus_voltage = Property('electrical/bus-voltage', 'Main electrical bus voltage [V]')

# controls command
aileron_cmd = BoundedProperty('fcs/aileron-cmd-norm', 'aileron commanded position, normalised', -1., 1.)
elevator_cmd = BoundedProperty('fcs/elevator-cmd-norm', 'elevator commanded position, normalised', -1., 1.)
rudder_cmd = BoundedProperty('fcs/rudder-cmd-norm', 'rudder commanded position, normalised', -1., 1.)
throttle_cmd = BoundedProperty('fcs/throttle-cmd-norm', 'throttle commanded position, normalised', 0., 1.)
mixture_cmd = BoundedProperty('fcs/mixture-cmd-norm', 'engine mixture setting, normalised', 0., 1.)
throttle_1_cmd = BoundedProperty('fcs/throttle-cmd-norm[1]', 'throttle 1 commanded position, normalised', 0., 1.)
mixture_1_cmd = BoundedProperty('fcs/mixture-cmd-norm[1]', 'engine mixture 1 setting, normalised', 0., 1.)
gear_all_cmd = BoundedProperty('gear/gear-cmd-norm', 'all landing gear commanded position, normalised', 0, 1)

# simulation
sim_dt = Property('simulation/dt', 'JSBSim simulation timestep [s]')
sim_time_s = Property('simulation/sim-time-sec', 'Simulation time [s]')

# initial conditions
initial_altitude_ft = Property('ic/h-sl-ft', 'initial altitude MSL [ft]')
initial_terrain_altitude_ft = Property('ic/terrain-elevation-ft', 'initial terrain alt [ft]')
initial_longitude_geoc_deg = Property('ic/long-gc-deg', 'initial geocentric longitude [deg]')
initial_latitude_geod_deg = Property('ic/lat-geod-deg', 'initial geodesic latitude [deg]')
initial_u_fps = Property('ic/u-fps', 'body frame x-axis velocity; positive forward [ft/s]')
initial_v_fps = Property('ic/v-fps', 'body frame y-axis velocity; positive right [ft/s]')
initial_w_fps = Property('ic/w-fps', 'body frame z-axis velocity; positive down [ft/s]')
initial_p_radps = Property('ic/p-rad_sec', 'roll rate [rad/s]')
initial_q_radps = Property('ic/q-rad_sec', 'pitch rate [rad/s]')
initial_r_radps = Property('ic/r-rad_sec', 'yaw rate [rad/s]')
initial_roc_fpm = Property('ic/roc-fpm', 'initial rate of climb [ft/min]')
initial_heading_deg = Property('ic/psi-true-deg', 'initial (true) heading [deg]')

# --- Environmental Parameters ---
initial_sun_azimuth_deg = Property('ic/sun-azimuth-deg', 'initial sun azimuth (degrees)')
initial_sun_elevation_deg = Property('ic/sun-elevation-deg', 'initial sun elevation (degrees)')
initial_sun_intensity = Property('ic/sun-intensity', 'initial sun intensity')

# --- Weather and Atmosphere ---
initial_atmospheric_pressure_hpa = Property('ic/atmospheric-pressure-hpa', 'initial atmospheric pressure [hPa]')
initial_temperature_celsius = Property('ic/temperature-celsius', 'initial temperature [°C]')
initial_wind_speed_knots = Property('ic/wind-speed-knots', 'initial wind speed [knots]')
initial_wind_direction_deg = Property('ic/wind-direction-deg', 'initial wind direction [degrees]')

# --- Day/Night Cycle ---
initial_time_of_day = Property('ic/time-of-day', 'time of day in simulation [hh:mm:ss]')  # 24-hour format

# --- Other Environmental Factors ---
initial_cloud_cover = Property('ic/cloud-cover', 'initial cloud cover percentage [%]')
initial_visibility_km = Property('ic/visibility-km', 'initial visibility distance [km]')



class Vector2(object):
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def heading_deg(self):
        """ Calculate heading in degrees of vector from origin """
        heading_rad = math.atan2(self.x, self.y)
        heading_deg_normalised = (math.degrees(heading_rad) + 360) % 360
        return heading_deg_normalised

    @staticmethod
    def from_sim(sim: 'simulation.Simulation') -> 'Vector2':
        return Vector2(sim[v_east_fps], sim[v_north_fps])


class GeodeticPosition(object):
    def __init__(self, latitude_deg: float, longitude_deg: float):
        self.lat = latitude_deg
        self.lon = longitude_deg

    def heading_deg_to(self, destination: 'GeodeticPosition') -> float:
        """ Determines heading in degrees of course between self and destination """
        difference_vector = destination - self
        return difference_vector.heading_deg()

    @staticmethod
    def from_sim(sim: 'simulation.Simulation') -> 'GeodeticPosition':
        """ Return a GeodeticPosition object with lat and lon from simulation """
        lat_deg = sim[lat_geod_deg]
        lon_deg = sim[lng_geoc_deg]
        return GeodeticPosition(lat_deg, lon_deg)

    def __sub__(self, other) -> Vector2:
        """ Returns difference between two coords as (delta_lat, delta_long) """
        return Vector2(self.lon - other.lon, self.lat - other.lat)
