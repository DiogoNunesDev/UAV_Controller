import gym
import numpy as np
import random
import types
import math
import enum
import warnings
from collections import namedtuple
import gym_jsbsim.properties as prp
from gym_jsbsim import assessors, rewards, utils
from gym_jsbsim.simulation import Simulation
from gym_jsbsim.properties import BoundedProperty, Property
from gym_jsbsim.aircraft import Aircraft
from gym_jsbsim.rewards import RewardStub
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Dict, Tuple, NamedTuple, Type
from gym import spaces


class Task(ABC):
    """
    Interface for Tasks, modules implementing specific environments in JSBSim.

    A task defines its own state space, action space, termination conditions and agent_reward function.
    """

    @abstractmethod
    def task_step(self, sim: Simulation, action: Sequence[float], sim_steps: int) \
            -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Calculates new state, reward and termination.

        :param sim: a Simulation, the simulation from which to extract state
        :param action: sequence of floats, the agent's last action
        :param sim_steps: number of JSBSim integration steps to perform following action
            prior to making observation
        :return: tuple of (observation, reward, done, info) where,
            observation: array, agent's observation of the environment state
            reward: float, the reward for that step
            done: bool, True if the episode is over else False
            info: dict, optional, containing diagnostic info for debugging etc.
        """

    ...

    @abstractmethod
    def observe_first_state(self, sim: Simulation) -> np.ndarray:
        """
        Initialise any state/controls and get first state observation from reset sim.

        :param sim: Simulation, the environment simulation
        :return: np array, the first state observation of the episode
        """
        ...

    @abstractmethod
    def get_initial_conditions(self) -> Optional[Dict[Property, float]]:
        """
        Returns dictionary mapping initial episode conditions to values.

        Episode initial conditions (ICs) are defined by specifying values for
        JSBSim properties, represented by their name (string) in JSBSim.

        JSBSim uses a distinct set of properties for ICs, beginning with 'ic/'
        which differ from property names during the simulation, e.g. "ic/u-fps"
        instead of "velocities/u-fps". See https://jsbsim-team.github.io/jsbsim/

        :return: dict mapping string for each initial condition property to
            initial value, a float, or None to use Env defaults
        """
        ...

    @abstractmethod
    def get_state_space(self) -> gym.Space:
        """ Get the task's state Space object """
        ...

    @abstractmethod
    def get_action_space(self) -> gym.Space:
        """ Get the task's action Space object """
        ...


class FlightTask(Task, ABC):
    """
    Abstract superclass for flight tasks.

    Concrete subclasses should implement the following:
        state_variables attribute: tuple of Propertys, the task's state representation
        action_variables attribute: tuple of Propertys, the task's actions
        get_initial_conditions(): returns dict mapping InitialPropertys to initial values
        _is_terminal(): determines episode termination
        (optional) _new_episode_init(): performs any control input/initialisation on episode reset
        (optional) _update_custom_properties: updates any custom properties in the sim
    """
    INITIAL_ALTITUDE_FT = 5000
    base_state_variables = (prp.altitude_sl_ft, prp.pitch_rad, prp.roll_rad,
                            prp.u_fps, prp.v_fps, prp.w_fps,
                            prp.p_radps, prp.q_radps, prp.r_radps,
                            prp.aileron_left, prp.aileron_right, prp.elevator,
                            prp.rudder)
    base_initial_conditions = types.MappingProxyType(  # MappingProxyType makes dict immutable
        {prp.initial_altitude_ft: INITIAL_ALTITUDE_FT,
         prp.initial_terrain_altitude_ft: 0.00000001,
         prp.initial_longitude_geoc_deg: -2.3273,
         prp.initial_latitude_geod_deg: 51.3781  # corresponds to UoBath
         }
    )
    last_agent_reward = Property('reward/last_agent_reward', 'agent reward from step; includes'
                                                             'potential-based shaping reward')
    last_assessment_reward = Property('reward/last_assess_reward', 'assessment reward from step;'
                                                                   'excludes shaping')
    state_variables: Tuple[BoundedProperty, ...]
    action_variables: Tuple[BoundedProperty, ...]
    assessor: assessors.Assessor
    State: Type[NamedTuple]

    def __init__(self, assessor: assessors.Assessor, debug: bool = False) -> None:
        self.last_state = None
        self.assessor = assessor
        self._make_state_class()
        self.debug = debug

    def _make_state_class(self) -> None:
        """ Creates a namedtuple for readable State data """
        # get list of state property names, containing legal chars only
        legal_attribute_names = [prop.get_legal_name() for prop in
                                 self.state_variables]
        self.State = namedtuple('State', legal_attribute_names)

    def task_step(self, sim: Simulation, action: Sequence[float], sim_steps: int) \
            -> Tuple[NamedTuple, float, bool, Dict]:
        # input actions
        for prop, command in zip(self.action_variables, action):
            sim[prop] = command

        # run simulation
        for _ in range(sim_steps):
            sim.run()

        self._update_custom_properties(sim)
        state = self.State(*(sim[prop] for prop in self.state_variables))
        done = self._is_terminal(sim)
        reward = self.assessor.assess(state, self.last_state, done)
        if done:
            reward = self._reward_terminal_override(reward, sim)
        if self.debug:
            self._validate_state(state, done, action, reward)
        self._store_reward(reward, sim)
        self.last_state = state
        info = {'reward': reward}

        return state, reward.agent_reward(), done, info

    def _validate_state(self, state, done, action, reward):
        if any(math.isnan(el) for el in state):  # float('nan') in state doesn't work!
            msg = (f'Invalid state encountered!\n'
                   f'State: {state}\n'
                   f'Prev. State: {self.last_state}\n'
                   f'Action: {action}\n'
                   f'Terminal: {done}\n'
                   f'Reward: {reward}')
            warnings.warn(msg, RuntimeWarning)

    def _store_reward(self, reward: rewards.Reward, sim: Simulation):
        sim[self.last_agent_reward] = reward.agent_reward()
        sim[self.last_assessment_reward] = reward.assessment_reward()

    def _update_custom_properties(self, sim: Simulation) -> None:
        """ Calculates any custom properties which change every timestep. """
        pass

    @abstractmethod
    def _is_terminal(self, sim: Simulation) -> bool:
        """ Determines whether the current episode should terminate.

        :param sim: the current simulation
        :return: True if the episode should terminate else False
        """
        ...

    @abstractmethod
    def _reward_terminal_override(self, reward: rewards.Reward, sim: Simulation) -> bool:
        """
        Determines whether a custom reward is needed, e.g. because
        a terminal condition is met.
        """
        ...

    def observe_first_state(self, sim: Simulation) -> np.ndarray:
        self._new_episode_init(sim)
        self._update_custom_properties(sim)
        state = self.State(*(sim[prop] for prop in self.state_variables))
        self.last_state = state
        return state

    def _new_episode_init(self, sim: Simulation) -> None:
        """
        This method is called at the start of every episode. It is used to set
        the value of any controls or environment properties not already defined
        in the task's initial conditions.

        By default it simply starts the aircraft engines.
        """
        sim.start_engines()
        sim.raise_landing_gear()
        self._store_reward(RewardStub(1.0, 1.0), sim)

    @abstractmethod
    def get_initial_conditions(self) -> Dict[Property, float]:
        ...

    def get_state_space(self) -> gym.Space:
        state_lows = np.array([state_var.min for state_var in self.state_variables])
        state_highs = np.array([state_var.max for state_var in self.state_variables])
        return gym.spaces.Box(low=state_lows, high=state_highs, dtype='float')

    def get_action_space(self) -> gym.Space:
        action_lows = np.array([act_var.min for act_var in self.action_variables])
        action_highs = np.array([act_var.max for act_var in self.action_variables])
        return gym.spaces.Box(low=action_lows, high=action_highs, dtype='float')


class Shaping(enum.Enum):
    STANDARD = 'STANDARD'
    EXTRA = 'EXTRA'
    EXTRA_SEQUENTIAL = 'EXTRA_SEQUENTIAL'


class HeadingControlTask(FlightTask):
    """
    A task in which the agent must perform steady, level flight maintaining its
    initial heading.
    """
    THROTTLE_CMD = 0.8
    MIXTURE_CMD = 0.8
    INITIAL_HEADING_DEG = 270
    DEFAULT_EPISODE_TIME_S = 60.
    ALTITUDE_SCALING_FT = 150
    TRACK_ERROR_SCALING_DEG = 8
    ROLL_ERROR_SCALING_RAD = 0.15  # approx. 8 deg
    SIDESLIP_ERROR_SCALING_DEG = 3.
    MIN_STATE_QUALITY = 0.0  # terminate if state 'quality' is less than this
    MAX_ALTITUDE_DEVIATION_FT = 1000  # terminate if altitude error exceeds this
    target_track_deg = BoundedProperty('target/track-deg', 'desired heading [deg]',
                                       prp.heading_deg.min, prp.heading_deg.max)
    track_error_deg = BoundedProperty('error/track-error-deg',
                                      'error to desired track [deg]', -180, 180)
    altitude_error_ft = BoundedProperty('error/altitude-error-ft',
                                        'error to desired altitude [ft]',
                                        prp.altitude_sl_ft.min,
                                        prp.altitude_sl_ft.max)
    action_variables = (prp.aileron_cmd, prp.elevator_cmd, prp.rudder_cmd)

    def __init__(self, shaping_type: Shaping, step_frequency_hz: float, aircraft: Aircraft,
                 episode_time_s: float = DEFAULT_EPISODE_TIME_S, positive_rewards: bool = True):
        """
        Constructor.

        :param step_frequency_hz: the number of agent interaction steps per second
        :param aircraft: the aircraft used in the simulation
        """
        self.max_time_s = episode_time_s
        episode_steps = math.ceil(self.max_time_s * step_frequency_hz)
        self.steps_left = BoundedProperty('info/steps_left', 'steps remaining in episode', 0,
                                          episode_steps)
        self.aircraft = aircraft
        self.extra_state_variables = (self.altitude_error_ft, prp.sideslip_deg,
                                      self.track_error_deg, self.steps_left)
        self.state_variables = FlightTask.base_state_variables + self.extra_state_variables
        self.positive_rewards = positive_rewards
        assessor = self.make_assessor(shaping_type)
        super().__init__(assessor)

    def make_assessor(self, shaping: Shaping) -> assessors.AssessorImpl:
        base_components = self._make_base_reward_components()
        shaping_components = ()
        return self._select_assessor(base_components, shaping_components, shaping)

    def _make_base_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
        base_components = (
            rewards.AsymptoticErrorComponent(name='altitude_error',
                                             prop=self.altitude_error_ft,
                                             state_variables=self.state_variables,
                                             target=0.0,
                                             is_potential_based=False,
                                             scaling_factor=self.ALTITUDE_SCALING_FT),
            rewards.AsymptoticErrorComponent(name='travel_direction',
                                             prop=self.track_error_deg,
                                             state_variables=self.state_variables,
                                             target=0.0,
                                             is_potential_based=False,
                                             scaling_factor=self.TRACK_ERROR_SCALING_DEG),
            # add an airspeed error relative to cruise speed component?
        )
        return base_components

    def _select_assessor(self, base_components: Tuple[rewards.RewardComponent, ...],
                         shaping_components: Tuple[rewards.RewardComponent, ...],
                         shaping: Shaping) -> assessors.AssessorImpl:
        if shaping is Shaping.STANDARD:
            return assessors.AssessorImpl(base_components, shaping_components,
                                          positive_rewards=self.positive_rewards)
        else:
            wings_level = rewards.AsymptoticErrorComponent(name='wings_level',
                                                           prop=prp.roll_rad,
                                                           state_variables=self.state_variables,
                                                           target=0.0,
                                                           is_potential_based=True,
                                                           scaling_factor=self.ROLL_ERROR_SCALING_RAD)
            no_sideslip = rewards.AsymptoticErrorComponent(name='no_sideslip',
                                                           prop=prp.sideslip_deg,
                                                           state_variables=self.state_variables,
                                                           target=0.0,
                                                           is_potential_based=True,
                                                           scaling_factor=self.SIDESLIP_ERROR_SCALING_DEG)
            potential_based_components = (wings_level, no_sideslip)

        if shaping is Shaping.EXTRA:
            return assessors.AssessorImpl(base_components, potential_based_components,
                                          positive_rewards=self.positive_rewards)
        elif shaping is Shaping.EXTRA_SEQUENTIAL:
            altitude_error, travel_direction = base_components
            # make the wings_level shaping reward dependent on facing the correct direction
            dependency_map = {wings_level: (travel_direction,)}
            return assessors.ContinuousSequentialAssessor(base_components, potential_based_components,
                                                          potential_dependency_map=dependency_map,
                                                          positive_rewards=self.positive_rewards)

    def get_initial_conditions(self) -> Dict[Property, float]:
        extra_conditions = {prp.initial_u_fps: self.aircraft.get_cruise_speed_fps(),
                            prp.initial_v_fps: 0,
                            prp.initial_w_fps: 0,
                            prp.initial_p_radps: 0,
                            prp.initial_q_radps: 0,
                            prp.initial_r_radps: 0,
                            prp.initial_roc_fpm: 0,
                            prp.initial_heading_deg: self.INITIAL_HEADING_DEG,
                            }
        return {**self.base_initial_conditions, **extra_conditions}

    def _update_custom_properties(self, sim: Simulation) -> None:
        self._update_track_error(sim)
        self._update_altitude_error(sim)
        self._decrement_steps_left(sim)

    def _update_track_error(self, sim: Simulation):
        v_north_fps, v_east_fps = sim[prp.v_north_fps], sim[prp.v_east_fps]
        track_deg = prp.Vector2(v_east_fps, v_north_fps).heading_deg()
        target_track_deg = sim[self.target_track_deg]
        error_deg = utils.reduce_reflex_angle_deg(track_deg - target_track_deg)
        sim[self.track_error_deg] = error_deg

    def _update_altitude_error(self, sim: Simulation):
        altitude_ft = sim[prp.altitude_sl_ft]
        target_altitude_ft = self._get_target_altitude()
        error_ft = altitude_ft - target_altitude_ft
        sim[self.altitude_error_ft] = error_ft

    def _decrement_steps_left(self, sim: Simulation):
        sim[self.steps_left] -= 1

    def _is_terminal(self, sim: Simulation) -> bool:
        # terminate when time >= max, but use math.isclose() for float equality test
        terminal_step = sim[self.steps_left] <= 0
        state_quality = sim[self.last_assessment_reward]
        state_out_of_bounds = state_quality < self.MIN_STATE_QUALITY  # TODO: issues if sequential?
        return terminal_step or state_out_of_bounds or self._altitude_out_of_bounds(sim)

    def _altitude_out_of_bounds(self, sim: Simulation) -> bool:
        altitude_error_ft = sim[self.altitude_error_ft]
        return abs(altitude_error_ft) > self.MAX_ALTITUDE_DEVIATION_FT

    def _get_out_of_bounds_reward(self, sim: Simulation) -> rewards.Reward:
        """
        if aircraft is out of bounds, we give the largest possible negative reward:
        as if this timestep, and every remaining timestep in the episode was -1.
        """
        reward_scalar = (1 + sim[self.steps_left]) * -1.
        return RewardStub(reward_scalar, reward_scalar)

    def _reward_terminal_override(self, reward: rewards.Reward, sim: Simulation) -> rewards.Reward:
        if self._altitude_out_of_bounds(sim) and not self.positive_rewards:
            # if using negative rewards, need to give a big negative reward on terminal
            return self._get_out_of_bounds_reward(sim)
        else:
            return reward

    def _new_episode_init(self, sim: Simulation) -> None:
        super()._new_episode_init(sim)
        sim.set_throttle_mixture_controls(self.THROTTLE_CMD, self.MIXTURE_CMD)
        sim[self.steps_left] = self.steps_left.max
        sim[self.target_track_deg] = self._get_target_track()

    def _get_target_track(self) -> float:
        # use the same, initial heading every episode
        return self.INITIAL_HEADING_DEG

    def _get_target_altitude(self) -> float:
        return self.INITIAL_ALTITUDE_FT

    def get_props_to_output(self) -> Tuple:
        return (prp.u_fps, prp.altitude_sl_ft, self.altitude_error_ft, self.target_track_deg,
                self.track_error_deg, prp.roll_rad, prp.sideslip_deg, self.last_agent_reward,
                self.last_assessment_reward, self.steps_left)


class TurnHeadingControlTask(HeadingControlTask):
    """
    A task in which the agent must make a turn from a random initial heading,
    and fly level to a random target heading.
    """

    def get_initial_conditions(self) -> [Dict[Property, float]]:
        initial_conditions = super().get_initial_conditions()
        random_heading = random.uniform(prp.heading_deg.min, prp.heading_deg.max)
        initial_conditions[prp.initial_heading_deg] = random_heading
        return initial_conditions

    def _get_target_track(self) -> float:
        # select a random heading each episode
        return random.uniform(self.target_track_deg.min,
                              self.target_track_deg.max)
        
class NavigationTask(FlightTask):
    def __init__(self, shaping, step_frequency_hz: float, aircraft: Aircraft, target_point: Tuple[float, float], episode_time_s: float = 60):

        # Initialize target coordinates and other variables
        self.target_lat = target_point[0]
        self.target_lon = target_point[1]
        self.target_alt = 300
        self.max_time_s = 60
        episode_steps = math.ceil(self.max_time_s * step_frequency_hz)
        self.steps_left = BoundedProperty('info/steps_left', 'steps remaining in episode', 0, episode_steps)
        self.aircraft = aircraft
        self.target_point = target_point
        self.cumulative_altitude_dist = 0
        self.n_steps = 0
        
        self.state_variables = (
            prp.roll_rad,             # Roll angle in radians
            prp.pitch_rad,            # Pitch angle in radians
            prp.heading_deg,          # Heading/Yaw angle in degrees
            prp.throttle_cmd,         # Throttle command (range 0 to 1)
            prp.altitude_agl_ft,       # Altitude realative to the ground in feet
            prp.lat_geod_deg,    # Current latitude in degrees
            prp.lng_geoc_deg    # Current longitude in degrees
        )
        
        self.state_bounds = {
            prp.roll_rad: (-np.pi, np.pi),
            prp.pitch_rad: (-np.pi / 2, np.pi / 2),
            prp.heading_deg: (0, 360),
            prp.throttle_cmd: (0, 1),
            prp.altitude_agl_ft: (0, 1000),  
            prp.lat_geod_deg: (-90, 90),
            prp.lng_geoc_deg: (-180, 180)
        }
        
        self.action_variables = (
            prp.aileron_cmd,
            prp.elevator_cmd,
            prp.rudder_cmd,
            prp.throttle_cmd
        )
        
        assessor = None
        super().__init__(assessor)
        #self.reset_target_point(37.6190, -122.3750)

    def setReward(self, distance, crashed, alt_dist):
        """ Sets the reward of the individual"""
        crash_penalty = -100 if crashed else 0
        reward = 0.6*(1 / (distance + 1)) + crash_penalty - 0.4 * (alt_dist/300)
        return reward
    
    
    def task_step(self, sim: Simulation, action: Sequence[float], sim_steps: int) -> Tuple[NamedTuple, float, bool, Dict]:
        for prop, command in zip(self.action_variables, action):
            sim[prop] = command
        for _ in range(sim_steps):
            sim.run()

        
        observation = self.observe_first_state(sim)
        self.n_steps += 1
        current_altitude = sim[prp.altitude_agl_ft] * 0.3048
        alt_dist = abs(300 - current_altitude)
        crashed = current_altitude <= 100
        distance_to_target = observation[5]
        reward = self.setReward(distance_to_target, crashed, alt_dist)
        done = self._is_terminal(sim, distance_to_target, current_altitude)

        if done:
            reward = self._reward_terminal_override(reward, sim, distance_to_target, current_altitude)

        # Info dictionary
        info = {
            'distance_to_target': distance_to_target,
            'current_altitude': current_altitude
        }

        return observation, reward, done, info

    def get_initial_conditions(self) -> Dict[Property, float]:
        # Reset target point at the start of each episode
        #self.reset_target_point(start_lat=37.6190, start_lon=-122.3750, radius=250)
        initial_conditions = {
            prp.initial_altitude_ft: 1000,     # ~= 300 meters          
            prp.initial_latitude_geod_deg: 37.6190,     
            prp.initial_longitude_geoc_deg: -122.3750,  
            prp.initial_terrain_altitude_ft: 0,        
            prp.engine_running: 1,                      
            prp.throttle_cmd: 0.5,                      
            prp.initial_u_fps: 150,
        }
        return initial_conditions

    def normalize_yaw(self, yaw):
        while yaw > math.pi:
            yaw -= 2 * math.pi
        while yaw < -math.pi:
            yaw += 2 * math.pi
        return yaw

    def observe_first_state(self, sim: Simulation) -> np.ndarray:
        """
        Extracts the current observation for the episode.
        """
        
        current_roll = sim[prp.roll_rad]  # Roll in radians
        current_pitch = sim[prp.pitch_rad]  # Pitch in radians
        current_yaw = math.radians(sim[prp.heading_deg])  # Yaw converted to radians
        current_yaw = self.normalize_yaw(current_yaw)  # Normalize to [-π, π]
        throttle = sim[prp.throttle_cmd] * 100  # Throttle as percentage (normalized to 0-100)
        current_altitude = sim[prp.altitude_agl_ft] * 0.3048  # Altitude (AGL) from feet to meters
        current_altitude_msl = sim[prp.altitude_sl_ft] * 0.3048  # Altitude (MSL) from feet to meters

        # GPS Position
        current_lat = sim[prp.lat_geod_deg]  # Latitude 
        current_lon = sim[prp.lng_geoc_deg]  # Longitude 

        # Velocities
        velocity_north = sim[prp.v_north_fps] * 30.48  # Velocity North in cm/s
        velocity_east = sim[prp.v_east_fps] * 30.48  # Velocity East in cm/s
        velocity_down = sim[prp.v_down_fps] * 30.48  # Velocity Down in cm/s
        ground_speed = math.sqrt(velocity_north**2 + velocity_east**2)  # Ground speed in cm/s

        # Heading
        heading = sim[prp.heading_deg] * 100  # Heading in centi-degrees (0-36000)

        # Angular Velocities
        roll_speed = sim[prp.p_radps]  # Roll rate in rad/s
        pitch_speed = sim[prp.q_radps]  # Pitch rate in rad/s
        yaw_speed = sim[prp.r_radps]  # Yaw rate in rad/s
        
        distance = self.calculate_distance(sim[prp.lat_geod_deg], sim[prp.lng_geoc_deg], self.target_alt)
        yaw_angle_to_target = self.calculate_yaw_angle(sim[prp.lat_geod_deg], sim[prp.lng_geoc_deg], current_yaw)
        pitch_angle_to_target = self.calculate_pitch_angle(current_altitude)
        
        observation = np.array([
            current_roll,
            current_pitch,
            current_yaw,
            throttle,
            current_altitude,
            distance,
            yaw_angle_to_target,
            pitch_angle_to_target
        ], dtype=np.float32)
        
        
        """
            observation = np.array([
            current_roll,
            current_pitch,
            current_yaw,
            throttle,
            current_altitude,
            distance,
            yaw_angle_to_target,
            pitch_angle_to_target,
            
            current_altitude_msl,
            current_lat,
            current_lon,
            velocity_north,
            velocity_east,
            velocity_down,
            ground_speed,
            heading,
            roll_speed,
            pitch_speed,
            yaw_speed,
            
        ], dtype=np.float32)
        """
        
        return observation

    def calculate_distance(self, lat1: float, lon1: float, alt1: float) -> float:
        lat2, lon2, alt2 = self.target_lat, self.target_lon, self.target_alt
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        horizontal_distance = 6371000 * c

        vertical_distance = abs(alt2 - alt1)
        return math.sqrt(horizontal_distance ** 2 + vertical_distance ** 2)

    def calculate_yaw_angle(self, lat1: float, lon1: float, heading: float) -> float:
        lat2, lon2 = self.target_lat, self.target_lon
        
        delta_lon = math.radians(lon2 - lon1)
        lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)

        #Bearing from the aircraft to the target
        x = math.sin(delta_lon) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
        
        bearing = math.atan2(x, y)
        bearing_deg = math.degrees(bearing)

        if bearing_deg < 0:
            bearing_deg += 360
        
        #Yaw angle = difference between the aircraft's heading and the bearing
        yaw_angle = bearing_deg - heading
        
        #-180° and 180°
        if yaw_angle > 180:
            yaw_angle -= 360
        elif yaw_angle < -180:
            yaw_angle += 360
        
        return math.radians(yaw_angle)

    def calculate_pitch_angle(self, alt1: float) -> float:
        alt2 = self.target_alt
        distance_horizontal = self.calculate_distance(self.target_lat, self.target_lon, alt1)
        return math.atan2(alt2 - alt1, distance_horizontal)

    def calculate_circle_point(self, lat, lon, radius, angle):
        EARTH_RADIUS = 6371000
        lat_rad, lon_rad, angle_rad = map(math.radians, (lat, lon, angle))

        new_lat_rad = math.asin(math.sin(lat_rad) * math.cos(radius / EARTH_RADIUS) +
                                math.cos(lat_rad) * math.sin(radius / EARTH_RADIUS) * math.cos(angle_rad))
        new_lon_rad = lon_rad + math.atan2(math.sin(angle_rad) * math.sin(radius / EARTH_RADIUS) * math.cos(lat_rad),
                                           math.cos(radius / EARTH_RADIUS) - math.sin(lat_rad) * math.sin(new_lat_rad))
        return math.degrees(new_lat_rad), math.degrees(new_lon_rad)
    
    def reset_target_point(self, start_lat: float, start_lon: float, radius: float = 250):
        # Generate points around the circle at the given radius
        circle_points = [self.calculate_circle_point(start_lat, start_lon, radius, i * (360 / 15)) for i in range(15)]
        self.target_point = random.choice(circle_points)
        self.target_lat, self.target_lon = self.target_point[0], self.target_point[1]

    def _is_terminal(self, sim: Simulation, distance_to_target: float, current_altitude: float) -> bool:
        """Determines if the episode should end based on distance to target or altitude."""

        if distance_to_target < 5.0 or current_altitude < 100.0:
            return True
        
        #print(f'Current Altitude: {current_altitude} ||| Distance to Target: {distance_to_target}')
        
        return False

    def _reward_terminal_override(self, reward: float, sim: Simulation, distance_to_target: float, current_altitude: float) -> float:
        """Overrides the reward if terminal conditions are met, adding a bonus for reaching target or penalty for crashing."""
        if distance_to_target < 5.0:
            reward += 100  
        elif current_altitude < 3.0:
            reward -= 100  
        return reward
    
    def get_props_to_output(self) -> Tuple:
        
        return (
            prp.u_fps,                 # Forward velocity (x-axis) in feet per second
            prp.altitude_sl_ft,        # Altitude above sea level in feet
            prp.roll_rad,              # Roll angle in radians
            prp.pitch_rad,             # Pitch angle in radians
            prp.heading_deg,           # Heading/Yaw angle in degrees
            prp.sideslip_deg,          # Sideslip angle in degrees
            prp.throttle_cmd,          # Throttle command (0 to 1 normalized)
        )

    def get_state_space(self):
        lows = np.array([
            -np.pi,   # Roll
            -np.pi/2, # Pitch
            -np.pi,   # Yaw (normalized to -π to π)
            0.0,      # Throttle (0 to 1)
            0.0,      # Altitude (meters)
            0.0,      # Distance (meters)
            -180,     # Yaw angle (degrees)
            -90       # Pitch angle (degrees)
        ], dtype=np.float32)

        highs = np.array([
            np.pi,    # Roll
            np.pi/2,  # Pitch
            np.pi,    # Yaw
            1.0,      # Throttle
            900,     # Altitude 
            1000,     # Distance
            np.pi,      # Yaw angle
            np.pi/2        # Pitch angle
        ], dtype=np.float32)
        return spaces.Box(low=lows, high=highs, dtype=np.float32)

"""
    def get_state_space(self):
        lows = np.array([self.state_bounds[var][0] for var in self.state_variables], dtype=np.float32)
        highs = np.array([self.state_bounds[var][1] for var in self.state_variables], dtype=np.float32)
        return spaces.Box(low=lows, high=highs, dtype=np.float32)
    
"""