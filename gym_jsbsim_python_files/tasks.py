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
        self.initial_lat = 37.6190
        self.initial_lon = -122.3750
        self.target_alt = 300
        self.max_time_s = 240
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
            prp.altitude_agl_ft: (0, 3000),  
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

    def setReward(self, crashed, altitude, heading_to_target, forward_vel, roll, pitch):
        """ Sets the reward"""

        u_vel_penalty = 0
        if forward_vel < 100:
            u_vel_penalty = - 100 / (forward_vel +1)    

        
        pitch_penalty = 0
        if abs(pitch) > math.pi / 4: #0.785 -> 45
            pitch_penalty = -20

        roll_penalty = 0
        if abs(roll) > math.pi / 5: # min = 0.628 max = 1.57
            roll_penalty = -20 * abs(roll) # 
            
        altitude_penalty = 0
        if altitude < 250:
            altitude_deviation = abs(300 - altitude) # max = 200 min = 50
            altitude_penalty = - (altitude_deviation/50) * 20 # varia entre -20 e -80

        heading_reward = 15.7 # = 3.14 * 5
        if abs(heading_to_target) > 0.0873: # 5 e 180 graus
            heading_pen = abs(heading_to_target) * 5 # Varia entre 0.44 e 15.7
            heading_reward -= heading_pen # Valor final varia entre 0 e 15.26

        elif abs(heading_to_target) > 0.0174533:
            heading_reward += 10

        crash_penalty = -1000 if crashed else 0
        #target_reward = (1 / (distance + 1)) * 1000
        
        #reward = 0.7 * target_reward + 0.3 * altitude_penalty + crash_penalty + heading_reward
        reward = heading_reward + crash_penalty + altitude_penalty + u_vel_penalty + roll_penalty + pitch_penalty
        #print(f'Reward: {reward}')
        #print(f"Crash: {crashed}, Altitude Deviation: {altitude_deviation}, Heading Target: {heading_to_target}, U_Vel: {forward_vel}")
        #print(f"U_Vel: {u_vel_penalty}, Altitude_pen: {altitude_penalty}, Heading Reward: {heading_reward}, Total: {reward}")
        return reward
    
    def task_step(self, sim: Simulation, action: Sequence[float], sim_steps: int) -> Tuple[NamedTuple, float, bool, Dict]:
        #print(action)
        action[3] = max(0.5, action[3])
        for prop, command in zip(self.action_variables, action):
            sim[prop] = command
        for _ in range(sim_steps):
            sim.run()

        
        observation = self.observe_first_state(sim)
        self.n_steps += 1
        current_altitude = sim[prp.altitude_agl_ft] * 0.3048
        #altitude_deviation = abs(300 - current_altitude)
        crashed = current_altitude <= 100
        unnormalized_observations = self.unnormalize_observation(observation)
        #distance_to_target = unnormalized_observations[5]
        distance_to_target = self.calculate_distance(sim[prp.lat_geod_deg], sim[prp.lng_geoc_deg])
        heading_to_target = unnormalized_observations[4]
        #print(f"Heading to target: {heading_to_target}")
        engine_bool = sim[prp.engine_running]
        #print(f"Engine: {engine_bool}")
        reward = self.setReward(crashed, current_altitude, heading_to_target, unnormalized_observations[6], unnormalized_observations[0], unnormalized_observations[1])
        #print(f"Reward: {reward}")
        
        if distance_to_target < 20.0:
            reward = 1000
            self.reset_target_point(sim[prp.lat_geod_deg], sim[prp.lng_geoc_deg])
            #print("Reached the Target Point")
        
        done = self._is_terminal(sim, distance_to_target, current_altitude, observation)

        # Info dictionary
        info = {
            'distance_to_target': distance_to_target,
            'current_altitude': current_altitude
        }

        return observation, reward, done, info

    def get_initial_conditions(self) -> Dict[Property, float]:

        initial_conditions = {
            prp.initial_altitude_ft: 1000,     # ~= 300 meters          
            prp.initial_latitude_geod_deg: 37.6190,     
            prp.initial_longitude_geoc_deg: -122.3750,  
            prp.initial_terrain_altitude_ft: 0,   

            # Engine Start Configuration
            prp.mixture_cmd: 1,   # Full rich mixture
            prp.throttle_cmd: 1,       
            prp.engine_ignition: 1,  # Ignition ON
            prp.engine_magnetos: 3,  # Set magnetos to BOTH (1 = Left, 2 = Right, 3 = Both)
            prp.propeller_rpm: 2200, # Propeller running at cruise RPM
            prp.engine_thrust_lbs: 250,  # Initial thrust
            prp.engine_cutoff: 0,
            prp.engine_starter: 1,
            prp.engine_running: 1,  # Engine ON

   
            prp.flaps: 0,  # Flaps retracted
            prp.gear: 0,  # Gear retracted

            # Electrical System
            prp.battery_on: 1,  # Battery ON
            prp.alternator_on: 1,  # Alternator ON

            # Fuel Levels
            prp.fuel_tank_left: 100,  # Fuel left tank
            prp.fuel_tank_right: 100,  # Fuel right tank

            prp.initial_u_fps: 150,
            prp.initial_p_radps: 0,
            prp.initial_q_radps: 0,
            prp.initial_r_radps: 0,

            prp.initial_sun_azimuth_deg: 180,
            prp.initial_sun_elevation_deg: 60,
            prp.initial_sun_intensity: 1.0,
            prp.initial_time_of_day: 12,
        }
        return initial_conditions

    def observe_first_state(self, sim: Simulation) -> np.ndarray:
        """
        Extracts the current observation for the episode.
        """
        
        current_roll = sim[prp.roll_rad]  # Roll in radians
        current_pitch = sim[prp.pitch_rad]  # Pitch in radians
        current_yaw = math.radians(sim[prp.heading_deg])  # Yaw converted to radians
        current_yaw = self.normalize_yaw(current_yaw)  # Normalized to [-π, π]
        
        throttle = sim[prp.throttle_cmd] # [0, 1]
        current_altitude = sim[prp.altitude_agl_ft] * 0.3048  # Altitude (AGL) from feet to meters
        altitude_deviation = current_altitude - self.target_alt
         
        distance = self.calculate_distance(sim[prp.lat_geod_deg], sim[prp.lng_geoc_deg])
        yaw_angle_to_target = self.calculate_yaw_angle(sim[prp.lat_geod_deg], sim[prp.lng_geoc_deg], current_yaw)
        pitch_angle_to_target = self.calculate_pitch_angle(sim[prp.lat_geod_deg], sim[prp.lng_geoc_deg], current_altitude)
        
        u_vel = sim[prp.u_fps]
        altitude_rate = sim[prp.altitude_rate_fps] 
        p_rad = sim[prp.p_radps]
        q_rad = sim[prp.q_radps]
        r_rad = sim[prp.r_radps]
        
        observation = np.array([
            current_roll,
            current_pitch,
            current_yaw,
            #throttle,
            #current_altitude,
            altitude_deviation, 
            #distance,
            yaw_angle_to_target,
            pitch_angle_to_target,
            u_vel,
            altitude_rate,
            p_rad,
            q_rad,
            r_rad,
        ], dtype=np.float32)
        
        for i, (low, high) in enumerate(zip(self.get_state_space().low, self.get_state_space().high)):
            if observation[i] is None:
                print(f"Observation {i}: None value detected!")
            elif np.isnan(observation[i]):
                print(f"Observation {i}: NaN detected!")
            elif np.isinf(observation[i]):
                print(f"Observation {i}: Inf detected!")
            elif not (low <= observation[i] <= high):
                print(f"Observation {i}: {observation[i]} is out of range! Expected: [{low}, {high}]")

        assert not any(obs is None or np.isnan(obs) or np.isinf(obs) for obs in observation), "Observation contains invalid values (None, NaN, or Inf)!"
        assert self.get_state_space().contains(observation), f"Observation out of bounds: {observation}"
        
        #Normalizing using min-max scaler
        observation = self.normalize_observation(observation)
        
        return observation

    def _is_terminal(self, sim: Simulation, distance_to_target: float, current_altitude: float, observation: list) -> bool:
        """Determines if the episode should end based on distance to target or altitude."""
                        
        if current_altitude <= 100.0 or current_altitude >= 599.0 or distance_to_target >= 9950:
            return True
                
        return False

    def _reward_terminal_override(self, reward: float, sim: Simulation, distance_to_target: float, current_altitude: float) -> float:
        """Overrides the reward if terminal conditions are met, adding a bonus for reaching target or penalty for crashing or going over the altitude limit."""
        if distance_to_target < 20.0:
            reward += 100  
        elif current_altitude < 100:
            reward -= 100  
        elif 290 < current_altitude < 310:
             reward += 50
        

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
            #0.0,      # Throttle (0 to 1)
            -205,      # Altitude Deviation (meters)
            #0.0,      # Distance (meters)
            -np.pi,     # Yaw angle (degrees)
            -np.pi/2,       # Pitch angle (degrees)
            -2200,
            -250,
            -2 * math.pi, 
            -2 * math.pi,
            -2 * math.pi,
        ], dtype=np.float32)

        highs = np.array([
            np.pi,    # Roll
            np.pi/2,  # Pitch
            np.pi,    # Yaw
            #1.0,      # Throttle
            305,     # Altitude Deviation (meters)
            #10000,     # Distance
            np.pi,      # Yaw angle
            np.pi/2,        # Pitch angle
            2200,
            250,
            2 * math.pi,
            2 * math.pi,
            2 * math.pi,
        ], dtype=np.float32)
        return spaces.Box(low=lows, high=highs, dtype=np.float32)


    """ Auxiliar Functions"""
    
    def normalize_yaw(self, yaw):
        while yaw > math.pi:
            yaw -= 2 * math.pi
        while yaw < -math.pi:
            yaw += 2 * math.pi
        return yaw

    def normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Normalizes each feature of the observation to be within [-1,1].
        """
        lows = self.get_state_space().low
        highs = self.get_state_space().high

        # Min-max normalization formula: x_norm = (x - min) / (max - min)
        normalized_obs = 2 * (observation - lows) / (highs - lows) - 1

        return normalized_obs
    
    def unnormalize_observation(self, normalized_obs: np.ndarray) -> np.ndarray:
        """
        Reverts the normalization of an observation from [-1, 1] back to the original scale.
        """
        lows = self.get_state_space().low
        highs = self.get_state_space().high

        # Reverse min-max scaling: x = 0.5 * ((x_norm + 1) * (max - min)) + min
        original_obs = 0.5 * ((normalized_obs + 1) * (highs - lows)) + lows

        return original_obs
    
    def calculate_distance(self, lat1: float, lon1: float) -> float:
        lat2, lon2 = self.target_lat, self.target_lon
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        horizontal_distance = 6371000 * c  # Earth's radius in meters

        return horizontal_distance

    def calculate_yaw_angle(self, lat1: float, lon1: float, heading: float) -> float:
        lat2, lon2 = self.target_lat, self.target_lon
                
        delta_lon = math.radians(lon2 - lon1)
        lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)

        # Bearing from the aircraft to the target
        x = math.sin(delta_lon) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
        
        bearing = math.atan2(x, y)
        yaw_angle = bearing - heading

        # Ensuring yaw is in the range -π to π
        if yaw_angle > math.pi:
            yaw_angle -= 2 * math.pi
        elif yaw_angle < -math.pi:
            yaw_angle += 2 * math.pi

        return yaw_angle

    def calculate_pitch_angle(self, lat1: float, lon1: float, alt1: float) -> float:
        alt2 = self.target_alt
        distance_horizontal = self.calculate_distance(lat1, lon1)
        return math.atan2(alt2 - alt1, distance_horizontal)


    """Target Point Functions"""

    def calculate_circle_point(self, lat, lon, radius, angle):
        """
        This function calculates a point on the surface of the Earth given a 
        center latitude/longitude, a radius in meters, and an angle in degrees.
        """
        EARTH_RADIUS = 6371000
        
        lat_rad, lon_rad, angle_rad = map(math.radians, (lat, lon, angle))

        new_lat_rad = math.asin(math.sin(lat_rad) * math.cos(radius / EARTH_RADIUS) +
                                math.cos(lat_rad) * math.sin(radius / EARTH_RADIUS) * math.cos(angle_rad))
        new_lon_rad = lon_rad + math.atan2(math.sin(angle_rad) * math.sin(radius / EARTH_RADIUS) * math.cos(lat_rad),
                                           math.cos(radius / EARTH_RADIUS) - math.sin(lat_rad) * math.sin(new_lat_rad))
        
        return math.degrees(new_lat_rad), math.degrees(new_lon_rad)

    def generate_equally_spaced_target_points(self, n):
        """
        Generates `n` equally spaced angles around a circle (in degrees).
        """
        random_offset = np.random.uniform(0, 360) 
        return [(random_offset + i * (360 / n)) % 360 for i in range(n)] 

    def create_target_points(self, start_lat, start_lon, radius, n=30):
        """
        Creates `n` equally spaced target points around a circle centered at the
        provided (start_lat, start_lon), using a given radius in meters.
        """
        angles = self.generate_equally_spaced_target_points(n)
        return [self.calculate_circle_point(start_lat, start_lon, radius, angle) for angle in angles]
    
    def reset_target_point(self, initial_lat, initial_lon):
        """
        Randomly selects a target point from a generated circle of points.
        """
        CIRCLE_RADIUS = random.randint(4000, 4500)
        target_points = self.create_target_points(initial_lat, initial_lon, CIRCLE_RADIUS, 10)
        self.target_point = random.choice(target_points)
        self.target_lat, self.target_lon = self.target_point
