"""
Drayage Trailer Dispatching Environment (Gymnasium-compatible)

This environment models the trailer repositioning problem in drayage operations
as described in the Dray-Q methodology. It handles:
- Empty trailer repositioning decisions
- Automatic load dispatch when trailers are available
- Time-window based delay penalties
- Multi-yard deficit tracking with n-step lookahead

Key entities:
- Trailers: Can be empty, loading, loaded, on_the_way, unloading, empty_moving
- Orders: Jobs with pickup/delivery yards, time windows, and durations
- Yards: Locations with capacity, minimum empty levels, and importance weights
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import copy


class TrailerStatus(Enum):
    EMPTY = "empty"
    LOADING = "loading"
    LOADED = "loaded"
    ON_THE_WAY = "on_the_way"
    UNLOADING = "unloading"
    EMPTY_MOVING = "empty_moving"


class OrderStatus(Enum):
    TODO = "toDo"
    DELAYED = "delayed"
    LOADING = "loading"
    LOADED = "loaded"
    ON_THE_WAY = "on_the_way"
    UNLOADING = "unloading"
    DELIVERED = "delivered"


@dataclass
class Order:
    """Domain model for a drayage order"""
    order_id: int
    pickup_yard_id: int
    delivery_yard_id: int
    pickup_time_window: Tuple[int, int]  # (earliest, latest) in time steps
    delivery_time_window: Tuple[int, int]
    pickup_duration: int  # minutes
    delivery_duration: int  # minutes
    base_cost: float
    status: OrderStatus = OrderStatus.TODO
    assigned_trailer_id: Optional[int] = None
    packing_start_time: int = 0  # time step when loading can begin
    real_pickup_time: int = 0  # actual pickup time step (after delays)
    real_delivery_time: int = 0  # actual delivery time step
    arrive_time: int = 999999
    accumulated_delay_cost: float = 0.0


@dataclass
class Trailer:
    """Domain model for a trailer"""
    trailer_id: int
    current_yard_id: int
    status: TrailerStatus = TrailerStatus.EMPTY
    assigned_order_id: Optional[int] = None
    status_end_time: int = 999999  # when current status terminates


@dataclass
class Yard:
    """Domain model for a yard"""
    yard_id: int
    min_empty_level: int
    importance: float
    empty_attach_time: int  # minutes to attach/detach empty trailer


class DrayageEnv(gym.Env):
    """
    Gymnasium environment for trailer repositioning in drayage operations.
    
    State: [trailer_status (binary) | n-step deficit per yard]
    Action: (trailer_id, yard_id) flattened to single discrete index
    Reward: Multi-objective function balancing deficit levels and delays
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        num_trailers: int,
        num_yards: int,
        orders_config: List[Dict],
        travel_time_matrix: np.ndarray,
        yard_min_empty: List[int],
        yard_importance: List[float],
        yard_attach_times: List[int],
        step_minutes: int = 15,
        min_hour: int = 6,
        max_hour: int = 24,
        n_step_deficit: int = 4,
        delay_cost_per_step: float = 70.0,
        forbidden_action_penalty: float = 30.0,
        laplace_scale: float = 1.0,
        terminate_on_forbidden: bool = True,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        # Configuration
        self.num_trailers = num_trailers
        self.num_yards = num_yards
        self.step_minutes = step_minutes
        self.min_hour = min_hour
        self.max_hour = max_hour
        self.n_step_deficit = n_step_deficit
        self.delay_cost_per_step = delay_cost_per_step
        self.forbidden_penalty = forbidden_action_penalty
        self.laplace_scale = laplace_scale
        self.terminate_on_forbidden = terminate_on_forbidden
        
        # Time management
        self.time_steps = self._build_time_steps()
        self.current_step = 0
        
        # Static data
        self.travel_matrix = travel_time_matrix
        self.yard_configs = [
            Yard(i, yard_min_empty[i], yard_importance[i], yard_attach_times[i])
            for i in range(num_yards)
        ]
        self.yard_importance = np.array(yard_importance, dtype=np.float32)
        self.orders_config = orders_config
        
        # Dynamic entities (initialized in reset)
        self.trailers: Dict[int, Trailer] = {}
        self.orders: Dict[int, Order] = {}
        self.yards: Dict[int, Yard] = {}
        
        # Gym spaces
        # Action: discrete index for (trailer, yard) pairs
        self.action_space = spaces.Discrete(num_trailers * num_yards)
        
        # State: [trailer_binary_status | yard_n_step_deficits]
        state_dim = num_trailers + num_yards
        self.observation_space = spaces.Box(
            low=-100, high=100, shape=(state_dim,), dtype=np.int32
        )
        
        self._np_random = None
        if seed is not None:
            self.seed(seed)
    
    def _build_time_steps(self) -> List[int]:
        """Generate time step indices from min_hour to max_hour"""
        minutes_per_day = (self.max_hour - self.min_hour) * 60
        num_steps = minutes_per_day // self.step_minutes
        return list(range(num_steps))
    
    def _time_to_step(self, hour: float) -> int:
        """Convert hour (e.g., 8.5 for 8:30am) to time step index"""
        minutes_from_start = (hour - self.min_hour) * 60
        return int(minutes_from_start / self.step_minutes)
    
    def _minutes_to_steps(self, minutes: int) -> int:
        """Convert duration in minutes to number of time steps"""
        return max(1, int(np.ceil(minutes / self.step_minutes)))
    
    def seed(self, seed: int):
        """Set random seed for reproducibility"""
        self._np_random = np.random.RandomState(seed)
        return [seed]
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        if seed is not None:
            self.seed(seed)
        
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Initialize trailers with random yard assignments
        self.trailers = {}
        for i in range(self.num_trailers):
            yard_id = self._np_random.randint(0, self.num_yards)
            self.trailers[i] = Trailer(trailer_id=i, current_yard_id=yard_id)
        
        # Initialize orders with random perturbations to time windows
        self.orders = {}
        for i, order_cfg in enumerate(self.orders_config):
            # Add random hour offset [-2, +2] to make episodes diverse
            hour_offset = self._np_random.randint(-2, 3)
            
            pickup_yard = self._np_random.randint(0, self.num_yards)
            delivery_yard = self._np_random.randint(0, self.num_yards)
            while delivery_yard == pickup_yard:
                delivery_yard = self._np_random.randint(0, self.num_yards)
            
            pickup_earliest = max(self.min_hour + 1, order_cfg['pickup_tw'][0] + hour_offset)
            pickup_latest = min(self.max_hour, order_cfg['pickup_tw'][1] + hour_offset)
            
            delivery_earliest = max(self.min_hour + 1, order_cfg['delivery_tw'][0] + hour_offset)
            delivery_latest = min(self.max_hour, order_cfg['delivery_tw'][1] + hour_offset)
            
            pickup_tw = (self._time_to_step(pickup_earliest), self._time_to_step(pickup_latest))
            delivery_tw = (self._time_to_step(delivery_earliest), self._time_to_step(delivery_latest))
            
            pickup_duration_steps = self._minutes_to_steps(order_cfg['pickup_duration'])
            packing_start = max(0, pickup_tw[0] - pickup_duration_steps)
            
            order = Order(
                order_id=i,
                pickup_yard_id=pickup_yard,
                delivery_yard_id=delivery_yard,
                pickup_time_window=pickup_tw,
                delivery_time_window=delivery_tw,
                pickup_duration=order_cfg['pickup_duration'],
                delivery_duration=order_cfg['delivery_duration'],
                base_cost=order_cfg.get('cost', 0.0),
                packing_start_time=packing_start,
                real_pickup_time=pickup_tw[0],
                real_delivery_time=delivery_tw[0]
            )
            self.orders[i] = order
        
        # Initialize yard tracking
        self.yards = {cfg.yard_id: copy.deepcopy(cfg) for cfg in self.yard_configs}
        
        state = self._get_state()
        info = self._get_info()
        
        return state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step"""
        # Decode action
        trailer_id, target_yard_id = self._decode_action(action)
        
        # Check if action is forbidden
        forbidden = self._is_action_forbidden(trailer_id)
        
        # Apply empty trailer repositioning if valid
        if not forbidden:
            self._dispatch_empty_trailer(trailer_id, target_yard_id)
        
        # Process all orders (loading, dispatching, delivering)
        self._process_orders()
        
        # Update trailer statuses (complete movements)
        self._update_trailer_statuses()
        
        # Calculate reward
        reward = self._calculate_reward(forbidden)
        
        # Advance time
        self.current_step += 1
        
        # Check termination
        terminated = self.current_step >= len(self.time_steps) or (forbidden and self.terminate_on_forbidden)
        truncated = False
        
        state = self._get_state()
        info = self._get_info()
        
        return state, reward, terminated, truncated, info
    
    def _decode_action(self, action: int) -> Tuple[int, int]:
        """Convert flat action index to (trailer_id, yard_id)"""
        trailer_id = action // self.num_yards
        yard_id = action % self.num_yards
        return trailer_id, yard_id
    
    def _is_action_forbidden(self, trailer_id: int) -> bool:
        """Check if trailer is available for repositioning"""
        trailer = self.trailers.get(trailer_id)
        if trailer is None:
            return True
        return trailer.status != TrailerStatus.EMPTY
    
    def _dispatch_empty_trailer(self, trailer_id: int, target_yard_id: int):
        """Reposition an empty trailer to target yard"""
        trailer = self.trailers[trailer_id]
        
        if trailer.current_yard_id == target_yard_id:
            return  # Already at target
        
        # Calculate travel time
        origin = trailer.current_yard_id
        attach_time = self.yards[origin].empty_attach_time
        detach_time = self.yards[target_yard_id].empty_attach_time
        travel_minutes = self.travel_matrix[origin, target_yard_id]
        total_minutes = attach_time + detach_time + travel_minutes
        
        duration_steps = self._minutes_to_steps(total_minutes)
        arrival_step = min(self.current_step + duration_steps, len(self.time_steps) - 1)
        
        # Update trailer
        trailer.status = TrailerStatus.EMPTY_MOVING
        trailer.current_yard_id = target_yard_id
        trailer.status_end_time = arrival_step
    
    def _process_orders(self):
        """Main order processing loop: loading, dispatch, delivery"""
        # Sort orders by accumulated delay cost (highest priority first)
        sorted_orders = sorted(
            self.orders.values(),
            key=lambda o: o.accumulated_delay_cost,
            reverse=True
        )
        
        for order in sorted_orders:
            if order.status in [OrderStatus.TODO, OrderStatus.DELAYED]:
                self._try_start_loading(order)
            
            if order.status == OrderStatus.LOADING:
                self._check_loading_complete(order)
            
            if order.status == OrderStatus.LOADED:
                self._dispatch_loaded_trailer(order)
            
            if order.status == OrderStatus.ON_THE_WAY:
                self._check_arrival(order)
            
            if order.status == OrderStatus.UNLOADING:
                self._check_unloading_complete(order)
    
    def _try_start_loading(self, order: Order):
        """Attempt to assign an empty trailer and start loading"""
        if self.current_step < order.packing_start_time:
            return
        
        # Find available empty trailer at pickup yard
        pickup_yard = order.pickup_yard_id
        available_trailer = None
        
        for trailer in self.trailers.values():
            if (trailer.current_yard_id == pickup_yard and
                trailer.status == TrailerStatus.EMPTY):
                available_trailer = trailer
                break
        
        if available_trailer is None:
            # Apply delay
            self._apply_delay(order)
            return
        
        # Assign trailer and start loading
        duration_steps = self._minutes_to_steps(order.pickup_duration)
        end_step = min(self.current_step + duration_steps, len(self.time_steps) - 1)
        
        available_trailer.status = TrailerStatus.LOADING
        available_trailer.assigned_order_id = order.order_id
        available_trailer.status_end_time = end_step
        
        order.status = OrderStatus.LOADING
        order.assigned_trailer_id = available_trailer.trailer_id
        order.arrive_time = order.real_delivery_time
    
    def _apply_delay(self, order: Order):
        """Apply delay penalty when no trailer is available"""
        order.status = OrderStatus.DELAYED
        order.packing_start_time += 1
        order.real_pickup_time += 1
        order.real_delivery_time += 1
        
        # Calculate time-window based delay using Equation 4
        pickup_duration_steps = self._minutes_to_steps(order.pickup_duration)
        travel_steps = self._minutes_to_steps(
            self.travel_matrix[order.pickup_yard_id, order.delivery_yard_id]
        )
        first_arrival = self.current_step + pickup_duration_steps + travel_steps
        last_delivery = order.delivery_time_window[1]
        
        delay = max(0, first_arrival - last_delivery)
        
        # Accumulate cost if within critical window (8 steps = 2 hours)
        if delay > -8:
            order.accumulated_delay_cost += self.delay_cost_per_step
    
    def _check_loading_complete(self, order: Order):
        """Transition from loading to loaded"""
        if self.current_step >= order.real_pickup_time:
            order.status = OrderStatus.LOADED
            trailer = self.trailers[order.assigned_trailer_id]
            trailer.status = TrailerStatus.LOADED
    
    def _dispatch_loaded_trailer(self, order: Order):
        """Send loaded trailer to delivery yard"""
        trailer = self.trailers[order.assigned_trailer_id]
        
        travel_minutes = self.travel_matrix[order.pickup_yard_id, order.delivery_yard_id]
        travel_steps = self._minutes_to_steps(travel_minutes)
        arrival = min(self.current_step + travel_steps, len(self.time_steps) - 1)
        
        # Use max of calculated arrival and earliest delivery window
        arrival = max(arrival, order.delivery_time_window[0])
        
        trailer.status = TrailerStatus.ON_THE_WAY
        trailer.status_end_time = arrival
        trailer.current_yard_id = order.delivery_yard_id
        
        order.status = OrderStatus.ON_THE_WAY
        order.arrive_time = arrival
    
    def _check_arrival(self, order: Order):
        """Check if loaded trailer has arrived and start unloading"""
        trailer = self.trailers[order.assigned_trailer_id]
        
        if self.current_step >= trailer.status_end_time:
            duration_steps = self._minutes_to_steps(order.delivery_duration)
            end_step = min(self.current_step + duration_steps, len(self.time_steps) - 1)
            
            trailer.status = TrailerStatus.UNLOADING
            trailer.status_end_time = end_step
            
            order.status = OrderStatus.UNLOADING
    
    def _check_unloading_complete(self, order: Order):
        """Complete delivery and free trailer"""
        trailer = self.trailers[order.assigned_trailer_id]
        
        if self.current_step >= trailer.status_end_time - 1:
            order.status = OrderStatus.DELIVERED
            
            trailer.status = TrailerStatus.EMPTY
            trailer.assigned_order_id = None
            trailer.status_end_time = len(self.time_steps)
    
    def _update_trailer_statuses(self):
        """Complete empty trailer movements"""
        for trailer in self.trailers.values():
            if (trailer.status == TrailerStatus.EMPTY_MOVING and
                self.current_step >= trailer.status_end_time - 1):
                trailer.status = TrailerStatus.EMPTY
                trailer.status_end_time = len(self.time_steps)
    
    def _calculate_n_step_deficit(self, yard_id: int) -> int:
        """
        Calculate deficit for a yard over next n steps (Equation 2).
        Deficit = (Incoming - Outgoing) + CurrentCapacity - MinLevel
        """
        lookahead = min(self.n_step_deficit, len(self.time_steps) - self.current_step)
        
        incoming = 0
        outgoing = 0
        
        for order in self.orders.values():
            # Count deliveries to this yard
            if (order.delivery_yard_id == yard_id and
                self.current_step <= order.arrive_time <= self.current_step + lookahead):
                incoming += 1
            
            # Count pickups from this yard
            if (order.pickup_yard_id == yard_id and
                self.current_step <= order.real_pickup_time <= self.current_step + lookahead and
                order.status in [OrderStatus.TODO, OrderStatus.DELAYED]):
                outgoing += 1
        
        # Current capacity (trailers at yard)
        capacity = sum(1 for t in self.trailers.values() if t.current_yard_id == yard_id)
        
        min_level = self.yards[yard_id].min_empty_level
        
        return (incoming - outgoing) + capacity - min_level
    
    def _laplace_function(self, x: np.ndarray) -> np.ndarray:
        """Laplace distribution for deficit evaluation (Equation 6)"""
        return (1.0 / (2.0 * self.laplace_scale)) * np.exp(-np.abs(x) / self.laplace_scale)
    
    def _calculate_reward(self, forbidden: bool) -> float:
        """
        Multi-objective reward function (Equation 3).
        R = w_a * Σ L(D_i) * Im_i - w_b * Σ Dl_i - w_c * FP
        """
        # Part 1: Deficit-based reward with importance weighting
        deficits = np.array([self._calculate_n_step_deficit(i) for i in range(self.num_yards)])
        laplace_values = self._laplace_function(deficits)
        deficit_reward = np.sum(laplace_values * self.yard_importance) * 100.0
        
        # Part 2: Accumulated delay costs
        total_delay_cost = sum(order.accumulated_delay_cost for order in self.orders.values())
        
        # Part 3: Forbidden action penalty
        forbidden_cost = self.forbidden_penalty if forbidden else 0.0
        
        reward = deficit_reward - total_delay_cost - forbidden_cost
        
        return float(reward)
    
    def _get_state(self) -> np.ndarray:
        """
        Construct state vector (Equation 7).
        State = [trailer_status | n_step_deficits]
        """
        state = np.zeros(self.observation_space.shape[0], dtype=np.int32)
        
        # Trailer status (1 if empty, 0 otherwise)
        for i, trailer in enumerate(self.trailers.values()):
            state[i] = 1 if trailer.status == TrailerStatus.EMPTY else 0
        
        # N-step deficit for each yard
        for i in range(self.num_yards):
            deficit = self._calculate_n_step_deficit(i)
            state[self.num_trailers + i] = deficit
        
        return state
    
    def _get_info(self) -> Dict[str, Any]:
        """Return diagnostic information"""
        return {
            'current_step': self.current_step,
            'total_delay_cost': sum(o.accumulated_delay_cost for o in self.orders.values()),
            'orders_delivered': sum(1 for o in self.orders.values() if o.status == OrderStatus.DELIVERED),
            'orders_delayed': sum(1 for o in self.orders.values() if o.status == OrderStatus.DELAYED),
        }
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: 'human' for visual display, 'rgb_array' for image array
            
        Returns:
            RGB array if mode='rgb_array', None otherwise
        """
        if mode == 'human':
            if not hasattr(self, '_renderer') or self._renderer is None:
                from drayage_renderer import DrayageRenderer
                self._renderer = DrayageRenderer(self)
            return self._renderer.render(mode='human')
        elif mode == 'rgb_array':
            if not hasattr(self, '_renderer') or self._renderer is None:
                from drayage_renderer import DrayageRenderer
                self._renderer = DrayageRenderer(self)
            return self._renderer.render(mode='rgb_array')
        else:
            print(f"\n=== Time Step {self.current_step}/{len(self.time_steps)} ===")
            for yard_id in range(self.num_yards):
                deficit = self._calculate_n_step_deficit(yard_id)
                print(f"Yard {yard_id}: Deficit={deficit}")
            return None
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, '_renderer') and self._renderer is not None:
            self._renderer.close()
            self._renderer = None