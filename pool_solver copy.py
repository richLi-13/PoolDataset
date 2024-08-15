#
# Pool solver that finds the shot parameters needed to satisfy a sequence of events and a text description of the end state 
#

from pool import Pool
from utils import *
import torch, PIL, os
from typing import List, Dict, Tuple

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer, UtilityFunction

class Optimisers:

    default_params = {
        "V0": 1,
        "theta": 0,
        "phi": 0,
        "a": 0,
        "b": 0
    }

    ### Search Parameters
    ###    - Initial random points to sample in the parameter space
    ###    - Bayesian opt search steps to perform
    INITIAL_RANDOM = 150
    SEARCH_STEPS = 20

    ### Below this threshold, we consider a ball to be sufficiently close to a target position
    DISTANCE_THRESHOLD = 0.25

    ### 白球应该在的目标位置
    VALID_DESCRIPTIONS = {
        "any":           (0.5, 1.0),
        "top" :          (0.5, 1.5),
        "bottom" :       (0.5, 0.5),
        "top left" :     (0.25, 1.75),
        "top right" :    (0.75, 1.75),
        "bottom left" :  (0.25, 0.25),
        "bottom right" : (0.75, 0.25),
    }

    BAYES_OPTIMIZER = None
    BOUNDS_TRANSFORMER = None

    @staticmethod
    def BayesOptimiser(reward_function, param_space : dict) -> Tuple[dict, float, float]:
        
        # Bounded region of parameter space
        pbounds = {
            'V0': param_space["V0"], 
            'phi': param_space["phi"]
        }

        ### Bounds Transformer focuses the search on a smaller region of the parameter space as the search progresses, good and bad
        Optimisers.BOUNDS_TRANSFORMER = SequentialDomainReductionTransformer(
            gamma_osc=0.8,
            gamma_pan=1.0,
        )
        Optimisers.BAYES_OPTIMIZER = BayesianOptimization(
            f=reward_function,
            pbounds=pbounds,
            random_state=1,
            bounds_transformer=Optimisers.BOUNDS_TRANSFORMER
        )

        ### Initial random points to sample in the parameter space
        search_done = False
        for i in range(Optimisers.INITIAL_RANDOM):
            next_point = {
                "V0": np.random.uniform(*param_space["V0"]),
                "phi": np.random.uniform(*param_space["phi"])
            }
            target = reward_function(**next_point)
            Optimisers.BAYES_OPTIMIZER.register(params=next_point, target=target)
            # print(f"Initial random point: {i}/{Optimisers.INITIAL_RANDOM} with reward: {target}")

            if float(Optimisers.BAYES_OPTIMIZER.max["target"])>0.95:
                search_done = True
                break

        ### Bayesian Optimisation
        if not search_done:
            utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
            for i in range(Optimisers.SEARCH_STEPS):
                next_point = Optimisers.BAYES_OPTIMIZER.suggest(utility)
                target = reward_function(**next_point)
                Optimisers.BAYES_OPTIMIZER.register(params=next_point, target=target)
                # print(f"Search point: {i}/{Optimisers.SEARCH_STEPS} with reward: {target}")

                if float(Optimisers.BAYES_OPTIMIZER.max["target"])>0.95:
                    break

        best_shot = Optimisers.BAYES_OPTIMIZER.max
        
        # Get variance of the best shot
        _, std = Optimisers.BAYES_OPTIMIZER._gp.predict(np.array([best_shot["params"]["V0"], best_shot["params"]["phi"]]).reshape(1, -1), return_std=True)

        # Convert std array to float 
        std = float(sum(std) / len(std))
        
        return {
            "V0": float(best_shot["params"]["V0"]),
            "phi": float(best_shot["params"]["phi"]),
            "theta": Optimisers.default_params["theta"],
            "a": Optimisers.default_params["a"],
            "b": Optimisers.default_params["b"]
        }, float(best_shot["target"]), std
    
    @staticmethod
    def description_reward(description : str, events : List[Event]) -> float:
        """
        Reward function for the description of the end state
        """

        description = description.replace(".", "")
        description = description.lower().strip()
        
        # Check if the description is valid
        if not description in Optimisers.VALID_DESCRIPTIONS.keys():
            return 0
        
        # If the description is any, then we return 1
        if description == "any":
            return 1

        # Check if the cue ball stop event is in the events
        cue_stop_event = None
        for e in events:
            if e.event_type == EventType.BALL_STOP and e.arguments[0] == "cue" and e.pos:
                cue_stop_event = e
                break
        if not cue_stop_event:
            # Likely potted the cue ball
            return 0
        
        target_position = Optimisers.VALID_DESCRIPTIONS[description]  
        cue_stop_event_position = cue_stop_event.pos

        euclidean_distance = np.linalg.norm(
            np.array(cue_stop_event_position) - np.array(target_position)
        )

        # If the distance is less than a threshold, then we return 1, as we are close enough and any further optimization is not needed
        if euclidean_distance < Optimisers.DISTANCE_THRESHOLD:
            return 1.0
        
        return 1.0 - np.clip(euclidean_distance / 2.0, 0, 1)

    # @staticmethod
    # def event_reward(events : List[Event], new_events : List[Event]) -> float:
    #     """
    #     Reward function for the event sequence
    #     """
        # ord_r = 0
        # for event in events:
            
        #     if event in new_events:
        #         ord_r += 1
        #         new_events = new_events[new_events.index(event)+1:]

        #     else:
        #         # # Check for near misses
        #         # closest_event = Event.get_closest_event(event, new_events)
        #         # if closest_event:
        #         #     ord_r += Event.distance(event, closest_event)
        #         #     new_events = new_events[new_events.index(closest_event)+1:]
        #         # else:
        #         #     # If there is no near miss, then we break as order must be preserved
        #         #     break
        #         break

        # ord_r = ord_r / len(events)

        # return ord_r

    
    @staticmethod
    def event_reward(events: List[Event], new_events: List[Event]) -> float:
        """
        Reward function for the event sequence
        """
        matched_events = 0
        for event, new_event in zip(events, new_events):
            if event == new_event:
                matched_events += 1
            else:
                break

        reward = matched_events / len(events)
        return reward


    
    @staticmethod
    def correct_event_reward(events: List[Event], new_events: List[Event]) -> float:
        """
        Reward function for the correct event in the goal
        """
        # Find the special event in the original events list
        special_event = None
        for event in events:
            if event.event_type == EventType.BALL_CUSHION_COLLISION:
                special_event = event
                break
        
        if special_event:  # If special event is found (翻袋，或者非直接击球)
            cushion_collisions = [e for e in new_events if e.event_type == EventType.BALL_CUSHION_COLLISION]
            
            # Check if there is exactly one cushion collision with the same arguments
            if len(cushion_collisions) == 1 and cushion_collisions[0].arguments[0] == special_event.arguments[0]:
                return 1
            else:
                return 0
        else:  # If no special event, ensure there are no cushion collisions (直接击球)
            for e in new_events:
                if e.event_type == EventType.BALL_CUSHION_COLLISION:
                    return 0
            return 1
        
    @staticmethod
    def length_reward(intended_events : List[Event], actual_events : List[Event]) -> float:
        """
        Reward function for the length of the event sequence, reward approaches 0 and length of new_events increases
        """
 
        r_len = len(actual_events) - ( 2 + len(intended_events) ) # plus two for hitting cue ball and cue ball stopping
        return 0.9 ** r_len

                                     
class PoolSolver:
    def __init__(self):
        self.param_space = {
            "V0": [0.25, 5],
            "phi": [0, 360],
        }

        self.pool = Pool()

        self.verbosity = os.getenv("VERBOSITY", "NONE")
    

    # TODO: make the return value a struct of some kind 
    def get_shot(self, state : State, events : List[Event], end_state_description : str, weights) -> Tuple[Dict[str, float], State, List[Event], float, float]:

        # 1. embed end state text
        # 2. perform search in param space 
        # 3. optimise for the params that cause the event sequence and end state is close to the embedded end state text

        # Some simple checks about the board
        if len(events) == 0:
            return Optimisers.default_params, state, [], 0, 1
        elif state.is_potted("cue"):
            return Optimisers.default_params, state, [], 0, 1
        elif all([state.is_potted(ball) for ball in ["red", "yellow", "blue"]]):
            return Optimisers.default_params, state, [], 1, 1

        INITIAL_STATE = state
        POOL = self.pool

        def reward_function(V0, phi) -> float:
            
            params = {
                "V0": V0,
                "theta": Optimisers.default_params["theta"],
                "phi": phi,
                "a": Optimisers.default_params["a"],
                "b": Optimisers.default_params["b"]
            }

            return PoolSolver.rate_shot(POOL, INITIAL_STATE, events, end_state_description, params, weights)
        
        params, rating, std_dev = Optimisers.BayesOptimiser(reward_function, self.param_space)

        #plot_optimizer(Optimisers.BAYES_OPTIMIZER, self.param_space)

        self.pool.from_state(INITIAL_STATE)
        self.pool.strike(**params)
        new_events = self.pool.get_events()

        if self.verbosity == "INFO":
            print(f"Found shot: {params} with rating: {rating}")

        return params, self.pool.get_state(), new_events, rating, std_dev
        
    @staticmethod
    def rate_shot(pool : Pool, state : State, events : List[Event], end_state_description : str, params : Dict[str, float], weights: List[float]) -> float:
        pool.from_state(state)
        try:
            pool.strike(**params)
        except Exception as e:
            return 0
        new_events, _ = pool.get_events(), pool.get_state()
        
        rewards = {
            # "description": (weights[0], Optimisers.description_reward(end_state_description, new_events)),
            "event": (1, Optimisers.event_reward(events, new_events)),
            # "special_event": (weights[2], Optimisers.correct_event_reward(events, new_events))
        
        }
        # print(rewards)
        return sum([v[0] * v[1] for v in rewards.values()])
        
