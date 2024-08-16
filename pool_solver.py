#
# Pool solver that finds the shot parameters needed to satisfy a sequence of events and a text description of the end state 
#

from pool import Pool
from utils import *
import os
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
    INITIAL_RANDOM = 100
    SEARCH_STEPS = 100


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
            bounds_transformer=Optimisers.BOUNDS_TRANSFORMER,
            allow_duplicate_points=True
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

            if float(Optimisers.BAYES_OPTIMIZER.max["target"])>=1.00:
                search_done = True
                break

        ### Bayesian Optimisation
        if not search_done:
            utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
            for i in range(Optimisers.SEARCH_STEPS):
                next_point = Optimisers.BAYES_OPTIMIZER.suggest(utility)
                target = reward_function(**next_point)
                Optimisers.BAYES_OPTIMIZER.register(params=next_point, target=target)
                # print(f"Search point: {i + 1}/{Optimisers.SEARCH_STEPS} with reward: {target}")

                if float(Optimisers.BAYES_OPTIMIZER.max["target"])>=1.00:
                    print(f"Success at step: {i}/{Optimisers.SEARCH_STEPS}")
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
    def event_reward(events: List[Event], new_events: List[Event]) -> float:
        """
        Reward function for the event sequence
        """

        # print(new_events)
        matched_events = 0
        cue_ball_collided = False  # 用于标记白球和目标球碰撞的事件是否出现

        new_event_index = 0
        for event in events:
            if cue_ball_collided:
                # 跳过new_events中所有白球接下来的运动
                while new_event_index < len(new_events) and new_events[new_event_index].arguments[0] == "cue":
                    new_event_index += 1
            
            if new_event_index >= len(new_events):
                break

            new_event = new_events[new_event_index]

            if event == new_event:
                matched_events += 1
                new_event_index += 1
                if event.event_type == EventType.BALL_BALL_COLLISION and event.arguments[0] == "cue":
                    cue_ball_collided = True
            else:
                break

        reward = matched_events / len(events)
        return reward

                                  
class PoolSolver:
    def __init__(self):
        self.param_space = {
            "V0": [0.25, 5],
            "phi": [0, 360],
        }

        self.pool = Pool()

        self.verbosity = os.getenv("VERBOSITY", "NONE")
    

    # TODO: make the return value a struct of some kind 
    def get_shot(self, state : State, events : List[Event]) -> Tuple[Dict[str, float], State, List[Event], float, float]:

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

            return PoolSolver.rate_shot(POOL, INITIAL_STATE, events, params)
        
        params, rating, std_dev = Optimisers.BayesOptimiser(reward_function, self.param_space)

        #plot_optimizer(Optimisers.BAYES_OPTIMIZER, self.param_space)

        self.pool.from_state(INITIAL_STATE)
        self.pool.strike(**params)
        new_events = self.pool.get_events()

        if self.verbosity == "INFO":
            print(f"Found shot: {params} with rating: {rating}")

        return params, self.pool.get_state(), new_events, rating, std_dev
        
    @staticmethod
    def rate_shot(pool : Pool, state : State, events : List[Event], params : Dict[str, float]) -> float:
        pool.from_state(state)
        try:
            pool.strike(**params)
        except Exception as e:
            return 0
        new_events, _ = pool.get_events(), pool.get_state()
        
        rewards = {
            "event": (1, Optimisers.event_reward(events, new_events)),
        }
        return sum([v[0] * v[1] for v in rewards.values()])
        
