from typing import List
import pooltool as pt
import random
from utils import State, Event

LIMITS = {
    "V0": (0.1, 5),   # CUE BALL SPEED
    "theta": (0, 90), # CUE INCLINATION
    "phi": (0, 360),  # CUE ANGLE
    "a": (0, 1),      # CUE OFFSET
    "b": (0, 1)       # CUE OFFSET
}

class Pool():

    def __init__(self, visualizable=False) -> None:

        self.visualizable = visualizable
        self.interface = pt.ShotViewer() if self.visualizable else None

        self.table = pt.Table.default()
        self.balls = self.setup_balls()
        self.cue = pt.Cue(cue_ball_id="cue")
        self.shot = pt.System(table=self.table, balls=self.balls, cue=self.cue)

    def setup_balls(self) -> dict:

        '''
        Initialise board state to threeball (and cue ball) default
        '''

        x1, y1 = 0.25, 1.5
        x2, y2 = 0.5, 1.5
        x3, y3 = 0.75, 1.5

        return {
            "cue": pt.Ball.create("cue", xy=(0.5, 0.5)),
            "red": pt.Ball.create("red", xy=(x1, y1), ballset=pt.game.layouts.DEFAULT_SNOOKER_BALLSET),
            "yellow": pt.Ball.create("yellow", xy=(x2, y2), ballset=pt.game.layouts.DEFAULT_SNOOKER_BALLSET),
            "blue": pt.Ball.create("blue", xy=(x3, y3), ballset=pt.game.layouts.DEFAULT_SNOOKER_BALLSET),
        }

    def reset(self) -> None:

        '''
        Completely reset state to default board
        '''

        self.shot.reset_history()
        self.shot.reset_balls()

        self.table = pt.Table.default()
        self.balls = self.setup_balls()
        self.cue = pt.Cue(cue_ball_id="cue")

        del self.shot.table 
        del self.shot.balls
        del self.shot.cue

        self.shot.table = self.table
        self.shot.balls = self.balls
        self.shot.cue = self.cue

    def from_state(self, state : State) -> bool:

        '''
        :param state: State object
        Update the PoolTool state to provided :param state:.
        '''

        self.reset()

        positions = state.positions

        for ball, x_y in positions.items():
            x, y = x_y

            new_ball_state = self.shot.balls[ball].state.copy()
            new_ball_state.s = 0 # stationary
            
            new_ball_state.rvw[0][0] = x
            new_ball_state.rvw[0][1] = y
            new_ball_state.rvw[1][0] = 0
            new_ball_state.rvw[1][1] = 0

            self.shot.balls[ball].history = pt.BallHistory()
            self.shot.balls[ball].history.add(new_ball_state)
            self.shot.balls[ball].state = new_ball_state


        # try:
        #     pt.simulate(self.shot, inplace=True)
        # except Exception as e:
        #     print(e)
        #     return False
        
        return True

    def strike(self, V0, phi, theta, a, b) -> None:
        self.shot.strike(V0=V0, phi=phi, theta=theta, a=a, b=b)
        pt.simulate(self.shot, inplace=True)

    def random_strike(self) -> dict:
        V0 = random.uniform(LIMITS["V0"][0], LIMITS["V0"][1])
        theta = random.uniform(LIMITS["theta"][0], LIMITS["theta"][1])
        phi = random.uniform(LIMITS["phi"][0], LIMITS["phi"][1])
        a = random.uniform(LIMITS["a"][0], LIMITS["a"][1])
        b = random.uniform(LIMITS["b"][0], LIMITS["b"][1])
        self.strike(V0, phi, theta, a, b)
        return {
            "V0": V0,
            "theta": theta,
            "phi": phi,
            "a": a,
            "b": b
        }

    def get_state(self) -> State:
        board_state = self.shot.get_board_state()
        state = State()
        state.from_board_state(board_state)
        return state

    def visualise(self) -> None:
        '''
        Visualize the simulation using PoolTool's ShotViewer
        '''
        if self.visualizable:
            self.interface.show(self.shot)
            self.interface.stop()
        else:
            raise Exception("Cannot visualize without visualizable=True")
    
    def get_events(self) -> List[Event]:
        '''
        returns a list of events starting from the latest shot:
        - collision between balls, cushion, cue stick ;
        - state transition of balls from rolling to sliding etc...
        - pocketing events, bool id & pocket id.
        '''

        pt_events = self.shot.events
        events = []

        for e in pt_events:
            typ_obj : tuple[str, str, str] = e.typ_obj

            if typ_obj == ("","",""):
                continue

            typ_obj = typ_obj[0] + "-" + typ_obj[1] + "-" + typ_obj[2]
            events.append(Event.from_encoding(typ_obj))

        return events