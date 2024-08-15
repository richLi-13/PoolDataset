import numpy as np
import base64, io, PIL, time
from typing import Tuple, List, Dict
from check_feasibility import get_pocket_pos

def extract_commands(response : str, commands : List[str]) -> dict:
    """
    Extract the commands from the response
    """
    commands = [ c.strip() for c in commands]
    words = response.split()
    command_dict = {}
    current_command = None

    # Extract commands
    for word in words:
        if word in commands:
            current_command = word
            command_dict[current_command] = []
        elif current_command:
            command_dict[current_command].append(word)

    # Convert list of words to string
    final_dict = {}
    for k, v in command_dict.items():

        if len(v) == 0:
            continue

        final_dict[k[:-1]] = " ".join(v)

    return final_dict

class Image():
    def __init__(self, image_base64, image_tensor) -> None:
        self.image_base64 = image_base64
        self.image_tensor = image_tensor

    def save_image(self, filename : str) -> None:
        image_data = base64.b64decode(self.image_base64)
        with open(filename, "wb") as f:
            f.write(image_data)

    def show_image(self) -> None:
        PIL.Image.open(io.BytesIO(base64.b64decode(self.image_base64))).show()

def draw_pool_table(board_state, image_width=168, image_height=336) -> Image :
    board_state = board_state.copy()

    # Fixed hole positions
    holes = [
        (0.0, 0.0), 
        (0.0, 2.0), 
        (-0.025, 1,0),
        (1.0, 0.0),
        (1.0, 2.0),
        (1.025, 1,0),
    ]

    # Change cue ball color to white
    if "cue" in board_state:
        board_state["white"] = board_state.pop("cue")
    
    # Create a blank white image
    image = PIL.Image.new("RGB", (image_width, image_height), "green")
    draw = PIL.ImageDraw.Draw(image)
    
    # Convert coordinates from 0-1 in X and 0-2 in Y to pixel coordinates
    def convert_coordinates(position):
        x = int(position[0] * image_width)
        y = int((2.0 - position[1]) * image_height / 2) # FIX: Flip Y axis as image was upside down
        return (x, y)
    
    # Draw holes
    HOLE_SCALE = 0.05
    hole_radius = int(image_width * HOLE_SCALE)  # proportional to image width
    for hole_position in holes:
        hole_position_pixel = convert_coordinates(hole_position)
        draw.ellipse((hole_position_pixel[0] - hole_radius, hole_position_pixel[1] - hole_radius,
                      hole_position_pixel[0] + hole_radius, hole_position_pixel[1] + hole_radius),
                     fill="black")
    
    # Draw balls
    BALL_SCALE = 0.03
    ball_radius = int(image_width * BALL_SCALE)  # proportional to image width
    for ball_color, ball_position in board_state.items():
        clipped_ball_position = [0, 0]
        clipped_ball_position[0] = np.clip(ball_position[0], -100, 100)
        clipped_ball_position[1] = np.clip(ball_position[1], -100, 100)
        ball_position_pixel = convert_coordinates(clipped_ball_position)
        draw.ellipse((ball_position_pixel[0] - ball_radius, ball_position_pixel[1] - ball_radius,
                      ball_position_pixel[0] + ball_radius, ball_position_pixel[1] + ball_radius),
                     fill=ball_color)
        
    # Pad image with white space, to make a square image (2*width, height)
    image = image.crop((0, 0, 2*image_width, image_height))
    
    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode()

    # Convert image to numpy array 
    image = np.array(image).reshape(1, 3, 336, 336)
    
    return Image(image_base64, image)


class EventType:
    NULL = -2
    ERROR = -1
    STICK_BALL = 0
    BALL_POCKET = 1
    BALL_BALL_COLLISION = 2
    BALL_CUSHION_COLLISION = 3
    BALL_STOP = 4

class Event():

    VALID_BALLS = ["cue", "yellow", "blue", "red"]
    VALID_POCKETS = ["rt", "lt", "rb", "lb", "rc", "lc"]

    w = 1.9812 / 2
    l = 1.9812

    POCKET_POSITIONS = {
        'lt': (0, l),
        'rt': (w, l),
        'lb': (0, 0),
        'rb': (w, 0),
        'lc': (0, l / 2),
        'rc': (w, l / 2)
    }

    DISTANCE_THRESHOLD = 0.0

    def __init__(self) -> None:
        self.encoding : str = ""
        self.event_type : EventType = EventType.NULL
        self.arguments : Tuple[str, str] = ("","")
        self.pos : Tuple[float, float] = None

    @staticmethod
    def distance(event1 : 'Event', event2 : 'Event') -> float:
        '''
        Calculate the distance between two events. 
        '''

        ### Returning a value for how close the events are, higher is better as we are adding this value to the bayeso opt
        ### R = eps ** dist, where eps is a small number and dist is the distance between the two events
        ### Max 0.9 so that a near miss is never chosen over the actual event
        def reward(dist):
            return 0.1 ** dist - 0.1
        
        # Pocketing balls events are close to cushion events that have a position 
        if event1.event_type == EventType.BALL_POCKET and event1.arguments[1] != "" and event2.event_type == EventType.BALL_CUSHION_COLLISION and event2.pos is not None:
            dist = np.linalg.norm(np.array(Event.POCKET_POSITIONS[event1.arguments[1]]) - np.array(event2.pos))
            return reward(dist)
        if event2.event_type == EventType.BALL_POCKET and event1.arguments[1] != "" and event1.event_type == EventType.BALL_CUSHION_COLLISION and event1.pos is not None:
            dist = np.linalg.norm(np.array(Event.POCKET_POSITIONS[event2.arguments[1]]) - np.array(event1.pos))
            return reward(dist)
        
        # Pocketing balls events are close to ball stop events that have a position
        if event1.event_type == EventType.BALL_POCKET and event1.arguments[1] != "" and event2.event_type == EventType.BALL_STOP and event2.pos is not None:
            dist = np.linalg.norm(np.array(Event.POCKET_POSITIONS[event1.arguments[1]]) - np.array(event2.pos))
            return reward(dist)
        if event2.event_type == EventType.BALL_POCKET and event1.arguments[1] != "" and event1.event_type == EventType.BALL_STOP and event1.pos is not None:
            dist = np.linalg.norm(np.array(Event.POCKET_POSITIONS[event2.arguments[1]]) - np.array(event1.pos))
            return reward(dist)

        return 0.0
        
    @staticmethod
    def parse_position(pos : str) -> Tuple[float, float]:
        '''
        Parse a position string, like "(0.1, 0.2)" into a tuple of floats
        '''
        pos = pos.replace("(", "")
        pos = pos.replace(")", "")
        pos = pos.split(",")
        try:
            return (float(pos[0]), float(pos[1]))
        except:
            print(f"Invalid position string: {pos}")
            raise ValueError

    @staticmethod
    def get_closest_event(event : 'Event', events : List['Event']) -> 'Event':
        '''
        Get the event in the list of events that is closest to the given event
        '''
        closest_event = None
        min_distance = np.inf

        for e in events:
            distance = Event.distance(event, e)
            if distance > Event.DISTANCE_THRESHOLD and distance < min_distance:
                min_distance = distance
                closest_event = e

        return closest_event

    @staticmethod
    def get_cooccurrence(events1 : List['Event'], events2 : List['Event']) -> List['Event']:
        '''
        Get the list of events that occur in both lists of events
        '''
        cooccurrence = []
        for event1 in events1:
            for event2 in events2:
                if event1 == event2:
                    cooccurrence.append(event1)

        return cooccurrence

    @staticmethod
    def ball_collision(ball1 : str, ball2 : str = "", pos : str = "") -> 'Event':
        '''
        A collision between two balls, ball1 -> ball2. If ball2 is not specified then it is a generic collision with ball1 and some other ball.
        '''
        event = Event()

        if ball1 not in Event.VALID_BALLS:
            return event
        
        event.event_type = EventType.BALL_BALL_COLLISION
        event.pos = Event.parse_position(pos) if pos != "" else None

        if ball2 == "":
            event.arguments = (ball1, "")
            event.encoding = f"ball-ball-{ball1}"
            return event
        
        if ball2 not in Event.VALID_BALLS:
            return event

        event.arguments = (ball1, ball2)
        event.encoding = f"ball-ball-{ball1}-{ball2}"
        return event
    
    @staticmethod
    def ball_pocket(ball : str, pocket : str = "", pos : str = "") -> 'Event':
        '''
        A ball being pocketed in a pocket. If pocket is not specified then it is a generic pocket.
        Balls = ["cue", "blue", "red", "yellow"]
        Pockets = ["rt", "lt", "rb", "lb", "rc", "lc"]
        '''
        event = Event()

        if ball not in Event.VALID_BALLS:
            return event
        
        event.event_type = EventType.BALL_POCKET
        event.pos = Event.parse_position(pos) if pos != "" else None

        if pocket == "":
            event.arguments = (ball, "")
            event.encoding = f"ball-pocket-{ball}"
            return event
        
        if pocket not in Event.VALID_POCKETS:
            return event

        event.arguments = (ball, pocket)
        event.encoding = f"ball-pocket-{ball}-{pocket}"
        return event
    
    @staticmethod
    def ball_cushion(ball : str, c_id : str = "", pos : str = "") -> 'Event':
        '''
        A ball colliding with a cushion.
        '''
        event = Event()

        if ball not in Event.VALID_BALLS:
            return event

        event.event_type = EventType.BALL_CUSHION_COLLISION
        event.pos = Event.parse_position(pos) if pos != "" else None

        if c_id == "":
            event.arguments = (ball,"")
            event.encoding = f"ball-cushion-{ball}"
            return event

        # TODO: cant find where a list of cushion ids could be found
        # if c_id not in Event.VALID_CUSHIONS:
        #     return event
        
        event.arguments = (ball, c_id)
        event.encoding = f"ball-cushion-{ball}-{c_id}"

        return event

    @staticmethod
    def ball_stop(ball : str, pos : str = "") -> 'Event':
        '''
        A ball stopping, at a particular position if supplied.
        '''
        event = Event()
        event.pos = Event.parse_position(pos) if pos != "" else None

        if ball not in Event.VALID_BALLS:
            return event

        event.event_type = EventType.BALL_STOP
        event.arguments = (ball,"")
        event.encoding = f"ball-stop-{ball}"
        return event
    
    @staticmethod
    def stick_ball(ball : str, pos : str = "") -> 'Event':
        '''
        The cue ball being struck.
        '''
        event = Event()
        event.pos = Event.parse_position(pos) if pos != "" else None

        if ball not in Event.VALID_BALLS:
            return event

        event.event_type = EventType.STICK_BALL
        event.arguments = (ball,"")
        event.encoding = f"stick-ball-{ball}"
        return event

    @staticmethod
    def from_encoding(encoding_str : str) -> 'Event':

        def error():
            print(f"Invalid event encoding: {encoding_str}")
            return Event()
        
        def encoding_index(encoding, index):
            if len(encoding) <= index:
                return ""
            return encoding[index]

        encoding = encoding_str.strip()
        encoding = encoding.replace("\\n", "")
        encoding = encoding.replace(" ", "")
        encoding = encoding.strip()

        encoding = encoding.lower()
        encoding = encoding.split("-")

        e_type = encoding[0] + "-" + encoding[1]
        args = [
            encoding_index(encoding, 2),
            encoding_index(encoding, 3),
        ]
        pos = encoding_index(encoding, 4)

        # print({
        #     "encoding": encoding_str,
        #     "e_type": e_type,
        #     "args": args,
        #     "pos": pos
        # })

        if args == ["", ""]:
            return error()
        
        if e_type == "stick-ball":
            return Event.stick_ball(args[1], pos)
        
        elif e_type == "ball-pocket":
            return Event.ball_pocket(args[0], args[1], pos)
            

        elif e_type == "ball-ball":
            return Event.ball_collision(args[0], args[1], pos)

        elif e_type == "ball-cushion":
            return Event.ball_cushion(args[0], args[1], pos)
        
        elif e_type == "ball-stop":
            return Event.ball_stop(args[0], pos)

        return Event()

    def to_encoding(self) -> str:
        return self.encoding
    
    def __eq__(self, other : 'Event') -> bool:

        if not isinstance(other, Event):
            return False
        
        if self.encoding == other.encoding:
            return True

        # Check for generic events
        either_generic = self.arguments[1] == "" or other.arguments[1] == ""
        if either_generic and self.event_type == other.event_type:
            return self.arguments[0] == other.arguments[0]

        return False

    
    def __str__(self) -> str:
        return self.encoding
    
    def __repr__(self) -> str:
        return self.encoding

class State():
    def __init__(self, positions : Dict[str, List[float]] = None, params : Dict[str, float] = None, random : bool = False):
        '''
        :param positions: Dict[str, List[float]]
        :param params: Dict[str, float] : V0, theta, phy, a,b
        '''
        self.positions = positions
        self.params = params
        self.static = params is None
        self.ball_radius = 0.028575

        if random:
            self.randomize()

    def is_potted(self, ball : str) -> bool:
        '''
        Check if a ball is potted by seeing if the coordinates are at infinity
        '''
        return self.positions[ball][0] == np.inf and self.positions[ball][1] == np.inf

    def from_board_state(self, board_state) -> None:
        self.params = None
        self.static = True

        positions : dict = board_state["balls"]
        self.positions = {
            "cue": positions.get("cue", [np.inf, np.inf]),
            "blue": positions.get("blue", [np.inf, np.inf]),
            "red": positions.get("red", [np.inf, np.inf]),
            "yellow": positions.get("yellow", [np.inf, np.inf]),
        }
    
    def get_image(self) -> Image:
        return draw_pool_table(self.positions)
    
    def get_state_description(self) -> str:
        caption = ''
        for key, el in self.positions.items():

            # Check if potted
            if el[0] == np.inf and el[1] == np.inf:
                caption += f'{key} ball is potted. '
                continue

            quadrant = ''
            if el[0] < 0.33:
                quadrant += 'near the left-{} pocket'
            elif el[0] > 0.33 and el[0] < 0.66:
                quadrant += 'in between the left-{} and right-{} pockets'
            else:
                quadrant += 'near the right-{} pocket'
            if el[1] < 0.66:
                column = 'bottom'
            elif el[1] > 0.66 and el[1] < 1.33:
                column = 'center'
            else:
                column = 'top'

            if 'in between' in quadrant:
                quadrant = quadrant.format(column, column)
            else:
                quadrant = quadrant.format(column)
            caption += f'{key} ball is {quadrant}. '
        return caption

    def set_params(self, params) -> 'State':
        self.params = params
        self.static = False
        return self
    
    def copy(self) -> 'State':
        return State(self.positions.copy(), self.params.copy())
    
    def randomize(self) -> None:

        # Use time as seed
        seed = int(time.time())
        np.random.seed(seed)

        self.positions = {
            "cue":      [np.random.rand(), 2.0 * np.random.rand()],
            "blue":     [np.random.rand(), 2.0 * np.random.rand()],
            "red":      [np.random.rand(), 2.0 * np.random.rand()],
            "yellow":   [np.random.rand(), 2.0 * np.random.rand()],
        }

        # Clip balls to be within table bounds (0.1 - 0.9 in X and 0.1 - 1.9 in Y)
        for ball in self.positions:
            self.positions[ball][0] = np.clip(self.positions[ball][0], 0.1, 0.9)
            self.positions[ball][1] = np.clip(self.positions[ball][1], 0.1, 1.9)

        # If balls are overlapping then randomize again
        if self.balls_overlapping():
            self.randomize()
    
    def balls_overlapping(self) -> bool:
        for ball1 in self.positions:
            for ball2 in self.positions:
                if ball1 != ball2:
                    if np.linalg.norm(np.array(self.positions[ball1]) - np.array(self.positions[ball2])) < 2*self.ball_radius:
                        return True
        return False

    def to_json(self) -> dict:
        return {
            "positions": self.positions,
            "params": self.params
        }
    
    @staticmethod
    def from_json(json_dict : dict) -> 'State':
        return State(json_dict["positions"], json_dict["params"])

def plot_optimizer(optimizer, param_space):

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    # Retrieve the target values the optimizer has found
    X = np.array([res["params"]["V0"] for res in optimizer.res] + [optimizer.max["params"]["V0"]])
    Y = np.array([res["params"]["phi"] for res in optimizer.res] + [optimizer.max["params"]["phi"]])
    Z = np.array([res["target"] for res in optimizer.res] + [optimizer.max["target"]])

    # Define the limits for the plot
    x_min, x_max = 0, 5
    y_min, y_max = 0, 360

    # Define the size of the grid. Keep the grid square-shaped
    grid_size = 100
    heatmap = np.zeros((grid_size, grid_size))

    # Assign Z values to the heatmap grid
    for i in range(len(Z)):

        x_idx = int((X[i] - x_min) / (x_max - x_min) * grid_size)
        y_idx = int((Y[i] - y_min) / (y_max - y_min) * grid_size)

        if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
            if heatmap[y_idx, x_idx] < Z[i]:
                heatmap[y_idx, x_idx] = Z[i]

    # Apply a Gaussian filter
    sigma = 2  # Adjust the sigma value for desired smoothness
    heatmap = gaussian_filter(heatmap, sigma=sigma)

    # Plot the heatmap keeping the aspect ratio square
    plt.imshow(heatmap, origin='lower', cmap='viridis', extent=[x_min, x_max, y_min, y_max], aspect='auto')

    plt.colorbar(label='Density')
    plt.clim(vmin=np.min(heatmap), vmax=np.max(heatmap))
    plt.xlabel('V0')
    plt.ylabel('phi')
    plt.title('Z Value Heatmap with Gaussian Filter')
    plt.show()