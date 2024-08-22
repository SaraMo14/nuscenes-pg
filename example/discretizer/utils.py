from enum import Enum, auto

class Velocity(Enum):
  STOPPED = auto()
  LOW = auto()
  MEDIUM = auto()
  HIGH = auto()
  #VERY_HIGH = auto()
  MOVING = auto()
  def __str__(self):
        return f'{self.__class__.__name__}({self.name})'


class Rotation(Enum):
  RIGHT = auto()
  #SLIGHT_RIGHT = auto()
  FORWARD = auto()
  #SLIGHT_LEFT = auto()
  LEFT = auto()

  def __str__(self):
        return f'{self.__class__.__name__}({self.name})'


class LanePosition(Enum):
    LEFT = auto()
    ALIGNED = auto()
    OPPOSITE = auto()
    NONE = auto() #for all the cases not includend in the previous categories (e.g car headed perpendicular to the road, parkins, etc..)
    #TODO: handle intersections    
    def __str__(self):
            return f'{self.__class__.__name__}({self.name})'

class BlockProgress(Enum):
    START = auto()
    MIDDLE = auto()
    END = auto()
    INTERSECTION = auto()
    NONE = auto() #for all the cases not includend in the previous categories (e.g. car parkings, walkway)

    def __str__(self):
            return f'{self.__class__.__name__}({self.name})'


class NextIntersection(Enum):
    #what is my behavior at the next intersection? Do i go left, STRAIGH or RIGHT?
    # This is something that you know being a driver of the vehicle and affects how you're going to drive.
    # This allows the check for desires.
    RIGHT = auto()
    LEFT = auto()
    STRAIGHT = auto()
    NONE = auto()

    def __str__(self):
        return f'{self.__class__.__name__}({self.name})'


class Detection:
    discretizations = {
        "multiple": ["0", "1-3", "4+"],
        "binary": ["NO", "YES"]
    }

    def __init__(self, count=0, discretization="multiple"):
        self.chunks = self.discretizations[discretization]

        if isinstance(count, str):
            self.count = count
        elif discretization == "multiple":
            
            if count == 0:
                self.count = self.chunks[0]
            elif count < 4:
                self.count = self.chunks[1]
            else:
                self.count = self.chunks[2]
        elif discretization == "binary":
            self.count = self.chunks[1] if count > 0 else self.chunks[0]
    
    def __str__(self) -> str:
        return f'{self.count}'

    def __eq__(self, other):
        return self.count == other.count
    
    def __hash__(self):
        return hash(self.count)

class FrontRightObjects(Detection):
    pass

class FrontLeftObjects(Detection):
    pass

class FrontObjects(Detection):
    pass

class PedestrianNearby(Detection):
    pass


class IsTwoWheelNearby(Enum): #bycicles + scooters
    YES = auto()
    NO = auto()
    
    def __str__(self):
            return f'{self.__class__.__name__}({self.name})'



class IsTrafficLightNearby(Enum):
  YES = auto()
  NO = auto()
  
  def __str__(self):
        return f'{self.__class__.__name__}({self.name})'


class IsZebraNearby(Enum):
  #includes pedestrian crossing and turn stop
  YES = auto()
  NO = auto()
  
  def __str__(self):
        return f'{self.__class__.__name__}({self.name})'
  


class IsStopSignNearby(Enum): 
  #includes Stop Sign and Yield Sign
  YES = auto()
  NO = auto()
  
  def __str__(self):
        return f'{self.__class__.__name__}({self.name})'
  

class SignNearby(Enum): 
  #includes Stop Sign and Yield Sign
  STOP = auto()
  YIELD = auto()
  NO = auto()
  
  def __str__(self):
        return f'{self.__class__.__name__}({self.name})'


class Action(Enum):
  IDLE = auto() 
  TURN_LEFT = auto()
  TURN_RIGHT = auto()
  GAS = auto() 
  BRAKE = auto()
  STRAIGHT = auto() #car keep going straight at same pace
  GAS_TURN_RIGHT= auto()
  GAS_TURN_LEFT= auto()
  BRAKE_TURN_RIGHT = auto()  
  BRAKE_TURN_LEFT = auto()





def calculate_object_distance(value1, value2):
        """
        Calculate the object distance based on the given rule:
        dist(0, 1-3)= 2, dist(0, 4+)= 3, dist(1-3, 4+) = 1
        """
        if value1 == '0':
            if value2 == '1-3':
                return 2
            elif value2 == '4+':
                return 3
        elif value1 == '1-3':
            if value2 == '0':
                return 2
            elif value2 == '4+':
                return 1
        elif value1 == '4+':
            if value2 == '0':
                return 3
            elif value2 == '1-3':
                return 1
        return 0
    


def calculate_velocity_distance(value1, value2):
        """
        Calculate the object distance based on the given rule:
        dist(not moving, slow)= 2, dist(not moving, medium)= 3, dist(not moving, high) = 4
        dist(slow, medium) = 1, dist(slow,high) = 2,
        dist(medium,high) = 1
        """

        if value1 == Velocity.STOPPED:
            if value2 == Velocity.LOW:
                return 2
            elif value2 == Velocity.MEDIUM:
                return 3
            elif value2 == Velocity.HIGH:
                return 4
        elif value1 == Velocity.LOW:
            if value2 == Velocity.STOPPED:
                return 2
            elif value2 == Velocity.MEDIUM:
                return 1
            elif value2 == Velocity.HIGH:
                return 2
        elif value1 == Velocity.MEDIUM:
            if value2 == Velocity.STOPPED:
                return 3
            elif value2 == Velocity.LOW:
                return 1
            elif value2 == Velocity.HIGH:
                return 1
        elif value1 == Velocity.HIGH:
            if value2 == Velocity.STOPPED:
                return 4
            elif value2 == Velocity.LOW:
                return 2
            elif value2 == Velocity.MEDIUM:
                return 1
        return 0  


