from NN import Neural_Network as nn
from sim import Action, Point
import math

class Agent:
    def __init__(self):
        pass

    def takeAction(self, data):

        pt = Point(int(round(x + (1000000 * math.cos(angle * (math.pi/180))))), int(round(y + (1000000 * math.sin(angle * (math.pi/180))))))
        
        return [Action(Point(0, 0), 0), Action(Point(0, 0), 0)]
