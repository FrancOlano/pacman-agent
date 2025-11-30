import random
import util
import time

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        # I setted treshold to 1 because based on experience it wins most of the time
        self.carrying_treshold = 1 
        

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        self.enemies_index = self.get_opponents(game_state)
        CaptureAgent.register_initial_state(self, game_state)
        # calculate the middle waiting position depending on the color
        if self.red:
            self.wait_middle_position = (game_state.data.layout.width//2 -2, game_state.data.layout.height//2)
            # Suppose most ghost will be around the food in the middle
            self.food_to_avoid = [(17, 6)]
        else:
            self.wait_middle_position = (game_state.data.layout.width//2 +1, game_state.data.layout.height//2 -1)
            self.food_to_avoid = [(14, 9)]

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        my_state = game_state.get_agent_state(self.index)
        num_carrying = my_state.num_carrying
        if num_carrying > self.carrying_treshold:
            actions.remove(Directions.STOP)


        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return best_actions[0]

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    Seeks food until the treshold is crossed and goes back to secure the points.
    If its on deffence, it will try to catch some enemies if they are on its path.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        
        if action == Directions.STOP: features['stop'] = 1

        # Agent info
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        num_carrying = my_state.num_carrying
        features['num_carrying'] = num_carrying
        features['return_distance'] = self.get_maze_distance(my_pos, self.start)

        # Food info
        food_list = self.get_food(successor).as_list()
        food_count = len(food_list)
        min_food_distance = 0
        if food_count > 0:
            food_distances =[]
            for food in food_list:
                # avoid middle food as an strategy
                if food in self.food_to_avoid: continue
                food_distances.append(self.get_maze_distance(my_pos, food))

            min_food_distance = min(food_distances)
        features['food_count'] = food_count
        features['min_food_distance'] = min_food_distance

        # Capsule info
        capsules = self.get_capsules(successor)
        capsule_count = len(capsules)
        capsule_distance = 0
        if capsule_count > 0:
            capsule_distance = min([self.get_maze_distance(my_pos, capsule) for capsule in capsules])

        features['capsule_count'] = capsule_count
        features['capsule_distance'] = capsule_distance

        # Enemies info
        enemies_pos = [successor.get_agent_position(enemy) for enemy in self.enemies_index if successor.get_agent_position(enemy) is not None]
        min_enemy_distance = 0
        if len(enemies_pos) > 0:
            min_enemy_distance = min([self.get_maze_distance(my_pos, enemy) for enemy in enemies_pos])
        enemies_states = [successor.get_agent_state(enemy) for enemy in self.enemies_index ]
        invaders = [enemy for enemy in enemies_states if enemy.is_pacman and enemy.get_position() is not None]
        features['num_invaders'] = len(invaders)
        
        features['min_enemy_distance'] = min_enemy_distance * num_carrying
        
        # Score
        features['successor_score'] = self.get_score(successor)

        return features

    def get_weights(self, game_state, action):
        my_state = game_state.get_agent_state(self.index)
        is_pacman = my_state.is_pacman
        num_carrying = my_state.num_carrying
        successor = self.get_successor(game_state, action)
        enemies_states = [successor.get_agent_state(enemy) for enemy in self.enemies_index]
        scared_enemies = [state.scared_timer > 0 for state in enemies_states]
        scared_enemies_count = scared_enemies.count(True)

        # Go back when carrying too much or when ghosts aren't scared
        if (num_carrying >= self.carrying_treshold and scared_enemies_count == 0) or num_carrying >= self.carrying_treshold +2:
            return_now = -10000000000000000             
        elif num_carrying > 3:
            return_now = -num_carrying * 200
        else:
            return_now = 0

        if self.get_score(successor) > 0:
            capsule_distance = -25
        else:
            capsule_distance = -17

        # Strategy:
        # Minimize food distance, food count, capsule distance and capsule count
        # Maximize enemy distance at all cost if its a pacman
        # Return if its carrying the treshold num of food
        return {
            'successor_score':      10,  
            'food_count':          -250, 
            'min_food_distance':    -25,  
            'capsule_count':       -800, 
            'capsule_distance':    capsule_distance, 
            'min_enemy_distance':   10000000000000  if is_pacman else -1000000000, 
            'return_distance':      return_now, 
            'num_invaders':         -1,   
            'num_carrying':         5,
            'stop':                 -50
        }


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        num_invaders = len(invaders)
        features['num_invaders'] = num_invaders
        if num_invaders > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
        else:
            # If there are no invaders go to the middle and protect the food there
            features['go_middle'] = self.get_maze_distance(my_pos, self.wait_middle_position)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        scared_timer = my_state.scared_timer
        # If it is scared go to atk mode and try to catch some food
        # Worst case scenario it gets killed and restarts as a regular ghost
        if scared_timer > 0:
            # Food info
            food_list = self.get_food(successor).as_list()
            food_count = len(food_list)
            min_food_distance = 0
            if food_count > 0:
                min_food_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])

            features['atk'] = food_count*100 + min_food_distance


        return features

    # Weights designed to look for invaders and kill them.
    # If there are no invaders wait on the middle
    def get_weights(self, game_state, action):
        return {
            'num_invaders': -15000,
            'on_defense': 120,
            'invader_distance': -2000, 
            'stop': -500,              
            'reverse': -1,            
            'atk': -1000000, #If scared always atack
            'go_middle': -500
        }

