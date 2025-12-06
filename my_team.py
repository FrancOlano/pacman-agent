import random
import util
import time

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, red,
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

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        self.enemies_index = self.get_opponents(game_state)
        self.carrying_treshold = len(self.get_food(game_state).as_list()) * 0.10

        walls = game_state.data.layout.walls
        able_positions = walls.as_list(False)

        width = walls.width
        height = walls.height

        center_y = height / 2
        mid_x = width // 2

        # set the center and border x:
        if self.red:
            side_positions = [p for p in able_positions if p[0] < mid_x]
            center_x = (mid_x + width) / 2 
            border_x = mid_x - 1
        else:
            side_positions = [p for p in able_positions if p[0] >= mid_x]
            center_x = mid_x / 2   
            border_x = mid_x

        # Find position in that half closest to that half-center
        distance_to_middle = [(pos, (pos[0] - center_x) ** 2 + (pos[1] - center_y) ** 2) for pos in side_positions]
        min_dist = distance_to_middle[0][1]
        self.middle_pos = distance_to_middle[0][0]
        for pos_dist in distance_to_middle:
            if pos_dist[1] < min_dist:
                min_dist = pos_dist[1]
                self.middle_pos = pos_dist[0]

        # All border positions on walkable tiles
        self.border_positions = [p for p in able_positions if p[0] == border_x]
        self.middle = center_x 

        # set friend index
        team = self.get_team(game_state)
        team.remove(self.index)
        self.friend_index = team[0]

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        # Update mode and treshold for current gamestate
        self.update_mode(game_state)
        food_left = len(self.get_food(game_state).as_list())
        self.carrying_treshold = food_left * 0.10

        actions = game_state.get_legal_actions(self.index)

        num_carrying = game_state.get_agent_state(self.index).num_carrying
        if num_carrying > self.carrying_treshold:
            actions.remove(Directions.STOP)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        return random.choice(best_actions)

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
        
    def get_game_state_info(self, game_state):
        """
        Sets all common information about the game_state 
        in properties so we dont have to compute them in update_mode, features and weights
        """
        self.my_state = game_state.get_agent_state(self.index)
        self.my_pos = self.my_state.get_position()
        self.scared_timer = self.my_state.scared_timer
        self.num_carrying = self.my_state.num_carrying
        self.num_capsules = len(self.get_capsules(game_state))
        self.enemies_states = [game_state.get_agent_state(enemy) for enemy in self.enemies_index]
        self.enemies_pos = [enemy.get_position() for enemy in self.enemies_states if enemy.get_position() is not None and enemy.scared_timer == 0 and not enemy.is_pacman]
        self.known_enemies_count = len(self.enemies_pos)
        self.invaders = [enemy.get_position() for enemy in self.enemies_states if enemy.is_pacman and enemy.get_position() is not None]
        self.num_invaders = len(self.invaders)
        self.scared_enemies_count = [enemy.scared_timer > 0 for enemy in self.enemies_states].count(True)
        self.food_list = self.get_food(game_state).as_list()
        self.food_count = len(self.food_list)
        self.capsules = self.get_capsules(game_state)
        self.capsule_count = len(self.capsules)
        self.score = self.get_score(game_state)
        self.past_game_state = self.get_previous_observation()
        if self.past_game_state is not None:
            self.past_capsule_count = len(self.get_capsules(self.past_game_state))
        else:
            self.past_capsule_count = 2
        self.return_distance = min([self.get_maze_distance(self.my_pos, pos) for pos in self.border_positions])
        self.friend_pos = game_state.get_agent_state(self.friend_index).get_position()
        self.friend_distance = self.get_maze_distance(self.my_pos, self.friend_pos)
        
    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        successor = self.get_successor(game_state, action)
        self.get_game_state_info(successor)
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

    def update_mode(self, game_state):
        util.raise_not_defined()

# Different modes for a offensive agent
class OffensiveModes:
    FOOD = 'Food'
    CAPSULES = 'Capsules'
    RETURN = 'Return' 
    DEFEND = 'Defend'

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    Seeks food until the treshold is crossed and goes back to secure the points.
    Tries to first look for capsules and then get food.
    If its on deffence, it will try to catch invaders.
    """
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        # Aim for capsules first
        self.mode = OffensiveModes.CAPSULES

    def in_offensive_mode(self):
        return self.mode == OffensiveModes.FOOD or self.mode == OffensiveModes.CAPSULES or self.mode == OffensiveModes.RETURN

    def get_attack_mode(self, capsule_count, scared_enemies_count):
        if capsule_count == 0 or scared_enemies_count > 0:
            return OffensiveModes.FOOD
        return OffensiveModes.CAPSULES

    def update_mode(self, game_state):
        # get game_state information
        self.get_game_state_info(game_state)

        # if we are about to win, we have the carrying_treshold food, 
        # or we are at the same distance to return than the amount of food carrying, RETURN.
        if self.food_count <= 2 or self.num_carrying > self.carrying_treshold or (self.num_carrying >0 and self.return_distance <= self.num_carrying):
            self.mode = OffensiveModes.RETURN
            return

        # If scared, go for capsules/food
        if self.my_state.scared_timer > 0:
            self.mode = self.get_attack_mode(self.num_capsules, self.scared_enemies_count)
            return
        
        # Check whether we encounter a pacman enemy we eat it
        for enemy in self.enemies_states:
            if enemy.is_pacman and enemy.get_position() is not None and not self.my_state.is_pacman:
                self.mode = OffensiveModes.DEFEND
                return

        if self.mode == OffensiveModes.FOOD:
            # When reaching the goal of carrying food change to return
            if self.num_carrying > self.carrying_treshold:
                self.mode = OffensiveModes.RETURN

        elif self.mode == OffensiveModes.CAPSULES:
            # When eating a capsule or no more capsules, change to food
            if self.past_capsule_count > self.num_capsules or self.num_capsules == 0:
                self.mode = OffensiveModes.FOOD

        elif self.mode == OffensiveModes.RETURN:
            # If food has been returned, change to capsules/food
            if self.num_carrying == 0:
                self.mode = self.get_attack_mode(self.num_capsules, self.scared_enemies_count)

        elif self.mode == OffensiveModes.DEFEND:
            # If still on Defensive duty, continue, else change to get capsules/food
            for enemy in self.enemies_states:
                if enemy.is_pacman and enemy.get_position() is not None and not self.my_state.is_pacman:
                    return
            self.mode = self.get_attack_mode(self.num_capsules, self.scared_enemies_count)


    def get_features(self, game_state, action):
        features = util.Counter()

        if self.in_offensive_mode() and self.known_enemies_count > 0 and self.my_state.is_pacman:
            features["enemy_distance"] = min([self.get_maze_distance(self.my_pos, pos) for pos in self.enemies_pos])

        if self.mode == OffensiveModes.FOOD:
            # Get distance to nearest food
            if self.food_count > 0:
                features["food_distance"] = min([self.get_maze_distance(self.my_pos, pos) for pos in self.food_list])
            else:
                features["food_distance"] = -1 # negative to make weights positive
            features["food_count"] = self.food_count

        elif self.mode == OffensiveModes.CAPSULES:
            # Get distance to nearest capsule
            if self.capsule_count > 0:
                features["capsule_distance"] = min([self.get_maze_distance(self.my_pos, capsule) for capsule in self.capsules])
            else:
                features["capsule_distance"] = -1 # negative to make weights positive
            features["capsule_count"] = self.capsule_count
            features["food_count"] = self.food_count

        elif self.mode == OffensiveModes.RETURN:
            # Get distance to return
            features["return_distance"] = self.return_distance

        elif self.mode == OffensiveModes.DEFEND:
            # get distance to ghost
            if self.num_invaders > 0:
                features["enemy_distance"] = min([self.get_maze_distance(self.my_pos, enemy_pos) for enemy_pos in self.invaders])
            else:
                features["enemy_distance"] = -1
            features["enemy_count"] = self.num_invaders
            features["friend_distance"] = self.friend_distance

        return features

    # Weights ajusted manually to try to follow policy
    def get_weights(self, game_state, action):
        weights = dict()

        if self.mode == OffensiveModes.FOOD:
            weights["food_distance"] =  -1
            weights["food_count"] = -100
            weights["enemy_distance"] = 1
            
        elif self.mode == OffensiveModes.CAPSULES:
            weights["capsule_distance"] =  -1
            weights["capsule_count"] = -100
            weights["food_count"] = -80 # if there is some food on the way catch it

        elif self.mode == OffensiveModes.RETURN:
            weights["return_distance"] =  -10

        elif self.mode == OffensiveModes.DEFEND:
            weights["enemy_distance"] = -2
            # If scared, do not try to eat
            if self.scared_timer > 0:
                weights["enemy_count"] = 5
            else:
                weights["enemy_count"] = -100
            #try to get far from friend to imprison an invader
            weights["friend_distance"] = 0.5  

        return weights

# Different modes for a deffensive agent
class DefensiveModes:
    ATTACK = 'Attack'
    MIDDLE = 'Middle'
    DEFEND = 'Defend'
    RETURN = 'Return'

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    Defensive agent first goes to the middle.
    If there are no enemies there, try to attack and get some points
    If an invader is detected, follow it.
    """
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        # start going to the middle position
        self.mode = DefensiveModes.MIDDLE

    def is_middle(self):
        return self.my_pos[0] == self.middle
    
    def update_mode(self, game_state):
        # get game_state information
        self.get_game_state_info(game_state)

        # Always defend if there are invaders
        if self.num_invaders > 0:
            self.mode = DefensiveModes.DEFEND
            return
        
        # Attack if it is scared or there are no known enemies in the middle
        if self.scared_timer > 0 or (self.mode == DefensiveModes.MIDDLE and self.is_middle() and self.known_enemies_count == 0):
            self.mode = DefensiveModes.ATTACK
            return
        
        # go to middle always there is not much happening
        if (self.mode == DefensiveModes.DEFEND and self.num_invaders == 0) or (self.mode == DefensiveModes.RETURN and self.num_carrying == 0):
            self.mode = DefensiveModes.MIDDLE
            return
        
        # return if is not safe or we already have enough food
        if self.mode == DefensiveModes.ATTACK and (self.known_enemies_count > 0 or self.num_carrying > self.carrying_treshold):
            self.mode = DefensiveModes.RETURN
            return


    def get_features(self, game_state, action):
        features = util.Counter()

        if self.mode == DefensiveModes.DEFEND:
            if self.num_invaders > 0:
                features["enemy_distance"] = min([self.get_maze_distance(self.my_pos, enemy_pos) for enemy_pos in self.invaders])
            else:
                features["enemy_distance"] = -1
            features["enemy_count"] = self.num_invaders

        elif self.mode == DefensiveModes.MIDDLE:
            features["middle_pos_distance"] = self.get_maze_distance(self.my_pos, self.middle_pos)

        elif self.mode == DefensiveModes.ATTACK:
            if self.capsule_count > 0:
                features["capsule_distance"] = min([self.get_maze_distance(self.my_pos, capsule) for capsule in self.capsules])
            else:
                features["capsule_distance"] = -1 # negative to make weights positive
            features["capsule_count"] = self.capsule_count

            if self.food_count > 0:
                features["food_distance"] = min([self.get_maze_distance(self.my_pos, pos) for pos in self.food_list])
            else:
                features["food_distance"] = -1 # negative to make weights positive
            features["food_count"] = self.food_count

        elif self.mode == OffensiveModes.RETURN:
            # Get distance to return
            features["return_distance"] = self.return_distance

        return features

    # Weights adjusted manually to try to follow policy correctly
    def get_weights(self, game_state, action):
        weights = dict()

        if self.mode == DefensiveModes.DEFEND:
            weights["enemy_distance"] = -1
            weights["enemy_count"] = -100

        elif self.mode == DefensiveModes.MIDDLE:
            weights["middle_pos_distance"] = -1

        elif self.mode == DefensiveModes.ATTACK:
            weights["capsule_distance"] = -2
            weights["capsule_count"] = -100
            weights["food_distance"] = -1
            weights["food_count"] = -100

        elif self.mode == OffensiveModes.RETURN:
            weights["return_distance"] = -10

        return weights
