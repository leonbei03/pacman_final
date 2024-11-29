# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='Defender', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class BaseAgent(CaptureAgent):

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.border = None
        self.mid = None

    def get_border(self, game_state):
        border = []
        for y in range(game_state.data.layout.height):
            if not game_state.get_walls()[self.mid][y]:  # Check if it's not a wall
                border.append((self.mid, y))
        return border

    def is_in_opponent_territory(self, game_state, agent_index):
        agent_pos = game_state.get_agent_position(agent_index)
        left_side = agent_pos[0] < self.mid
        if self.red:
            return not left_side
        else:
            return left_side

    def is_on_opponent_side(self, game_state, position):
        is_left = position[0] < (game_state.data.layout.width // 2)
        if self.red:
            return not is_left
        else:
            return is_left

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.mid = game_state.data.layout.width // 2

    def choose_action(self, game_state):
        raise NotImplementedError

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


class Offender(BaseAgent):

    def choose_action(self, game_state):
        depth = 2  # Depth for the Minimax search
        alpha = float('-inf')
        beta = float('inf')
        best_action = self.minmax(game_state, depth, self.index, alpha, beta, True)[1]
        return best_action

    def minmax(self, game_state, depth, agent_index, alpha, beta, is_max):
        if depth == 0:
            return self.evaluate(game_state), None

        num_agents = game_state.get_num_agents()
        next_agent = (agent_index + 1) % num_agents
        is_next_max = next_agent == self.index

        if is_max:
            best_score = float('-inf')
            best_action = None
            for action in game_state.get_legal_actions(agent_index):
                successor = self.get_successor(game_state, action)
                score = self.minmax(successor, depth - (next_agent == 0), next_agent, alpha, beta, is_next_max)[0]
                # Evaluate the state resulting from this action
                if score > best_score:
                    best_score = score
                    best_action = action
                alpha = max(alpha, best_score)
                if alpha >= beta:
                    break  # Prune
            return best_score, best_action
        else:
            best_score = float('inf')
            best_action = None
            for action in game_state.get_legal_actions(agent_index):
                successor = self.get_successor(game_state, action)
                score = self.minmax(successor, depth - (next_agent == 0), next_agent, alpha, beta, is_next_max)[
                    0]
                # Evaluate the state resulting from this action
                score = self.evaluate(game_state)
                if score < best_score:
                    best_score = score
                    best_action = action
                beta = min(beta, best_score)
                if alpha >= beta:
                    break  # Prune
            return best_score, best_action

    def evaluate(self, game_state):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state)
        weights = self.get_weights()
        return features * weights


"""
    def get_features(self, game_state):
        
        Returns a counter of features for the state
        
        features = util.Counter()


        # Feature 1: Minimum distance to food spot
        food_list = self.get_food(game_state).asList()
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        # Feature 2: Minimum distance to enemies:
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        if len(enemies) > 0:
            # We have to filter out the ghosts, only they can hit our pacman
            ghosts = [a for a in enemies if not a.isPacman and a.getPosition() is not None]
            ghost_distance = min(
                [self.get_maze_distance(pos, ghost.getPosition()) for ghost in ghosts]) if ghosts else float('inf')
            features['distance_to_ghosts'] = ghost_distance
        return features"""


class Defender(BaseAgent):

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.mid = None
        self.previous_food = set()
        self.border_index = 0

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.mid = self.get_mid(game_state)



    @staticmethod
    def get_mid(game_state):
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        mid_x = width // 2
        mid_y = height // 2
        return mid_x, mid_y



    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        my_pos = game_state.get_agent_position(self.index)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invader = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        current_food = self.get_food_you_are_defending(game_state).as_list()
        eaten_food = set(self.previous_food) - set(current_food)
        best_action = None
        if invader:
            # Get the closest invader's position
            minimum = float('-inf')
            """closest_invader = min(
                invader,
                key=lambda i: self.get_maze_distance(my_pos, i.get_position()),
                default=None
            )
            """
            for i in invader:
                distance = self.get_maze_distance(my_pos, i.get_position())
                if distance > minimum:
                    closest_invader = i.get_position()

            best_action = self.get_move_towards_position(game_state, closest_invader, actions)
        else:
            best_action = self.get_move_towards_position(game_state, self.mid, actions)
        self.previous_food = current_food
        return best_action

    def get_random_action(self, game_state, actions):
        return actions[random.randint(0, len(actions))]

    def get_move_towards_position(self, game_state, target_position, actions):
        best_action = None
        min_distance = float('inf')
        for action in actions:
            successor = game_state.generate_successor(self.index, action)
            successor_position = successor.get_agent_state(self.index).get_position()
            distance = self.get_maze_distance(successor_position, target_position)
            if distance < min_distance:
                min_distance = distance
                best_action = action
        return best_action


class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

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
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            second_best = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    second_best = best_action
                    best_action = action
                    best_dist = dist
            if best_action == Directions.STOP:
                return second_best
            return best_action

        return random.choice(best_actions)
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        if ghosts:
            ghost_distance = float('inf')
            for ghost in ghosts:

                ghost_pos = ghost.get_position()
                dist = self.distancer.get_distance(ghost_pos, my_pos)
                if dist < ghost_distance:
                    ghost_distance = dist
                    features['ghost_distance'] = ghost_distance
        else:
            features['ghost_distance'] = 1000
        # Compute distance to the nearest food
        # Number of food carried
        food_carried = my_state.num_carrying
        features['food_carried'] = food_carried
        mid_x = game_state.data.layout.width // 2
        if not self.red:
            mid_x += 1

        home_distance = abs(my_pos[0] - mid_x)
        features['home_distance'] = home_distance
        if food_carried < 1:
            features['home_distance'] = 0
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features


    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1,
                'ghost_distance': 2, 'food_carried': 10, 'home_distance': -10}


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
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
