# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions, Actions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveAgent', second = 'DefensiveAgent'):
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
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def registerInitialState(self, gameState):
    self.food_eaten = 0
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    next_action = random.choice(bestActions)  # we get next_action
    nextGameState = self.getSuccessor(gameState, next_action) # from next_action we find nextGameState

    foodLeft = len(self.getFood(gameState).asList())

    # if nextgamestate is not pacman, that means the ghost has died or returned home
    if not nextGameState.getAgentState(self.index).isPacman:
      # print('does it reach here when I want it to?')
      self.food_eaten = 0   # reset food_eaten because pacman dropped food after death or drops it at home

    # if nextgamestate's len(food) is decreased by 1, pacman ate it. so increase food eaten by 1
    self.food_eaten += len(self.getFood(gameState).asList()) - len(self.getFood(nextGameState).asList())
    # print(self.food_eaten, 'Total food eaten')

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist

      return bestAction

    return next_action

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveAgent(DummyAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)#self.getScore(successor)
    myPos = successor.getAgentState(self.index).getPosition()

    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    non_invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]

    # Compute distance to the nearest food
    # print(self.food_eaten, 'food')

    # changed it up a bit so it doesn't perform this all the time
    if len(foodList) > 0 and self.food_eaten == 0 and len(invaders) == 0: # This should always be True,  but better safe than sorry
      minFoodDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minFoodDistance

    # test for scared timer

    # for ghost in non_invaders:
        #   print(ghost.scaredTimer, 'test timer')

    # if there exists a non_invader (enemy ghost)
    for ghost in non_invaders:
      if ghost.scaredTimer == 0:  # if ghost is scared, no need to run away!
        if len(non_invaders) > 0:
          # min dist from myposition to ghost position
          features['nonInvader'] = min(self.getMazeDistance(myPos, ghost.getPosition()) for ghost in non_invaders)
        else:  # if not set it to 0
          features['nonInvader'] = 0
        # if distance is too big, just ignore it and set it to 0, otherwise if it's close pacman should run away
        if features['nonInvader'] > 2:
          features['nonInvader'] = 0

    # if I eat a food, go home and gain a point instead of being greedy for more
    # print(self.start)
    if self.food_eaten > 0:
      minHomeDistance = min([self.getMazeDistance(myPos, self.start)])
      features['distanceToHome'] = minHomeDistance

    # if there is an enemy in my territory, forget about eating food, go back and protect your food!
    if len(invaders) > 0:
      # print('does it reach here? Offensive')
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1, 'nonInvader': -10, 'distanceToHome': -10, 'invaderDistance': -10, 'stop': -100}

class DefensiveAgent(DummyAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    non_invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)

    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    # if there exists a non_invader, just follow that pacman to eat him as soon
    # as he enters my territory instead of wandering around randomly
    # also, if len(invaders) does not equal 0, that means enemy is in my territory
    # so chase after him instead of following non_invader
    if len(non_invaders) > 0 and len(invaders) == 0:
      # print('wtf why it not workin')
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in non_invaders]
      features['nonInvader'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    # print(features)
    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'nonInvader': -10, 'stop': -100, 'reverse': -2}
