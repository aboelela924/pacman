# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
import numpy as np
from optparse import OptionParser
import inspect

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        foodLocations = []
        for row in newFood:
            foodLocations.append(row)
        foodDistance  = 0
        foodCount = 0
        foodLocationsShape = np.asarray(foodLocations).shape
        for i in range(foodLocationsShape[0]):
            for j in range(foodLocationsShape[1]):
                if foodLocations[i][j]:
                    foodDistance += manhattanDistance(newPos, (i,j))
                    foodCount += 1

        # print "food distance is %d" % foodDistance
        ghostDistance = 0
        for ghostState in newGhostStates:
            if not ghostState.scaredTimer > 0:
                ghostDistance += manhattanDistance(newPos, ghostState.getPosition())
            else:
                foodDistance += manhattanDistance(newPos, ghostState.getPosition())
                foodCount += 1
        if foodCount != 0:
            foodDistance = foodDistance / foodCount
        ghostDistance = ghostDistance/len(newGhostStates)
        finalReward = - foodDistance + ghostDistance + successorGameState.getScore()
        return finalReward

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        # print inspect.getmembers(gameState, predicate=inspect.ismethod)
        return self.wrapper(gameState, self.depth)

    def wrapper(self, gameState,max_depth,current_depth = 0):
        maxi = -999999
        if current_depth == max_depth or len(gameState.getLegalActions(0)) == 0:
            return self.evaluationFunction(gameState)
        # print gameState.getLegalActions(0)
        maxAction = None
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            tempMax = self.min_value(nextState, 1, current_depth, max_depth)
            if maxi < tempMax:
                maxAction = action
            maxi = max(maxi, tempMax)
        return maxAction

    def max_value(self, gameState, current_depth, max_depth):

        current_depth += 1
        maxi = -999999

        if current_depth == max_depth or len(gameState.getLegalActions(0)) == 0:
            return self.evaluationFunction(gameState)
        # print gameState.getLegalActions(0)
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)

            maxi = max(maxi, self.min_value(nextState, 1, current_depth, max_depth))
        return maxi

    def min_value(self, gameState, agentIndex, current_depth, max_depth):

        mini = +999999

        if current_depth == max_depth or len(gameState.getLegalActions(agentIndex)) == 0:
            return self.evaluationFunction(gameState)
        # print gameState.getLegalActions(agentIndex)
        for action in gameState.getLegalActions(agentIndex):

            if gameState.getNumAgents() == agentIndex + 1:

                mini = min(mini, self.max_value(gameState.generateSuccessor(agentIndex, action), current_depth, max_depth))
            else:

                mini = min(mini, self.min_value(gameState.generateSuccessor(agentIndex, action), agentIndex + 1,
                                                    current_depth, max_depth))

        return mini


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.wrapper(gameState, self.depth)

    def wrapper(self, gameState,max_depth,current_depth = 0):
        maxi = -999999
        alpha = -999999
        beta = 999999
        if current_depth == max_depth or len(gameState.getLegalActions(0)) == 0:
            return self.evaluationFunction(gameState)
        # print gameState.getLegalActions(0)
        maxAction = None
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            tempMax = self.min_value(nextState, alpha, beta ,1, current_depth, max_depth)
            if maxi < tempMax:
                maxAction = action
            maxi = max(maxi, tempMax)
            if maxi > beta: return maxi
            alpha = max(alpha, maxi)
        return maxAction

    def max_value(self, gameState, alpha, beta, current_depth, max_depth):

        current_depth += 1
        maxi = -999999

        if current_depth == max_depth or len(gameState.getLegalActions(0)) == 0:
            return self.evaluationFunction(gameState)
        # print gameState.getLegalActions(0)
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            maxi = max(maxi, self.min_value(nextState, alpha, beta, 1, current_depth, max_depth))
            if maxi > beta: return maxi
            alpha = max(alpha, maxi)
        return maxi

    def min_value(self, gameState, alpha, beta, agentIndex, current_depth, max_depth):

        mini = +999999
        if current_depth == max_depth or len(gameState.getLegalActions(agentIndex)) == 0:
            return self.evaluationFunction(gameState)
        # print gameState.getLegalActions(agentIndex)
        for action in gameState.getLegalActions(agentIndex):

            if gameState.getNumAgents() == agentIndex + 1:
                mini = min(mini, self.max_value(gameState.generateSuccessor(agentIndex, action), alpha, beta, current_depth, max_depth))
                if mini < alpha: return mini
                beta = min(beta, mini)
            else:
                mini = min(mini, self.min_value(gameState.generateSuccessor(agentIndex, action), alpha, beta, agentIndex + 1,
                                                    current_depth, max_depth))
                if mini < alpha: return mini
                beta = min(beta, mini)
        return mini

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.wrapper(gameState, self.depth)

    def wrapper(self, gameState, max_depth, current_depth=0):
        maxi = -999999
        if current_depth == max_depth or len(gameState.getLegalActions(0)) == 0:
            return self.evaluationFunction(gameState)
        # print gameState.getLegalActions(0)
        maxAction = None
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            tempMax = self.min_value(nextState, 1, current_depth, max_depth)
            if maxi < tempMax:
                maxAction = action
            maxi = max(maxi, tempMax)
        return maxAction

    def max_value(self, gameState, current_depth, max_depth):

        current_depth += 1
        maxi = -999999

        if current_depth == max_depth or len(gameState.getLegalActions(0)) == 0:
            return self.evaluationFunction(gameState)
        # print gameState.getLegalActions(0)
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            maxi = max(maxi, self.min_value(nextState, 1, current_depth, max_depth))
        return maxi

    def min_value(self, gameState, agentIndex, current_depth, max_depth):
        mini = +999999
        stateValue = 0
        if current_depth == max_depth or len(gameState.getLegalActions(agentIndex)) == 0:
            return self.evaluationFunction(gameState)
        # print gameState.getLegalActions(agentIndex)
        for action in gameState.getLegalActions(agentIndex):
            if gameState.getNumAgents() == agentIndex + 1:
                stateValue += (1.0/len(gameState.getLegalActions(agentIndex))) * self.max_value(gameState.generateSuccessor(agentIndex, action), current_depth, max_depth)
                # mini = min(mini,self.max_value(gameState.generateSuccessor(agentIndex, action), current_depth, max_depth))
            else:
                stateValue += (1.0/len(gameState.getLegalActions(agentIndex))) * self.min_value(gameState.generateSuccessor(agentIndex, action), agentIndex + 1,
                                                current_depth, max_depth)
                # mini = min(mini, self.min_value(gameState.generateSuccessor(agentIndex, action), agentIndex + 1,
                #                                 current_depth, max_depth))
        return stateValue


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    foodLocations = []
    for row in newFood:
        foodLocations.append(row)
    foodDistance = 0
    foodCount = 0
    foodLocationsShape = np.asarray(foodLocations).shape
    for i in range(foodLocationsShape[0]):
        for j in range(foodLocationsShape[1]):
            if foodLocations[i][j]:
                foodDistance += manhattanDistance(newPos, (i, j))
                foodCount += 1

    # print "food distance is %d" % foodDistance
    ghostDistance = 0
    for ghostState in newGhostStates:
        if not ghostState.scaredTimer > 0:
            ghostDistance += manhattanDistance(newPos, ghostState.getPosition())
        else:
            foodDistance += manhattanDistance(newPos, ghostState.getPosition())
            foodCount += 1
    if foodCount != 0:
        foodDistance = foodDistance / foodCount
    ghostDistance = ghostDistance / len(newGhostStates)
    finalReward = - foodDistance + 0.5*ghostDistance + currentGameState.getScore()
    return finalReward

# Abbreviation
better = betterEvaluationFunction

