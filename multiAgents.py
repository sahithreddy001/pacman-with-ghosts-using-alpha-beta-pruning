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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # print(newScaredTimes)

        "*** YOUR CODE HERE ***"

        # initializing distances with infinite to compare it with every minimum distance and then update it
        food_distance_min = float("inf")
        ghost_distance_min = float("inf")

        # calculating the minimum food distance from the new position and updating food_distance_min for every iteration
        for i in newFood:
            food_distance_min = min(food_distance_min, manhattanDistance(newPos, i))

        # getting ghost positions
        ghost_positions = successorGameState.getGhostPositions()

        # for every ghost we are calculating its distance to the pacman's new position and
        # if it is less than 2 (which means ghost is next to pacman) return negative infinte so that
        # the pacman knows not to get to that new position
        for i in ghost_positions:
            ghost_distance_min = min(ghost_distance_min, manhattanDistance(newPos, i))
        if ghost_distance_min < 2:
            return -float("inf")

        # returning reciprocal of food distance because, the lower the food distance to that new position,
        # the maximum the value for that new position must be
        return successorGameState.getScore() + (1.0 / food_distance_min)


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

        # code for how maximizer node works
        def maximizer(state, depth, index_of_agent):
            depth = depth - 1
            # condition for termination of recursive method calls
            if depth < 0 or state.isLose() or state.isWin():
                return (self.evaluationFunction(state), None)
            # initialise `value` with negative infinite so that we can update for any maximum value we encounter
            value = -float("inf")
            # for every legal action
            for i in state.getLegalActions(index_of_agent):
                # getting the successor for every legal action in respective iteration
                successor = state.generateSuccessor(index_of_agent, i)
                # as the next state for a maximizer is minimizer we call for minimizer and take its value as score
                score = minimizer(successor, depth, index_of_agent + 1)[0]
                # for every legal action get the score from every corresponding minimizer i.e., ghosts,
                # and update the maximum value of score to the `value`
                # because the maximizer only looks for the maximum value from the minimizer
                if score > value:
                    value = score
                    maxiAction = i
            return (value, maxiAction)

        # code for how minimizer node works
        def minimizer(state, depth, index_of_agent):
            # condition for termination of recursive method calls
            if depth < 0 or state.isLose() or state.isWin():
                return (self.evaluationFunction(state), None)
            # initialise `value` with infinite so that we can update for any minimum value we encounter
            value = float("inf")
            # till we iterate through every ghost we will call evaluation_function as minimizer
            # after every ghost is iterated,
            # the work of minimzer is complete and it handles the function to maximizer to visit for next depth
            # hence, we set evaluation_function as maximizer in else condition
            if index_of_agent < state.getNumAgents() - 1:
                evaluation_function, next_agent = (minimizer, index_of_agent + 1)
            else:
                evaluation_function, next_agent = (maximizer, 0)

            for i in state.getLegalActions(index_of_agent):
                # getting the successor for every legal action in respective iteration
                successor = state.generateSuccessor(index_of_agent, i)
                # value from evaluation_function is stored in score
                score = evaluation_function(successor, depth, next_agent)[0]
                # for every legal action get the score from every corresponding evaluation_function,
                # and update the mimimum value of score to the `value`
                # because the minimizer only looks for the minimum value from the evaluation_function
                if score < value:
                    value = score
                    miniAction = i
            return (value, miniAction)

        # returning the better action as maximizer returns (value, action) as a tuple
        # hence maximizer(gameState, self.depth, 0)[1] returns action
        return maximizer(gameState, self.depth, 0)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def maximizer(state, depth, index_of_agent, alpha, beta):
            depth = depth - 1
            if depth < 0 or state.isLose() or state.isWin():
                return (self.evaluationFunction(state), None)
            value = -float("inf")
            for i in state.getLegalActions(index_of_agent):
                successor = state.generateSuccessor(index_of_agent, i)
                score = minimizer(successor, depth, index_of_agent + 1, alpha, beta)[0]
                if score > value:
                    value = score
                    maxiAction = i
                # condition for pruning in maximizer
                if value > beta:
                    return (value, maxiAction)
                alpha = max(alpha, value)
            return (value, maxiAction)

        def minimizer(state, depth, index_of_agent, alpha, beta):
            if depth < 0 or state.isLose() or state.isWin():
                return (self.evaluationFunction(state), None)
            value = float("inf")
            if index_of_agent < state.getNumAgents() - 1:
                evaluation_function, next_agent = (minimizer, index_of_agent + 1)
            else:
                evaluation_function, next_agent = (maximizer, 0)
            for i in state.getLegalActions(index_of_agent):
                successor = state.generateSuccessor(index_of_agent, i)
                score = evaluation_function(successor, depth, next_agent, alpha, beta)[0]
                if score < value:
                    value = score
                    miniAction = i
                # condition for pruning in minimizer
                if value < alpha:
                    return (value, miniAction)
                beta = min(beta, value)
            return (value, miniAction)

        # everything in the code is same from class MinimaxAgent except for the pruning code snippet
        # alpha and beta are extra arguments added to the methods
        return maximizer(gameState, self.depth, 0, -float("inf"), float("inf"))[1]


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
        number_of_agents = gameState.getNumAgents()
        agentindex_count = self.depth * number_of_agents
        self.expecti(gameState, agentindex_count, number_of_agents)
        # returning the action which causes the maximum value
        return self.maximum_action

    def expecti(self, gameState, agentindex_count, number_of_agents):
        # initialising a list to store the values of maximizer nodes and expectimax nodes respectively
        maxlist = []
        expectedlist = []
        # condition for termination of recursive method calls
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        # proceed only if there is at least 1 ghost
        if agentindex_count > 0:
            # initialising index of ghosts so that we can track how many ghosts have been iterated.
            # if every ghost at a particular depth is iterated then the index is returned to zero
            if agentindex_count % number_of_agents == 0:
                index_of_agent = 0
            else:
                index_of_agent = number_of_agents - (agentindex_count % number_of_agents)
            legal_actions = gameState.getLegalActions(index_of_agent)
            # for every legal action
            for i in legal_actions:
                # get successor state
                successor_state = gameState.generateSuccessor(index_of_agent, i)
                # if index_of_agent is zero then we are computing for maximizer,
                # then maximum value from expecti method is to be taken as we are calaculating for the maximizer node
                if index_of_agent == 0:
                    maxlist.append((self.expecti(successor_state, agentindex_count - 1, number_of_agents), i))
                    maxi = max(maxlist)
                    self.maximum_value = maxi[0]
                    self.maximum_action = maxi[1]
                # if index_of_agent is not zero, that means we are calculating for ghosts from expectimax nodes
                # hence we take average value from expecti method
                else:
                    expectedlist.append((self.expecti(successor_state, agentindex_count - 1, number_of_agents), i))
                    average = 0
                    for i in range(len(expectedlist)):
                        average = average + expectedlist[i][0]
                    average = average / (len(expectedlist))
                    self.average_value = average
            # if-else snippet which will return maximum value or average value depending on the index_of_agent value
            # as explained earlier in comments
            if index_of_agent == 0:
                return self.maximum_value
            else:
                return self.average_value
        # the below else block execute when there are no ghosts i.e., when agentindex_count <= 0
        else:
            return self.evaluationFunction(gameState)

        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    "*** YOUR CODE HERE ***"
    new_position = currentGameState.getPacmanPosition()
    new_food = currentGameState.getFood().asList()

    food_distance_min = float('inf')
    for food in new_food:
        food_distance_min = min(food_distance_min, manhattanDistance(new_position, food))

    ghost_distance = 0
    ghost_positions = currentGameState.getGhostPositions()
    # newScaredTimes = [ghostState.scaredTimer for ghostState in ghost_positions]

    for i in ghost_positions:
        ghost_distance = manhattanDistance(new_position, i)
        if (ghost_distance < 2):
            return -float('inf')

    food = currentGameState.getNumFood()
    pellet = len(currentGameState.getCapsules())

    food_coefficient = 999999
    pellet_coefficient = 19999
    food_distance_coefficient = 999

    game_rewards = 0
    if currentGameState.isLose():
        game_rewards = game_rewards - 99999
    elif currentGameState.isWin():
        game_rewards = game_rewards + 99999

    answer = (1.0 / (food + 1) * food_coefficient) + ghost_distance + (
            1.0 / (food_distance_min + 1) * food_distance_coefficient) + (
                     1.0 / (pellet + 1) * pellet_coefficient) + game_rewards

    # code written here is pretty much similar to the code from evaluationfunction in q1.
    # To optimize the performance of the pacman we introduced coefficients
    # which will determine the weights of food, pellet and when the pacman wins
    # these coefficients are randomly tweaked according to a priority
    # food_coefficient>pellet_coefficient, to get the optimal results
    return answer


# Abbreviation
better = betterEvaluationFunction
