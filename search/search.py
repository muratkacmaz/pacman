# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
from idlelib.autocomplete_w import LISTUPDATE_SEQUENCE
from xml.sax.xmlreader import AttributesImpl

import game
import searchAgents
import util
from game import Directions
from typing import List
from util import Stack

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    # Initialize the stack with the start state and an empty list of actions
    stack = Stack()
    stack.push((problem.getStartState(), []))
    visited = set()
    
    while not stack.isEmpty():
        # Pop the current state and the actions to reach it from the stack
        current_state, actions = stack.pop()

        # If the current state is the goal, return the actions
        if problem.isGoalState(current_state):
            return actions

        # If the current state has not been visited
        if current_state not in visited:
            # Mark the current state as visited
            # Get the successors of the current state
            for successor, action, step_cost in problem.getSuccessors(current_state):
                # Push the successor state and the updated actions to the stack
                stack.push((successor, actions + [action]))

    # Return an empty list if no solution is found
    return []

def breadthFirstSearch(problem: SearchProblem, startState = (-1,-1) ) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    """initilize the open and close LISt"""
    open_list = util.Queue()
    close_list = []
    temp_problem = problem
    goal_list = []
    
    if hasattr(temp_problem, 'corners'):
        goal_list = list(temp_problem.corners)
    elif hasattr(temp_problem, 'goal'):
        goal_list.append(temp_problem.goal)
    
    if(startState == (-1,-1)):
        open_list.push((temp_problem.getStartState(), []))
    else:
        open_list.push((startState, []))
    
    """while the open list is not empty"""
    while not open_list.isEmpty():
        """pop the state"""
        state, actions = open_list.pop()
        
        """if the state is the goal state"""
        if problem.isGoalState(state):
            if goal_list.__contains__(state):
                goal_list.remove(state)
            break
        
        """if the state is not in the close list"""
        if state not in close_list:
            """push the state to the close list"""
            close_list.append(state)
            
            """get the successors of the state"""
            successors = temp_problem.getSuccessors(state)
            
            """for each successor"""
            for successor in successors:
                """get the state, action and step cost"""
                state, action, step_cost = successor
                
                """push the successor to the open list"""
                open_list.push((state, actions + [action]))

    if goal_list == []:
        return actions
    
    if hasattr(temp_problem, 'corners'):
        temp_problem.corners = tuple(goal_list)
        
    return actions + breadthFirstSearch(temp_problem, state)

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None, goals = []) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    goals = []
    startState = problem.getStartState()
    if(type(problem) == searchAgents.PositionSearchProblem):
        goals.append(problem.goal)
        return aStarSearchGoals(problem, goals, startState, heuristic)
    elif(type(problem) == searchAgents.CornersProblem):
        goals = list(problem.corners)
        return aStarSearchGoals(problem, goals, startState, heuristic)
    elif(type(problem) == searchAgents.FoodSearchProblem):
        startState, food_positions = problem.start
        goals = convert_grid_to_coordinates(food_positions)
        return aStarSearchGoals(problem, goals, startState, heuristic)
    else:
        print("Invalid problem type, " + type(problem))

def aStarSearchGoals(problem: SearchProblem, goals, startState, heuristic=nullHeuristic) -> List[Directions]:
    """A* search for problems with 'goal' attribute."""
    open_list = util.PriorityQueue()
    close_list = []

    start_state = startState
    open_list.push((start_state, []), heuristic(start_state, problem, goals))

    while not open_list.isEmpty():
        state, actions = open_list.pop()

        if goals.__contains__(state):
            goals.remove(state)
            break

        if state not in close_list:
            close_list.append(state)
            successors = problem.getSuccessors(state)

            for successor in successors:
                next_state, action, step_cost = successor
                new_actions = actions + [action]
                cost = problem.getCostOfActions(new_actions) + heuristic(next_state, problem, goals)
                open_list.push((next_state, new_actions), cost)

    if goals == []:
        return actions
    else:
        return actions + aStarSearchGoals(problem, goals, state, heuristic)
    
def convert_grid_to_coordinates(grid):
    coordinates = []
    for row_index in range(grid.height):
        for col_index in range(grid.width):
            if grid[col_index][row_index] == True:
                coordinates.append((col_index, row_index))
    return coordinates

def convert_coordinates_to_grid(coordinates, width, height):
    grid = game.Grid(width, height)
    for coordinate in coordinates:
        col_index, row_index  = coordinate
        grid[col_index][row_index] = True
    return grid

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
