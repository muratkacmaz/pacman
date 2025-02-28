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

import util
from game import Directions
from typing import List
from util import Stack
from util import Queue
from util import PriorityQueue

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
    """

    print("Problem Type:", type(problem))
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))


    # Stack initialization with empty list and start state
    stack = Stack()
    start_state = problem.getStartState()
    stack.push((start_state, []))  
    
    # Holds the visited states
    visited = set()

    while not stack.isEmpty():
        current_state, path = stack.pop()

        # If the state is already visited, skip processing
        if current_state in visited:
            continue
        
        # Append visited list with the current state
        visited.add(current_state)

        # Check if goal is reached
        if problem.isGoalState(current_state):
            return path

        # Expand the node and add its successors
        for successor, action, cost in problem.getSuccessors(current_state):
            if successor not in visited:
                stack.push((successor, path + [action]))

    return []  # If no solution is found

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""

    # Queue initialization with empty list and start state
    queue = Queue()
    start_state = problem.getStartState()
    queue.push((start_state, []))
    
    # Holds the visited states
    visited = set()
    
    while not queue.isEmpty():
        current_state, path = queue.pop()

        # If the state is already visited, skip processing
        if current_state in visited:
            continue

        # Append visited list with the current state
        visited.add(current_state)

        # Check if goal is reached
        if problem.isGoalState(current_state):
            return path
        
        for successor, action, cost in problem.getSuccessors(current_state):
            if successor not in visited:
                queue.push((successor, path + [action]))


def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    
    priorityQueue = PriorityQueue()
    start_state = problem.getStartState()
    priorityQueue.push((start_state, [], 0), 0)
    
    visited = set()
    
    while not priorityQueue.isEmpty():
        state, path, cost = priorityQueue.pop()
        
        if state in visited:
            continue
            
        visited.add(state)
        
        if problem.isGoalState(state):
            return path
        
        for successor, action, step_cost in problem.getSuccessors(state):
            if successor not in visited:
                priorityQueue.push((successor, path + [action], cost + step_cost), cost + step_cost)
    return []

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # Priority queue stores (total_cost, state, path, g_cost)
    priorityQueue = PriorityQueue()
    start_state = problem.getStartState()
    priorityQueue.push((start_state, [], 0), 0)  # (state, path, cost), priority=f(n)

    # **Tracking visited states and their best g(n) cost**
    visited = {}

    while not priorityQueue.isEmpty():
        state, path, g_cost = priorityQueue.pop()

        # **If goal reached, return the path**
        if problem.isGoalState(state):
            return path

        # **If this state was visited with a lower cost before, skip it**
        if state in visited and visited[state] <= g_cost:
            continue
        
        # **Mark the state with the lowest found cost**
        visited[state] = g_cost

        # **Expand the current node**
        for successor, action, cost in problem.getSuccessors(state):
            new_g_cost = g_cost + cost  # g(n) = current cost + step cost
            f_cost = new_g_cost + heuristic(successor, problem)  # f(n) = g(n) + h(n)

            # **Push into priority queue only if it's a new or better path**
            if successor not in visited or visited[successor] > new_g_cost:
                priorityQueue.push((successor, path + [action], new_g_cost), f_cost)

    return []  # No solution found

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
