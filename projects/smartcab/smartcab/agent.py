import itertools
import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

ACTIONS = ['left', 'right', 'forward', None]

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    next_waypoint_states = ['right', 'forward', 'left']

    # every possible state (list of tuples)
    states = [','.join(i) for i in itertools.product(
        next_waypoint_states,
        ['oncomingblockingleft', 'True', 'False'],
        ['True', 'False'],
        ['True', 'False'],
        ['green', 'red']
    )]

    Q = {i: 12 for i in itertools.product(ACTIONS, states)}
    LEARNING_RATE = 0.2
    DISCOUNT_FACTOR = 0.5

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.state = self.get_current_state()
        self.estimated_best_next_value, self.next_action = self.best_next_action(self.state)

    def best_next_action(self, state):
        '''
        Returns the next action and expected value of that action for the
        action with the largest expected reward.

        To estimate future states, we have to consider that taking some action
        will put us in one of 4 random states. Average the 3 random states
        for each action and pick the max average.

        :return: (estimated_value, action) ex: (2.5, 'left')
        '''

        for remove_waypoint in self.next_waypoint_states:
            state = state.replace(remove_waypoint, '')

        possible_states = [i + state for i in self.next_waypoint_states]
        maximums = max(
            [(sum(
                [self.Q[(a, s)] for s in possible_states]
            ) / len(self.next_waypoint_states), a)
             for a in ACTIONS],
            # each item in the comprehension is a tuple - value, action
            key=lambda a: a[0]
        )

        if type(maximums) is list:
            # The max function can return a list of maximum values if they share
            # the same max value. In this case, return the first one.
            return maximums[0]
        else:
            return maximums

    def get_current_state(self):
        # Gather inputs
        next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        oncoming_blocking_left = inputs['oncoming'] in ['right', 'forward']
        if oncoming_blocking_left:
            oncoming = "oncomingblockingleft"
        elif inputs['oncoming']:
            oncoming = "True"
        else:
            oncoming = "False"

        left = str(inputs['left'] is None)
        right = str(inputs['right'] is None)

        # 3 x 3 x 2 x 2 x 2 possibilities = 64 possible states
        state_inputs = [next_waypoint, oncoming, left, right, inputs['light']]
        return ','.join(state_inputs)

    def update(self, t):

        # select best next action, done during learn step
        action = self.next_action

        # Execute action and get reward
        reward = self.env.act(self, action)

        prev_state = self.state
        self.state = self.get_current_state()

        self.learn(prev_state, self.state, action, reward)

        # Debug prints
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        print "LearningAgent.update(): deadline = {}, state = {}, action = {}, reward = {}, est = {}".format(deadline, prev_state, action, reward, self.estimated_best_next_value)  # [debug]

    def learn(self, prev_state, curr_state, action, reward):
        old_value = self.Q[(action, prev_state)]
        self.estimated_best_next_value, self.next_action = self.best_next_action(self.state)
        learned_value = reward + (self.DISCOUNT_FACTOR * self.estimated_best_next_value)
        self.Q[(action, prev_state)] = old_value + self.LEARNING_RATE * (learned_value - old_value)


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()


# if inputs['light'] == "red":
#     if inputs['right'] == 'straight':
#         self.state = self.RED_CANT_RIGHT
#     else:
#         self.state = self.RED_CAN_RIGHT
# else:
#     if inputs['oncoming'] in ['right', 'straight']:
#         self.state = self.GREEN_CANT_LEFT
#     else:
#         self.state = self.GREEN_CAN_LEFT