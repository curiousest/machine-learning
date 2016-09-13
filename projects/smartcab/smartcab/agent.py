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
        ['oncomingblocking', 'True', 'False'],
        ['True', 'False'],
        ['True', 'False'],
        ['green', 'red']
    )]

    Q = {i: 5 for i in itertools.product(ACTIONS, states)}

    def __init__(self, env, learning_rate, discount_factor):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.LEARNING_RATE = learning_rate
        self.DISCOUNT_FACTOR = discount_factor

        self.success_count = 0
        self.success_time = 0
        self.original_deadline = None
        self.errors_count = 0
        self.total_errors_count = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.state = self.get_current_state()
        self.estimated_best_next_value, self.next_action = self.best_next_action(self.state)
        self.original_deadline = None
        self.total_errors_count += self.errors_count
        # print "Errors: {}, Total: {}".format(self.errors_count, self.total_errors_count)
        self.errors_count = 0

    def best_next_action(self, state):
        '''
        Returns the next action and expected value of that action for the
        action with the largest expected reward.

        :return: (estimated_value, action) ex: (2.5, 'left')
        '''

        maximums = max(
            [(self.Q[(a, state)], a) for a in ACTIONS],
            # each item in the comprehension is a tuple - value, action
            key=lambda a: a[0]
        )

        if type(maximums) is list:
            # The max function can return a list of maximum values if they share
            # the same max value. In this case, return the first one.
            return maximums[0]
        else:
            return maximums

    def get_next_waypoint(self):
        next_waypoint = getattr(self, 'next_waypoint', None)
        if next_waypoint:
            return next_waypoint
        return self.planner.next_waypoint()

    def get_current_state(self):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)

        oncoming_blocking_left = inputs['oncoming'] in ['right', 'forward']
        if oncoming_blocking_left:
            oncoming = "oncomingblocking"
        elif inputs['oncoming'] is None:
            oncoming = "True"
        else:
            oncoming = "False"

        left = str(inputs['left'] is None)
        right = str(inputs['right'] is None)

        # 3 x 3 x 2 x 2 x 2 possibilities = 64 possible states
        state_inputs = [self.next_waypoint, oncoming, left, right, inputs['light']]
        return ','.join(state_inputs)

    def update(self, t):
        self.state = self.get_current_state()
        # select best next action, done during learn step
        action = self.next_action
        deadline = self.env.get_deadline(self)
        if not self.original_deadline:
            self.original_deadline = deadline

        # Execute action and get reward
        reward = self.env.act(self, action)
        if reward > 2:
            self.success_count += 1
            self.success_time += self.original_deadline - deadline
        elif reward < 0:
            self.errors_count += 1
            #print "Errored in state: {} action: {}".format(self.state, action)

        prev_state = self.state
        next_state = self.get_current_state()
        last_est = self.estimated_best_next_value
        self.estimated_best_next_value, self.next_action = self.best_next_action(next_state)
        self.learn(
            prev_state, next_state, action, reward
        )

        # Debug prints
        #print "LearningAgent.update(): deadline = {}, state = {}, action = {}, reward = {}, est. reward = {}".format(deadline, prev_state, action, reward, last_est)  # [debug]

    def learn(self, prev_state, curr_state, action, reward):
        old_value = self.Q[(action, prev_state)]
        learned_value = reward + (self.DISCOUNT_FACTOR * self.estimated_best_next_value)
        self.Q[(action, prev_state)] = old_value + self.LEARNING_RATE * (learned_value - old_value)


def run():
    """Run the agent for a finite number of trials."""

    trials = []

    for i in range(0, 110, 10):
        # values 0.0, 0.1, 0.2, ... , 1.0
        learning_rate = i / 100.0
        for j in range(0, 110, 10):
            discount_factor = j / 100.0
            # Set up environment and agent
            e = Environment()  # create environment (also adds some dummy traffic)
            a = e.create_agent(LearningAgent, learning_rate, discount_factor)  # create agent
            e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
            # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

            # Now simulate it
            sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
            # NOTE: To speed up simulation, reduce update_delay and/or set display=False

            sim.run(n_trials=100)  # run for a specified number of trials
            # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

            trials.append((
                learning_rate,
                discount_factor,
                a.success_count,
                a.success_time / max(a.success_count, 1)
            ))


    print "\n=== RESULTS ===\n"

    for trial in trials:
        print "Learning Rate: {}, Discount Factor: {}, # successes: {}/100, average success time: {}".format(
            *[t for t in trial]
        )

    print trials


if __name__ == '__main__':
    run()

