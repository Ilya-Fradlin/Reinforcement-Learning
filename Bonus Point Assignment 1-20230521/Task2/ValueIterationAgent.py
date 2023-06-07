from agent import Agent
import numpy as np

# TASK 2
class ValueIterationAgent(Agent):

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your value iteration agent take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        states = self.mdp.getStates()
        number_states = len(states)
        # *************
        #  TODO 2.1 a)
        self.V = {key: 0 for key in states}
        self.best_a = {s: self.mdp.getPossibleActions(s)[-1] if self.mdp.getPossibleActions(s) else None for s in states}

        # ************

        for i in range(iterations):
            print(f'Iteration number: {i}')
            newV = {}
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                # **************
                # TODO 2.1. b)
                if len(actions) < 1:
                    self.best_a[s] = None
                    newV[s] = self.V[s]

                else:
                    max_value = float('-inf')
                    best_a = None
                    for a in actions:
                        if self.mdp.isTerminal(s):
                            newV[s] = 0.0

                        else:
                            transition_states_and_probs = self.mdp.getTransitionStatesAndProbs(s, a)
                            reward = self.mdp.getReward(s, a, None)
                            temp = 0
                            for j in transition_states_and_probs:
                                temp += j[1] * (reward + self.discount * self.V[j[0]])

                            if temp >= max_value:
                                max_value = temp
                                best_a = a
                                newV[s] = max_value

                    self.best_a[s] = best_a
                    newV[s] = max_value

            # Update value function with new estimate
            self.V = newV
            # ***************

    def getValue(self, state):
        """
        Look up the value of the state (after the indicated
        number of value iteration passes).
        """
        # **********
        # TODO 2.2
        return self.V[state]
        # **********

    def getQValue(self, state, action):
        """
        Look up the q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that value iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        # ***********
        # TODO 2.3.
        transition_states_and_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        reward = self.mdp.getReward(state, action, None)

        q_value = 0
        for i in transition_states_and_probs:
            q_value += i[1] * (reward + self.discount * self.V[i[0]])

        return q_value
        # **********

    def getPolicy(self, state):
        """
        Look up the policy's recommendation for the state
        (after the indicated number of value iteration passes).
        """

        actions = self.mdp.getPossibleActions(state)
        if len(actions) < 1:
            return None

        else:

        # **********
        # TODO 2.4
            return self.best_a[state]
        # ***********

    def getAction(self, state):
        """
        Return the action recommended by the policy.
        """
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Not used for value iteration agents!
        """

        pass
