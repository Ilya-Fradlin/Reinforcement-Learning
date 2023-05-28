import numpy as np
from agent import Agent

# TASK 1

class PolicyIterationAgent(Agent):

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your policy iteration agent take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        states = self.mdp.getStates()
        number_states = len(states)
        # Policy initialization
        # ******************
        # TODO 1.1.a)
        # self.V = ...
        self.V = {key: 0 for key in states}
        # *******************

        self.pi = {s: self.mdp.getPossibleActions(s)[-1] if self.mdp.getPossibleActions(s) else None for s in states}

        counter = 0

        while True:
            # Policy evaluation
            for i in range(iterations):
                newV = {}
                for s in states:
                    a = self.pi[s]
                    # *****************
                    # TODO 1.1.b)
                    # if...
                    if self.mdp.isTerminal(s):
                        newV[s] = 0.0
                    # else:...
                    else:
                        transition_states_and_probs = self.mdp.getTransitionStatesAndProbs(s, a)
                        reward = self.mdp.getReward(s, a, None)
                        
                        temp = 0
                        for j in transition_states_and_probs:

                            temp += j[1]*(reward + self.discount*self.V[j[0]])
                            
                        newV[s] = temp  
                # update value estimate
                # self.V=...
                self.V = newV
                
                checked = False
                if self.V[mdp.getStartState()] != 0.0 and not checked:
                    checked = True
                    print('Rounds of policy evaluation: ',10*counter + i)
                # ******************

            policy_stable = True
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                if len(actions) < 1:
                    self.pi[s] = None
                else:
                    old_action = self.pi[s]
                    # ************
                    # TODO 1.1.c)
                    # self.pi[s] = ...
                    
                    max_value_action = []
                    for action in actions:
                        transition_states_and_probs = self.mdp.getTransitionStatesAndProbs(s, action)
                        reward = self.mdp.getReward(s, action, None)
                        
                        temp = 0
                        for l in transition_states_and_probs:
                            temp += l[1]*(reward + self.discount*self.V[l[0]])
                        
                        max_value_action.append(temp)
                        
                    new_action = actions[np.argmax(max_value_action)]
                    self.pi[s] = new_action
                       
                    # policy_stable =

                    if old_action != self.pi[s]:
                        policy_stable = False
                    # ****************
            counter += 1

            if policy_stable: break

        print("Policy converged after %i iterations of policy iteration" % counter)

    def getValue(self, state):
        """
        Look up the value of the state (after the policy converged).
        """
        # *******
        # TODO 1.2.
        return self.V[state]
        # ********

    def getQValue(self, state, action):
        """
        Look up the q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that policy iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        # *********
        # TODO 1.3.
        transition_states_and_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        reward = self.mdp.getReward(state, action, None)
        
        q_value = 0
        for i in transition_states_and_probs:                
            q_value +=  i[1]*(reward + self.discount*self.V[i[0]])
          
        return q_value
        # **********

    def getPolicy(self, state):
        """
        Look up the policy's recommendation for the state
        (after the indicated number of value iteration passes).
        """
        # **********
        # TODO 1.4.
        return self.pi[state]
        # **********

    def getAction(self, state):
        """
        Return the action recommended by the policy.
        """
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Not used for policy iteration agents!
        """

        pass
