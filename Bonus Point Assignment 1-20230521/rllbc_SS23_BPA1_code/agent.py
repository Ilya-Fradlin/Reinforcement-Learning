import abstract


# ABSTRACT CLASS FOR AGENTS

class Agent:

    def getAction(self, state):
        """
            For the given state, get the agent's chosen
            action.  The agent knows the legal actions
        """
        abstract

    def getValue(self, state):
        """
            Get the value of the state.
        """
        abstract

    def getQValue(self, state, action):
        """
            Get the q-value of the state action pair.
        """
        abstract

    def getPolicy(self, state):
        """
            Get the policy recommendation for the state.

            May or may not be the same as "getAction".
        """
        abstract

    def update(self, state, action, nextState, reward):
        """
            Update the internal state of a learning agent
            according to the (state, action, nextState)
            transistion and the given reward.
        """
        abstract

    def reset(self):
        """
            called to reset the agent at the beginning of an episode
        """
        pass
