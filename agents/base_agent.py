class BaseAgent:
    def __init__(self):
        self.model = None
        self.env = None
        self.episodes = None
        self.max_actions_per_episode = None
        self.epsilon = 1
        self.min_epsilon = 0.01
        self.eps_decay = 0.005
        self.learning_rate = 0.8
        self.discount_factor = 0.95
        self.rewards = []
        self.test_eps = None
        self.test_rewards = []

    def train(self):
        assert False, "This method should be implemented by the base class."

        return self.model

    def test(self):
        assert False, "This method should be implemented by the base class."

        return self.test_rewards

    def take_action(self):
        assert False, "This method should be implemented by the base class."

        return None
