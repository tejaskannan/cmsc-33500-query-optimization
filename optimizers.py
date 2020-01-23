import numpy as np
from typing import Dict, Any, List, Tuple


class Optimizer:

    def __init__(self, queries: List[str]):
        self._queries = queries

    @property
    def queries(self) -> List[str]:
        return self._queries

    def update(self, query_index: int, reward: float):
        pass

    def get_query(self, time: int) -> Tuple[int, str]:
        raise NotImplementedError()


class ConstantOptimizer(Optimizer):

    def __init__(self, constant: int, queries: List[str]):
        super().__init__(queries)
        self._constant = constant

    @property
    def constant(self) -> int:
        return self._constant

    def get_query(self, time: int) -> Tuple[int, str]:
        return self.constant, self.queries[self.constant]


class RandomOptimizer(Optimizer):

    def get_query(self, time: int) -> Tuple[int, str]:
        index = np.random.randint(low=0, high=len(self.queries))
        return index, self.queries[index]


class EpsilonGreedyOptimizer(Optimizer):

    def __init__(self, epsilon: float, queries: List[str]):
        super().__init__(queries)
        self._epsilon = epsilon
        self._rewards = np.zeros(shape=len(queries))
        self._counts = np.zeros(shape=len(queries))

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def update(self, query_index: int, reward: float):
        self._rewards[query_index] += reward
        self._counts[query_index] += 1.0

    def get_query(self, time: int) -> Tuple[int, str]:
        if np.random.random() < self.epsilon:
            query_index = np.random.randint(0, len(self.queries))
        else:
            avg_rewards = self._rewards / (self._counts + 1e-7)
            query_index = np.argmax(avg_rewards)
        return query_index, self.queries[query_index]


class UCB1Optimizer(Optimizer):

    def __init__(self, queries: List[str]):
        super().__init__(queries)
        self._rewards = np.zeros(shape=len(queries))
        self._counts = np.zeros(shape=len(queries))

    def update(self, query_index: int, reward: float):
        self._rewards[query_index] += reward
        self._counts[query_index] += 1.0

    def get_query(self, time: int) -> Tuple[int, str]:
        upper_bounds = np.sqrt((2 * np.log(time)) / (self._counts + 1e-7))
        avg_rewards = self._rewards / (self._counts + 1e-7)
        adjusted_avg_rewards = avg_rewards + upper_bounds
        query_index = np.argmax(adjusted_avg_rewards)
        return query_index, self.queries[query_index]


class EXP3Optimizer(Optimizer):

    def __init__(self, gamma: float, queries: List[str]):
        super().__init__(queries)
        self._gamma = gamma
        self._weights = np.ones(shape=len(queries))
        self._k = len(queries)

    def _get_probs(self) -> np.array:
        return (1 - self._gamma) * (self._weights / np.sum(self._weights)) + (self._gamma / self._k)

    def update(self, query_index: int, reward: float):
        probs = self._get_probs()
        prob_i = probs[query_index]
        estimated_reward = reward / prob_i
        self._weights[query_index] *= np.exp(self._gamma * estimated_reward / self._k)

    def get_query(self, time: int) -> Tuple[int, str]:
        probs = self._get_probs()
        index = np.random.choice(a=self._k, size=1, replace=False, p=probs)[0]
        return index, self._queries[index]


def get_optimizer(name: str, queries: List[str], **kwargs: Dict[str, Any]):
    name_lower = name.lower()
    if name_lower == 'constant':
        return ConstantOptimizer(constant=kwargs['constant'], queries=queries)
    elif name_lower in ('epsilon', 'epsilon_greedy'):
        return EpsilonGreedyOptimizer(epsilon=kwargs['epsilon'], queries=queries)
    elif name_lower in ('ucb1', 'upper_confidence_bound'):
        return UCB1Optimizer(queries=queries)
    elif name_lower == 'random':
        return RandomOptimizer(queries=queries)
    elif name_lower == 'exp3':
        return EXP3Optimizer(gamma=kwargs['gamma'], queries=queries)
    raise ValueError(f'Unknown optimizer {name}.')
