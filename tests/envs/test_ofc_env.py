import unittest

import rlcard
from rlcard.agents import RandomAgent


class TestOfcEnv(unittest.TestCase):
    def test_run(self):
        env = rlcard.make('ofc')
        env.set_agents([RandomAgent(env.num_actions) for _ in range(env.num_players)])
        trajectories, _ = env.run(is_training=False)
        self.assertEqual(len(trajectories), env.num_players)
        trajectories, _ = env.run(is_training=True)
        self.assertEqual(len(trajectories), env.num_players)
        for i in range(300):
            print(i)
            trajectories, _ = env.run(is_training=True)


if __name__ == '__main__':
    unittest.main()
