from collections import OrderedDict

from rl_card.ofcgame_rlcard import OfcGameRLCard

from rlcard.envs import Env

DEFAULT_GAME_CONFIG = {
    'game_num_players': 2
}


class OfcEnv(Env):
    def __init__(self, config):
        ''' Initialize the Ofc environment
        '''
        self.name = 'ofc'
        self.default_game_config = DEFAULT_GAME_CONFIG
        self.game = OfcGameRLCard()
        super().__init__(config)
        self.state_shape = [[12, 4, 13] for _ in range(self.num_players)]
        self.action_shape = [[self.game.get_num_actions()] for _ in range(self.num_players)]

    def _get_legal_actions(self):
        ''' Get all legal actions for current state.

        Returns:
            (list): A list of legal actions' id.

        Note: Must be implemented in the child class.
        '''
        return self.game.get_legal_actions()

    def _extract_state(self, state):
        ''' Extract the state representation from state dictionary for agent

        Args:
            state (dict): Original state from the game

        Returns:
            observation (list)
        '''
        legal_actions = OrderedDict({a: None for a in state['legal_actions']})
        return {'obs': state['board_from_player'], 'legal_actions': legal_actions, 'raw_obs': state,
                'raw_legal_actions': state['legal_actions_str'], 'action_record': self.action_recorder}

    def get_payoffs(self):
        ''' Return the payoffs of the game

        Returns:
            (list): Each entry corresponds to the payoff of one player
        '''
        return self.game.get_payoffs()

    def _decode_action(self, action_id):
        ''' Decode the action for applying to the game

        Args:
            action id (int): action id

        Returns:
            action id also
        '''
        return action_id
