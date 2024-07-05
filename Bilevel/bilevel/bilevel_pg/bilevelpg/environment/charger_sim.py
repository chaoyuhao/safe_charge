import numpy as np
from malib.bilevel_pg.bilevelpg.spaces import Discrete, Box, MASpace, MAEnvSpec

from sca3_game import sca3_step, query_max_k, query_chargers

from bilevel_pg.bilevelpg.environment.base_game import  BaseGame
from bilevel_pg.bilevelpg.error import EnvironmentNotFound, WrongNumberOfAgent, WrongNumberOfAction, WrongActionInputLength

class ChargerSim(BaseGame):

    def __init__(self, game_name, agent_num, action_num, payoff=None, repeated=False, memory=0, discrete_action=True, tuple_obs=False):
        self.game_name = game_name
        self.agent_num = agent_num
        self.action_num = action_num
        self.action_range = (0, 5)
        self.discrete_action = discrete_action # True
        self.tuple_obs = tuple_obs
        self.num_state = 1

        
        self.action_spaces = MASpace(tuple(Discrete(action_num) for _ in range(self.agent_num)))
        if memory == 0:
            self.observation_spaces = MASpace(tuple(Discrete(1) for _ in range(self.agent_num)))
        else:
            self.observation_spaces = MASpace(tuple(Discrete(5) for _ in range(self.agent_num)))
        # more operations of observation spaces


        self.env_specs = MAEnvSpec(self.observation_spaces, self.action_spaces)


        self.t = 0
        self.repeated = repeated
        self.max_step = query_max_k()
        self.memory = memory
        self.previous_action = 0
        self.previous_actions = []
        self.ep_rewards = np.zeros(2)

        if payoff is not None:
            payoff = np.array(payoff)
            assert payoff.shape == tuple([agent_num] + [action_num] * agent_num)
            self.payoff = payoff
        if payoff is None:
            self.payoff = np.zeros(tuple([agent_num] + [action_num] * agent_num))

    # TODO:
    def get_rewards(self, actions):
        # 最小开关值 当actions[i] >= asm 的时候就打开当前充电器
        args_switchon_min = 0.5
        # 拉格朗日松弛所用的参数
        args_lambda = 0.3
        # 最大辐射约束
        args_max_rad = 400.0
        
        reward_n = np.zeros((self.agent_num,))
        actions_bool = []
        if self.discrete_action:
            for i in range(self.agent_num):
                for j in range(self.action_range):
                    assert actions[i][j] in range(self.action_num)
                    if actions[i] >= args_switchon_min:
                        actions_bool.append(True)
                    else: actions_bool.append(False)
        
        penalty, efficiency = sca3_step(actions_bool)
        
        reward0 = efficiency - penalty / args_max_rad * args_lambda

        # no else
        return reward_n
        
    # TODO:
    def step(self, actions):
        if len(actions) != self.agent_num:
            raise WrongActionInputLength(f"Expected number of actions is {self.agent_num}")

        actions = np.array(actions).reshape((self.agent_num, self.action_range[0]))
        # actions = np.array(actions).reshape((self.agent_num,))
        reward_n = self.get_rewards(actions)
        self.rewards = reward_n
        info = {}
        done_n = np.array([True] * self.agent_num)
        if self.repeated:
            done_n = np.array([False] * self.agent_num)
        self.t += 1
        if self.t >= self.max_step: # max_k
            done_n = np.array([True] * self.agent_num)

        state = [0] * (self.action_num * self.agent_num * (self.memory) + 1)
        # state_n = [tuple(state) for _ in range(self.agent_num)]
        if self.memory > 0 and self.t > 0:
            # print('actions', actions)
            if self.discrete_action:
                state[actions[1] + 2 * actions[0] + 1] = 1
            else:
                state = actions

        # tuple for tublar case, which need a hashabe obersrvation
        if self.tuple_obs:
            state_n = [tuple(state) for _ in range(self.agent_num)]
        else:
            state_n = np.array([state for _ in range(self.agent_num)])

        # for i in range(self.agent_num):
        #     state_n[i] = tuple(state_n[i][:])

        self.previous_actions.append(tuple(actions))
        self.ep_rewards += np.array(reward_n)

        # print(state_n, reward_n, done_n, info)
        return state_n, reward_n, done_n, info

    
    def reset(self):
        self.ep_rewards = np.zeros(2)
        self.t = 0
        self.previous_action = 0
        self.previous_actions = []
        state = [0] * (self.action_num * self.agent_num * (self.memory)  + 1)
        if self.memory > 0:
            state = [0., 0.]
        if self.tuple_obs:
            state_n = [tuple(state) for _ in range(self.agent_num)]
        else:
            state_n = np.array([state for _ in range(self.agent_num)])
        return state_n
    
    @staticmethod
    def get_game_list():
        return {
            'charger' : {'agent_num' : 2, 'action_num' : 2}
        }
    
    def render():
        pass

    def terminate(self):
        pass

    def get_joint_reward(self):
        return self.rewards
    
if __name__ == '__main__':
    print(ChargerSim.get_game_list())
    game = ChargerSim('charger', agent_num=2, action_num=2)
    print(game)