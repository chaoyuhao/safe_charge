# Created by yingwen at 2019-03-15
import tensorflow as tf
import numpy as np
# from malib.agents.base_agent import OffPolicyAgent
# from malib.core import Serializable
# from malib.utils import tf_utils

from bilevel_pg.bilevelpg.agents.base_agents import OffPolicyAgent
from bilevel_pg.bilevelpg.core import Serializable
from bilevel_pg.bilevelpg.utils import tf_utils

class FollowerDDPGAgent(OffPolicyAgent):
    def __init__(self,
                 env_specs,
                 policy,
                 qf,
                 replay_buffer,
                 policy_optimizer=tf.optimizers.Adam(lr=0.1),
                 qf_optimizer=tf.optimizers.Adam(lr=0.1),
                 exploration_strategy=None,
                 exploration_interval=10,
                 target_update_tau=0.1,
                 target_update_period=10,
                 td_errors_loss_fn=None,
                 gamma=0.75,
                 reward_scale=1.0,
                 gradient_clipping=None,
                 train_sequence_length=None,
                 name='Bilevel_follower',
                 agent_id=-1
                 ):
        self._Serializable__initialize(locals())
        self._agent_id = agent_id
        self._env_specs = env_specs
        if self._agent_id >= 0:
            observation_space = self._env_specs.observation_space[self._agent_id]
            action_space = self._env_specs.action_space[self._agent_id]
        else:
            observation_space = self._env_specs.observation_space
            action_space = self._env_specs.action_space

        # self._exploration_strategy = exploration_strategy

        self._exploration_strategy = None

        self._target_policy = Serializable.clone(policy, name='target_policy_agent_{}'.format(self._agent_id))
        self._target_qf = Serializable.clone(qf, name='target_qf_agent_{}'.format(self._agent_id))

        self._policy_optimizer = policy_optimizer
        self._qf_optimizer = qf_optimizer

        self._target_update_tau = target_update_tau
        self._target_update_period = target_update_period
        self._td_errors_loss_fn = (
                td_errors_loss_fn or tf.losses.Huber)
        self._gamma = gamma
        self._reward_scale = reward_scale
        self._gradient_clipping = gradient_clipping
        self._train_step = 0
        self._exploration_interval = exploration_interval
        self._exploration_status = False

        self.required_experiences = ['observation', 'actions', 'rewards', 'next_observations',
                                     'opponent_actions', 'target_actions']

        super(FollowerDDPGAgent, self).__init__(
            observation_space,
            action_space,
            policy,
            qf,
            replay_buffer,
            train_sequence_length=train_sequence_length,
            name=name,
        )

    def act(self, observation, step=None, use_target=False):
        if use_target and self._target_policy is not None:
            return self._target_policy.get_actions_np(observation)
        # if self._exploration_strategy is not None and self._exploration_status:
        #     if step is None:
        #         step = self._train_step
        #     if step % self._exploration_interval == 0:
        #         self._exploration_strategy.reset()
        #     return self._exploration_strategy.get_action(self._train_step, observation, self._policy)
        policy = self._policy

        # observe_with_opponent = tf.concat((observation, opponent_action), 1)
        return policy.get_actions_np(observation)

    def get_policy_np(self,
                         input_tensor):
        return self._policy.get_policy_np(input_tensor)

    def init_opt(self):
        tf_utils.soft_variables_update(
            self._policy.trainable_variables,
            self._target_policy.trainable_variables,
            tau=1.0)
        tf_utils.soft_variables_update(
            self._qf.trainable_variables,
            self._target_qf.trainable_variables,
            tau=1.0)
        self._exploration_status = True

    def init_eval(self):
        self._exploration_status = False

    def get_policy(self, use_target=False):
        if use_target:
            return self._target_policy
        else:
            return self._policy

    def _update_target(self):
        tf_utils.soft_variables_update(
            self._policy.trainable_variables,
            self._target_policy.trainable_variables,
            tau=self._target_update_tau)
        tf_utils.soft_variables_update(
            self._qf.trainable_variables,
            self._target_qf.trainable_variables,
            tau=self._target_update_tau)

    def _train(self, batch, weights=None):
        critic_variables = self._qf.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert critic_variables, 'No qf variables to optimize.'
            tape.watch(critic_variables)
            critic_loss = self.critic_loss(batch['observations'],
                                           batch['actions'],
                                           batch['opponent_actions'],
                                           batch['target_actions'],
                                           batch['rewards'],
                                           batch['next_observations'],
                                           weights=weights)
        tf.debugging.check_numerics(critic_loss, 'qf loss is inf or nan.')
        critic_grads = tape.gradient(critic_loss, critic_variables)
        tf_utils.apply_gradients(critic_grads, critic_variables, self._qf_optimizer, self._gradient_clipping)

        actor_variables = self._policy.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert actor_variables, 'No actor variables to optimize.'
            tape.watch(actor_variables)
            actor_loss = self.actor_loss(batch['observations'], batch['opponent_actions'], weights=weights)
        tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
        actor_grads = tape.gradient(actor_loss, actor_variables)
        tf_utils.apply_gradients(actor_grads, actor_variables, self._policy_optimizer, self._gradient_clipping)
        self._train_step += 1

        if self._train_step % self._target_update_period == 0:
            self._update_target()

        losses = {
            'pg_loss': actor_loss.numpy(),
            'critic_loss': critic_loss.numpy(),
        }

        return losses

    def get_critic_value(self,
                         input_tensor):
        return self._qf.get_values(input_tensor)

    def critic_loss(self,
                    observations,
                    actions,
                    opponent_actions,
                    target_actions,
                    rewards,
                    next_observations,
                    weights=None):
        """Computes the critic loss for DDPG training.
        Args:
          observations: A batch of observations.
          actions: A batch of actions.
          rewards: A batch of rewards.
          next_observations: A batch of next observations.
          weights: Optional scalar or element-wise (per-batch-entry) importance
            weights.
        Returns:
          critic_loss: A scalar critic loss.
        """
        # target_actions = self._target_policy.get_actions(next_observations)
        # target_critic_input = [next_observations.astype(np.float32), tf.cast(target_actions, tf.float32)]
        # target_critic_input = np.hstack(
        #     (next_observations.astype(np.float32)[:, 0].reshape(-1, 1), target_actions))
        # print(next_observations[0])
        target_critic_input = (next_observations[:,0:1],
                                         target_actions)
        # print(next_observations.shape)
        # print(target_actions.shape)
        # print(target_actions.shape)
        target_q_values = self._target_qf.get_values(target_critic_input)

        print('reward shape:')
        print(rewards.shape)

        rewards = rewards.reshape(-1, 1)
        td_targets = tf.stop_gradient(
                self._reward_scale * rewards + self._gamma * target_q_values)

        # critic_net_input = [observations, np.concatenate((actions, opponent_actions), 1)]
        # critic_net_input = [observations, np.concatenate(actions, 1)]
        critic_net_input = (observations, np.hstack((actions, opponent_actions)))
        # print(observations.shape)
        # print(actions.shape)
        # print(opponent_actions.shape)
        q_values = self._qf.get_values(critic_net_input)

        critic_loss = self._td_errors_loss_fn(reduction=tf.losses.Reduction.NONE)(td_targets, q_values)

        if weights is not None:
            critic_loss = weights * critic_loss

        critic_loss = tf.reduce_mean(critic_loss)


        print(critic_loss)
        return critic_loss

    def actor_loss(self, observations, opponent_actions, weights=None):
        """Computes the actor_loss for DDPG training.
        Args:
          observations: A batch of observations.
          weights: Optional scalar or element-wise (per-batch-entry) importance
            weights.
          # TODO: Add an action norm regularizer.
        Returns:
          actor_loss: A scalar actor loss.
        """
        # print(observations.shape)
        mixed_observation = (observations, opponent_actions)
        actions = self._policy.get_actions(mixed_observation)
        actions = tf.concat((actions, opponent_actions), 1)
        q_values = self._qf.get_values([observations.astype(np.float32), actions])
        if weights is not None:
            q_values = weights * q_values
        actor_loss = -tf.reduce_mean(q_values)
        # print(actor_loss, 'actor_loss')

        print('actor-loss-1:')
        print(actor_loss)
        return actor_loss