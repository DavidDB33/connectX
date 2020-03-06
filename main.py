from kaggle_environments import evaluate, make, utils
import tensorflow as tf
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from random import choice

env = make("connectx", debug=True)
# env.render()

# This agent random chooses a non-empty column.
def my_agent(observation, configuration):
    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])


class Memory():
    def __init__(self, max_size = 1000):
        self.buffer = deque(maxlen = max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size = batch_size, replace = False)
        return [self.buffer[ii] for ii in idx]


class MyDQN(Model):
    def __init__(self, conf):
        super(MyDQN, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(conf.rows, activation=None)
        self.act = lambda x: np.argmax(x, axis=-1)
        self.max = lambda x: np.max(x, axis=-1)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

    def pred(self, x):
        return self.act(self())

alpha = 0.1
gamma = 1.

@tf.function
def train_step(experiences):
    states, actions, rewards, next_states, list(*zip(experiences))
    states = np.concatenate(states)
    next_states = np.concatenate(next_states)
    with tf.GradientTape() as tape:
        # q(s,a)_t+1 = q(s,a) - α*err
        # err = (q(s,a) - r+γ*max_a(q(s_n,a))
        expected_act_rewards = [e_s[a] for (e_s, a) in zip(model(states),actions)]
        expected_next_rewards = model.max(model(next_states))
        new_expected_act_rewards = expected_act_rewards - alpha*((expected_act_rewards - (rewards + gamma*expected_next_rewards)))
        loss = loss_object(new_expected_act_rewards, expected_act_rewards)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)
    

model = MyDQN(env.configuration)
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')



def my_dqn(observation, configuration):
    pass

# Play as first position against random agent.
trainer = env.train([None, "random"])

observation = trainer.reset()

while not env.done:
    my_action = my_agent(observation, env.configuration)
    print("My Action", my_action)
    observation, reward, done, info = trainer.step(my_action)
    env.render(mode="human", width=100, height=90, header=False, controls=False)
env.render()
