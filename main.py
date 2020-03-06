from random import random, choice
import tqdm
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from kaggle_environments import evaluate, make, utils

env = make("connectx", debug=True)
conf = env.configuration

def obs2input(observation):
    obs = np.array(observation).reshape(-1, 1, env.configuration.rows, env.configuration.columns)
    obs_0, obs_1, obs_2 = [np.zeros_like(obs) for _ in range(3)]
    obs_0[obs == 0] = 1
    obs_1[obs == 1] = 1
    obs_2[obs == 2] = 1
    input_net = np.concatenate([obs_0, obs_1, obs_2], axis = 1).astype(np.float32)
    assert (input_net.sum(axis=1).sum(axis=0)==len(input_net)).all()
    return input_net

def raw2tensor(experiences):
    states, actions, rewards, next_states = list(zip(*experiences))
    # states(20,42), actions(20), rewards(20), nstates(20,42)
    states, next_states = obs2input(states), obs2input(next_states)
    # states(20,3,7,6), nstates(20,3,7,6)
    return tf.convert_to_tensor(states), tf.convert_to_tensor(actions), tf.convert_to_tensor(rewards), tf.convert_to_tensor(next_states)

def swap_board_if_2p(observation):
    if observation.mark == 2:
        swap_token = {0:0,1:2,2:1}
        observation.board = [swap_token[cell] for cell in observation.board]

def my_me(observation, configuration):
    actions = range(1,configuration.columns+1)
    print("+---"*configuration.columns+"+")
    print("".join("| %d "%i for i in actions)+"|")
    print("| v "*configuration.columns+"|")
    env.render()
    a = 0
    while a not in actions:
        print("You have only 5 seconds per move")
        a = int(input(f"Action({min(actions)}-{max(actions)}): "))
    return a-1

# This agent random chooses a non-empty column.
def my_agent(observation, configuration):
    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])

def my_dqn_agent(observation, configuration):
    swap_board_if_2p(observation)
    obs_net = obs2input(observation.board)
    qvalues = model(obs_net)[0] + (1 - obs_net[0,0,0,:])*-99 # To restrict non possible actions
    action = int(model.act(qvalues))
    return action

def my_dqn_agent_exploring(observation, configuration):
    if lets_explore > random():
        first_row = observation.board[:configuration.columns]
        actions = [pos for pos in range(configuration.columns) if first_row[pos] == 0]
        action = choice(actions)
    else:
        swap_board_if_2p(observation)
        obs_net = obs2input(observation.board)
        qvalues = model(obs_net)[0] + (1 - obs_net[0,0,0,:])*-99 # To restrict non possible actions
        action = int(model.act(qvalues))
    return action

def play(agent_1, agent_2, render = True):
    data = env.run([agent_1, agent_2])
    # env.render()
    result = {1.:'WIN',.5:'DRAW',0.:'LOSE'}
    if id(agent_1) == id(agent_2):
        agent_2 = 'yourself'
    print(f"YOU {result[data[-1][0]['reward']]} against {agent_2}")
    print(f"Moves: {len(data)}")
    if render:
        print("final state:")
        env.render()

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
        super().__init__()
        self.no = conf.columns
        self.conv1 = Conv2D(64, 4, padding='same', activation='linear')
        self.maxp1 = tf.keras.layers.MaxPool2D()
        self.relu1 = tf.keras.layers.ReLU()
        self.conv2 = Conv2D(32, 3, padding='same', activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(64, activation='relu')
        self.d3 = Dense(32, activation='relu')
        self.d4 = Dense(self.no, activation=None)
        self.act = lambda x: np.argmax(x, axis=-1)
        self.sel = lambda a, x: tf.reduce_sum(x*tf.one_hot(a, model.no), axis=-1)
        self.max = lambda x: tf.reduce_max(x, axis=-1)

    def call(self, x):
        x = self.conv1(x)
        x = self.maxp1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return x

alpha = 0.1
gamma = 1.

@tf.function
def train_step(states, actions, rewards, next_states):
    with tf.GradientTape() as tape:
        # q(s,a)_t+1 = q(s,a) - α*err
        # err = (q(s,a) - r+γ*max_a(q(s_n,a))
        # expected_act_rewards = [e_s[a] for (e_s, a) in zip(model(states),actions)] # Fails because iterations in a propagation are forbidden
        expected_act_rewards = model.sel(actions, model(states))
        # ear(20)
        expected_next_rewards = model.max(model(next_states))
        # enr(20)
        new_expected_act_rewards = expected_act_rewards - alpha*((expected_act_rewards - (rewards + gamma*expected_next_rewards)))
        loss = loss_object(new_expected_act_rewards, expected_act_rewards)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    # train_accuracy(new_expected_act_rewards, expected_act_rewards)

model = MyDQN(env.configuration)
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')


# Play as first position against random agent.
def load_memory(batch = 20, memory_max_size = 1000):
    memory = Memory(max_size = memory_max_size)
    trainer = env.train([None, "random"])
    done = True
    for _ in range(batch):
        if done:
            observation = trainer.reset()
        action = my_agent(observation, env.configuration)
        next_observation, reward, done, info = trainer.step(my_agent(observation, env.configuration))
        experience = observation.board, action, reward, next_observation.board
        memory.add(experience)
        observation = next_observation
    return memory

batch = 20
memory = load_memory(batch, 1000000)
trainer1p = env.train([None, my_dqn_agent])
trainer2p = env.train([my_dqn_agent, None])
lets_explore = 0.2
total_epochs = 0
while True:
    EPOCHS = 10
    template = "Epoch {}/{}, Loss: {}, Moves: {}"
    # for epoch in tqdm.tqdm(range(EPOCHS)):
    for epoch in range(EPOCHS):
        total_epochs += 1
        moves = 0
        done = False
        trainer = choice([trainer1p, trainer2p])
        observation = trainer.reset()
        swap_board_if_2p(observation) # 
        while not env.done:
            moves += 1
            action = my_dqn_agent_exploring(observation, env.configuration)
            next_observation, reward, done, info = trainer.step(action)
            swap_board_if_2p(next_observation)
            reward -= 0.5
            memory.add((observation.board, action, reward, next_observation.board))
            experiences = memory.sample(batch)
            experiences_tensor = raw2tensor(experiences)
            train_step(*experiences_tensor)
            observation = next_observation
        print(template.format(epoch+1, total_epochs, train_loss.result(), moves))
    
        train_loss.reset_states()
        # env.render(mode="human", width=100, height=90, header=False, controls=False)
    
    
    # Playing versus random
    print("LETS PLAY\n")
    play(my_dqn_agent, 'random', render = False)
    play(my_dqn_agent, 'negamax', render = False)
    play(my_dqn_agent, my_dqn_agent, render = False)
    # print("YOU PLAY NOW! YOU CAN WIN!!!!")
    # try:
    #     play(my_me, my_dqn_agent)
    # except KeyError:
    #     pass
