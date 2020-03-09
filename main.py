from random import random, choice
import os
import datetime
from collections import deque
import math
import statistics as stt
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
    if next(lets_explore) > random():
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
    if agent_2 == 'random':
        data = [env.run([agent_1, agent_2]) for i in range(10)]
        wins = sum(d[-1][0]['reward']==1. for d in data)
        draws = sum(d[-1][0]['reward']==0.5 for d in data)
        loses = sum(d[-1][0]['reward']==0. for d in data)
        result = {0:'WIN',1:'DRAW',2:'LOSE'}[([wins,draws,loses].index(max([wins,draws,loses])))]
        result = "{}({}/{}/{})".format(result,wins,draws,loses)
        moves = stt.mean([len(d) for d in data])
    else:
        data = env.run([agent_1, agent_2])
        result_num = data[-1][0]['reward']
        result = {1.:'WIN ',.5:'DRAW',0.:'LOSE'}[result_num]
        moves = len(data)
    if id(agent_1) == id(agent_2):
        agent_2 = 'yourself'
    print(f"YOU {result} against {agent_2}")
    print(f"Moves: {moves}")
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
        self.conv1 = Conv2D(1024, 2, padding='valid', activation='relu')
        self.conv2 = Conv2D(2048, 2, padding='valid', activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(2048, activation='relu')
        self.d2 = Dense(512, activation='relu')
        self.d3 = Dense(128, activation='relu')
        self.d4 = Dense(32, activation='relu')
        self.d5 = Dense(self.no, activation=None)
        self.act = lambda x: np.argmax(x, axis=-1)
        self.sel = lambda a, x: tf.reduce_sum(x*tf.one_hot(a, model.no), axis=-1)
        self.max = lambda x: tf.reduce_max(x, axis=-1)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.d5(x)
        return x

alpha = 0.1
gamma = 1.
model = MyDQN(env.configuration)
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Nadam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')

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

# Play as first position against random agent.
def load_memory(env, conf, batch = 20, max_size = 1000):
    memory = Memory(max_size = max_size)
    done = True
    for _ in range(batch):
        if done:
            observation = env.reset()
        action = my_agent(observation, conf)
        next_observation, reward, done, info = env.step(my_agent(observation, conf))
        experience = observation.board, action, reward, next_observation.board
        memory.add(experience)
        observation = next_observation
    return memory

def load_model(model):
    """Load the weights of the model (mutate it) and return the recomended path to save it"""
    weights_path = "res/weights"
    _path = []
    for d in weights_path.split('/'):
        _path.append(d)
        try:
            os.mkdir('/'.join(_path))
        except FileExistsError:
            pass # It makes no sense to create it if already exists
    weights_name_load_files = sorted(os.listdir(weights_path), reverse=True)
    if weights_name_load_files:
        model.load_weights(os.path.join(weights_path, weights_name_load_files[0]))
    weights_name_save = 'weights'+datetime.datetime.now().isoformat(timespec='seconds')+'.h5'
    return os.path.join(weights_path, weights_name_save)


EPOCHS = 100
batch = 20
memory_trainer = env.train([None, "random"])
memory = load_memory(memory_trainer, env.configuration, batch=batch, max_size=1000000)
trainer1p = env.train([None, my_dqn_agent])
trainer2p = env.train([my_dqn_agent, None])
def exploring_gen(min_exp):
    i = 0
    while True:
        yield min_exp + (1-min_exp)*math.e**-(i/10000)
        i += 1
lets_explore = exploring_gen(0.2)
total_epochs = 0
weights_file_save = load_model(model)
while True:
    template = "Epoch {:02}/{}({}), Loss: {}, Moves: {}"
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
        print(template.format(epoch+1, EPOCHS, total_epochs, train_loss.result(), moves))
    
        train_loss.reset_states()
        # env.render(mode="human", width=100, height=90, header=False, controls=False)
    model.save_weights(weights_file_save)
    
    # Playing versus random
    print("LETS PLAY\n")
    play(my_dqn_agent, 'random', render = False)
    play(my_dqn_agent, 'negamax', render = False)
    # play(my_dqn_agent, my_dqn_agent, render = False)
print("YOU PLAY NOW! YOU CAN WIN!!!!")
try:
    play(my_me, my_dqn_agent)
    # play(my_me, 'negamax')
except KeyError:
    pass
