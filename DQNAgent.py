import copy
#import pylab
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from fourrooms import Fourrooms
import random

class DQNAgent:
    def __init__(self, state_size, action_space):
        self.action_space = action_space
        self.action_size = len(action_space)
        self.action_to_ind = {a:i for i,a in enumerate(action_space)}
        self.state_size = state_size
        self.discount_factor = 0.9
        self.learning_rate = 0.001
        self.update_count = 0

        self.epsilon = 1.
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.train_start = 100

        self.rng = np.random.RandomState(12345)

        self.memory = deque(maxlen=10000)
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def one_hot_state(self, state):
        return np.float32(np.eye(self.state_size)[state])

    def build_model(self):
        model = Sequential()
        model.add(Dense(100, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if self.rng.rand() <= self.epsilon:
            return self.action_space[self.rng.randint(self.action_size)]
        else:
            q_values = self.model.predict(self.one_hot_state(state)[None, :])
            return self.action_space[np.argmax(q_values[0])]

    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_replay(self):
        self.update_count += 1
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            state, action, reward, next_state, done = mini_batch[i]
            reward = np.float32(reward)
            target = self.model.predict(self.one_hot_state(state)[None, :])[0]
            #print(env.tocell[state], action, target[action])

            if done:
                target[self.action_to_ind[action]] = reward
            else:
                target[self.action_to_ind[action]] = reward + self.discount_factor * np.amax(self.model.predict(self.one_hot_state(next_state)[None, :])[0])

            update_input[i] = self.one_hot_state(state)
            update_target[i] = target

        #print(update_input, update_target)
        self.model.fit(update_input, update_target, batch_size=batch_size, epochs=1, verbose=0)

        if self.update_count%100 == 0: # every 100 time steps update the target model to be same with model
            self.update_target_model()

    def load_model(self, name):
        self.model.load_weights(name)

    def save_model(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = Fourrooms()
    agent = DQNAgent(len(env.tostate), list(range(env.n_actions)))

    global_step = 0
    # agent.load_model("same_vel_episode2 : 1000")
    scores, episodes = [], []

    EPISODES = 1000

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        current_discount = 1.0
        episode_step = 0
        while not done:
            #if agent.render:
            #    env.render()
            global_step += 1
            episode_step += 1

            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.replay_memory(state, action, reward, next_state, done)
            agent.train_replay()
            score += reward*current_discount
            current_discount *= agent.discount_factor

            state = copy.deepcopy(next_state)
            # print("reward:", reward, "  done:", done, "  time_step:", global_step, "  epsilon:", agent.epsilon, "  state:", env.currentcell, "  action:", action)
            
            # every 100 time steps update the target model to be same with model
            # if global_step % 100 == 0:
            #     agent.update_target_model()

            if done:
                scores.append(score)
                episodes.append(e)
                #pylab.plot(episodes, scores, 'b')
                #pylab.savefig("./save_graph/10by10.png")
                print("episode:", e, "  score:", score, "  memory length:", len(agent.memory),
                      "  epsilon:", agent.epsilon, "  number of steps:", episode_step)

        if e % 100 == 0:
        #    pass
            agent.save_model("./models/save_model1")

    # end of game
    print('game over')
    env.destroy()