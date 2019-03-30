import copy
#import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from fourrooms import Fourrooms
from DQNAgent import DQNAgent
from sklearn.cluster import KMeans

EPISODES = 1000

class HierarchicalDQNAgent:
    def __init__(self, state_size, primitive_action_space, num_clusters, N = 5):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.rng = np.random.RandomState(12345)
        self.num_clusters = num_clusters
        self.discount_factor = 0.9

        # get size of state and action
        self.state_size = state_size
        self.primitive_action_space = primitive_action_space
        self.primitive_action_size = len(primitive_action_space)
        self.action_vectors = self.rng.rand(self.primitive_action_size, N)
        kmeans = KMeans(n_clusters = self.num_clusters, random_state=0).fit(self.action_vectors)
        self.action_to_cluster = kmeans.labels_
        self.action_size = num_clusters

        self.lower_agents = {}
        self.cluster_to_actionset = {}
        for ind in range(self.num_clusters):
            self.cluster_to_actionset[ind] = np.where(self.action_to_cluster == ind)[0]
            self.lower_agents[ind] = DQNAgent(self.state_size, self.cluster_to_actionset[ind])

        self.upper_agent = DQNAgent(self.state_size, list(range(self.action_size)))

    def get_action(self, state):
        cluster = self.upper_agent.get_action(state)
        action = self.lower_agents[cluster].get_action(state)
        return cluster, action

    def replay_memory(self, state, cluster, action, reward, next_state, done):
        self.upper_agent.replay_memory(state, cluster, reward, next_state, done)
        self.lower_agents[cluster].replay_memory(state, action, reward, next_state, done)

    def train_replay(self, cluster):
        self.upper_agent.train_replay()
        self.lower_agents[cluster].train_replay()

    def load_model(self, name):
        self.upper_agent.load_model(name + '_u')
        for ind in range(self.num_clusters):
            self.lower_agents[ind].load_model(name + '_l' + str(ind))

    def save_model(self, name):
        self.upper_agent.save_model(name + '_u')
        for ind in range(self.num_clusters):
            self.lower_agents[ind].save_model(name + '_l' + str(ind))


if __name__ == "__main__":
    env = Fourrooms()
    agent = HierarchicalDQNAgent(len(env.tostate), list(range(env.n_actions)), 5)

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

            cluster, action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.replay_memory(state, cluster, action, reward, next_state, done)
            agent.train_replay(cluster) #only one of the lower level agents is trained at each step
            score += reward*current_discount
            current_discount *= agent.discount_factor

            state = copy.deepcopy(next_state)
            print("reward:", reward, "  done:", done, "  time_step:", global_step, "  epsilon:", agent.upper_agent.epsilon, "  state:", env.currentcell, "  action:", action , "  cluster:", cluster)

            if done:
                scores.append(score)
                episodes.append(e)
                #pylab.plot(episodes, scores, 'b')
                #pylab.savefig("./save_graph/10by10.png")
                print("episode:", e, "  score:", score, "  memory length:", len(agent.upper_agent.memory),
                      "  epsilon:", agent.upper_agent.epsilon, "  number of steps:", episode_step)
                print("\n----\n----\n-----\n-----")

        if e % 100 == 0:
        #    pass
            agent.save_model("./models/save_model1")

    # end of game
    print('game over')
    env.destroy()