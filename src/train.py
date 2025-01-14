from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!


"""
=====================================================================
Mike Huttenschmitt - Assignment in Reinforcement Learning (RL)
Master MVA - Mathematics, Vision, and Learning

Title: HIV Patient Treatment Policy Optimization
       Using Fitted Q-Iteration (FQI) with Random Forest

Description:
This project explores the application of reinforcement learning
to determine optimal treatment policies for HIV patients. 
FQI is used as the primary algorithm, leveraging Random Forest
as a regressor to approximate the Q-function.
Recall tha all my code is inspired by Emmanuel Rachelson's notebooks.

Author: Mike Huttenschmitt (MikeProgrammeur)
Date: January 2025
=====================================================================
"""

# My imports :
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import pickle
import gzip

#print(f" La dimension de l'espace des actions est {env.action_space.n} et la dimension de l'espace des états est {env.observation_space.shape[0]}")

# My class implementation
class ProjectAgent:
    def __init__(self):
        "Initialize the class with the necessary attributes for a reinforcement learning agent."
        self.state_dim = 6 # env.observation_space.n
        self.action_dim = 4 # env.action_space.n
        self.gamma = 0.90
        self.model = RandomForestRegressor()
        self.is_rf_fitted = False
        
    def collect_samples(self, env, horizon : int , eps : float):
        " Run the environnement to collect {horizon} tuples (s,a,r,s',d) ( deeply inspired by course notebook n°4 ;) )"
        S, A, R, S2, D = [], [], [], [], []
        state, _ = env.reset()
        for h in range(horizon):
            action = self.act(observation=state,use_random=(np.random.random()<eps))
            next_state, reward, done, trunc, _ = env.step(action)

            S.append(state)
            A.append(action)
            R.append(reward)
            S2.append(next_state)
            D.append(done)

            if done or trunc:
                state, _ = env.reset()
            else:
                state = next_state

        return (
            np.array(S),
            np.array(A).reshape(-1, 1),
            np.array(R),
            np.array(S2),
            np.array(D),
        )

    def train(self, max_iter : int, epochs : int ):
        "Train the agent using Fitted Q-Iteration (FQI)."
        horizon = 3000
        self.model = RandomForestRegressor()
        for epoch in range(epochs):
            # at each epoch we reset the buffer,
            print(f"Starting epoch n°{epoch+1}, buffer reset")
            S, A, R, S2, D = self.collect_samples(env, horizon = horizon, eps = 0.12 )
            for iteration in tqdm(range(max_iter)):
                Snew, Anew, Rnew, S2new, Dnew = self.collect_samples(env, horizon = 1000, eps = 0.05 )
                # We complete the buffer with new experiences ( we use specific stack cause of shapes )
                S = np.vstack((S, Snew))      # Vertically stack old states and new states
                A = np.vstack((A, Anew))      # Vertically stack old actions and new actions
                R = np.hstack((R, Rnew))      # Horizontally stack old rewards and new rewards
                S2 = np.vstack((S2, S2new))   # Vertically stack old next states and new next states
                D = np.hstack((D, Dnew))      # Horizontally stack old done flags and new done flags
                SA = np.append(S, A, axis=1)
                nb_samples = S.shape[0]
                if iteration == 0:
                    target_values = R.copy()
                else:
                    Q_next = np.zeros((nb_samples, self.action_dim))
                    for action in range(self.action_dim):
                        A_next = np.full((nb_samples, 1), action)
                        S2A_next = np.append(S2, A_next, axis=1)
                        Q_next[:, action] = self.model.predict(S2A_next)
                    max_Q_next = np.max(Q_next, axis=1)
                    target_values = R + self.gamma * (1 - D) * max_Q_next
                
                self.model.fit(SA, target_values)
                self.is_rf_fitted = True

    def act(self, observation, use_random=False):
        "Select an action based on the trained Q-functions or randomly"
        # Acts randomly for exploration
        if use_random or not(self.is_rf_fitted):
            return np.random.randint(self.action_dim)
        # Acts greedily
        Q_values = []
        for action in range(self.action_dim):
            obs_action = np.append(observation, action).reshape(1, -1)
            Q_values.append(self.model.predict(obs_action)[0])
        return np.argmax(Q_values)
    
    def save(self, path):
        "Save the model to a file."
        with gzip.open(path, 'wb') as f: # use zip to be under 100 Mb
            pickle.dump(self.model, f)
        print(f"Model saved to {path}")

    def load(self):
        "Load the model from a file, the file can't be given as argument and must be hard-written just below"
        path = "./ModelFinalMiki.gz"
        with gzip.open(path, 'rb') as f: # use zip to be under 100 Mb
            self.model = pickle.load(f)
        self.is_rf_fitted = True
        print(f"Model loaded from {path}")
        
if __name__ == "__main__":
    agentMiki = ProjectAgent()
    agentMiki.train(15,8)
    agentMiki.save("./ModelFinalMiki.gz")
    
    

