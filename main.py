import torch
from trainers.sarsa_train import SarsaTrain
from trainers.q_learning_train import QLearningTrain
from trainers.dql_train import DQLTrain
from trainers.q_learning_state_epsilon_train import QLearningStateEpsilonTrain
import time
import numpy as np
import matplotlib.pyplot as plt
from utils.helpers import export_q_table

# Reinforcement learning parameters
epsilon_init = 0.2
epsilon_decay = 0.9
tau_init = 10
tau_decay = 0.9

alpha_init = 0.15
alpha_decay = 0.95
gamma = 0.9
decay_rate = 100

# Game parameters
size_x = 40
size_y = 25

learning_rate = 0.00005
# train = SarsaTrain(epsilon_init, epsilon_decay, alpha_init, alpha_decay,
#     gamma, decay_rate, size_x, size_y)

# train = QLearningTrain(tau_init, tau_decay, alpha_init, alpha_decay,
#    gamma, decay_rate, size_x, size_y)

# train = QLearningStateEpsilonTrain(epsilon_init, epsilon_decay, alpha_init, alpha_decay,
#      gamma, decay_rate, size_x, size_y)

train = DQLTrain(epsilon_init, epsilon_decay, learning_rate,
   gamma, decay_rate, size_x, size_y, load_model_path="pretrained_models/trained_weights.pth")

# Main logic
iter = 0
start_time = time.monotonic()
try:
    while True:
        if (iter % 1000 > 998):
            train.iterate(visual=True, speed=25)
        train.iterate(visual=False)
        iter += 1

except KeyboardInterrupt:
    if not train.use_deep_learning:
        print("[!] Received interruption signal. Time elapsed : %ds" %
                (time.monotonic() - start_time))

        # Plot results
        t = np.arange(0, iter, train.decay_rate)

        fig, ax = plt.subplots()
        ax.plot(t[:-1], train.avg_scores)

        ax.set(xlabel='Number of episodes',
                ylabel='Average score over the last {} iterations'.format(
                    train.decay_rate),
                title='Evolution of average score across episodes')
        ax.grid()

        fig.savefig("test.png")
        export_q_table(train.q_dict, "q_table")
        plt.show()
        # plt.close()
    else:
        torch.save(train.policy_net.state_dict(), "weights.pth")
