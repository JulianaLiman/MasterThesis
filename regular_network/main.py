#from environment import WMSEnvironment
from env2 import WMSEnvironment
from agent import DqnAgent
import tensorflow as tf

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras

from pyparsing.helpers import infix_notation

#coordinates
#sparse
#coordinate=[[46, 297], [20, 13], [113, 202], [405, 314], [61, 381], [357, 371], [246, 302], [63, 87], [317, 315], [142, 454]]
#regular
#coordinate = [[373, 347], [212, 156], [15, 367], [408, 495], [257, 318], [347, 398], [429, 251], [86, 356], [30, 181], [219, 328]]
#dense
coordinate=[[330, 206], [227, 407], [335, 39], [489, 80], [97, 40], [238, 400], [162, 260], [383, 449], [397, 392], [208, 146]]

#initial channel assignments
#playerchannel=[2, 3, 2, 1, 1, 3, 1, 3, 2, 1, 1, 3, 1, 3, 3, 3, 1, 3, 3, 1] #20 players
#playerchannel=[9, 4, 5, 9, 9, 3, 1, 6, 7, 3, 8, 7, 6, 6, 2, 5, 4, 1, 7, 1, 8, 7, 10, 8, 6, 8, 5, 7, 9, 10] #30 players
playerchannel=[5, 5, 10, 2, 6, 6, 9, 6, 2, 6, 3, 8, 9, 1, 9, 2, 10, 10, 3, 3, 5, 9, 6, 9, 4, 9, 6, 1, 9, 9, 3, 8, 5, 3, 7, 3, 6, 2, 3, 10] #40 players

model = keras.Sequential()

total_episodes = 300
max_env_steps = 100


epsilon = 1.0               # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.999

time_history = []
rew_history = []
olr=[]

env= WMSEnvironment(coordinate=coordinate, playerchannel=playerchannel)

s_size = env.observation_space.shape[0]
a_size = env.action_space.n

#agent models
player1 = DqnAgent(s_size, a_size)
player2 = DqnAgent(s_size, a_size)
player3 = DqnAgent(s_size, a_size)
player4 = DqnAgent(s_size, a_size)
player5 = DqnAgent(s_size, a_size)
player6 = DqnAgent(s_size, a_size)
player7 = DqnAgent(s_size, a_size)
player8 = DqnAgent(s_size, a_size)
player9 = DqnAgent(s_size, a_size)
player10 = DqnAgent(s_size, a_size)
player11 = DqnAgent(s_size, a_size)
player12 = DqnAgent(s_size, a_size)
player13 = DqnAgent(s_size, a_size)
player14 = DqnAgent(s_size, a_size)
player15 = DqnAgent(s_size, a_size)
player16 = DqnAgent(s_size, a_size)
player17 = DqnAgent(s_size, a_size)
player18 = DqnAgent(s_size, a_size)
player19 = DqnAgent(s_size, a_size)
player20 = DqnAgent(s_size, a_size)
#"""
player21 = DqnAgent(s_size, a_size)
player22 = DqnAgent(s_size, a_size)
player23 = DqnAgent(s_size, a_size)
player24 = DqnAgent(s_size, a_size)
player25 = DqnAgent(s_size, a_size)
player26 = DqnAgent(s_size, a_size)
player27 = DqnAgent(s_size, a_size)
player28 = DqnAgent(s_size, a_size)
player29 = DqnAgent(s_size, a_size)
player30 = DqnAgent(s_size, a_size)
#"""
player31 = DqnAgent(s_size, a_size)
player32 = DqnAgent(s_size, a_size)
player33 = DqnAgent(s_size, a_size)
player34 = DqnAgent(s_size, a_size)
player35 = DqnAgent(s_size, a_size)
player36 = DqnAgent(s_size, a_size)
player37 = DqnAgent(s_size, a_size)
player38 = DqnAgent(s_size, a_size)
player39 = DqnAgent(s_size, a_size)
player40 = DqnAgent(s_size, a_size)
#"""

from pyparsing.helpers import infix_notation

tf.keras.backend.clear_session()
for e in range(total_episodes):

    state = env.reset()
    # print('state', state)
    state = np.reshape(state, [1, s_size])
    # rewardsum = []
    rewardsum = 0
    for time in range(max_env_steps):

        # Choose action
        if np.random.rand(1) < epsilon:
            action1 = np.random.randint(a_size)
            action2 = np.random.randint(a_size)
            action3 = np.random.randint(a_size)
            action4 = np.random.randint(a_size)
            action5 = np.random.randint(a_size)
            action6 = np.random.randint(a_size)
            action7 = np.random.randint(a_size)
            action8 = np.random.randint(a_size)
            action9 = np.random.randint(a_size)
            action10 = np.random.randint(a_size)
            action11 = np.random.randint(a_size)
            action12 = np.random.randint(a_size)
            action13 = np.random.randint(a_size)
            action14 = np.random.randint(a_size)
            action15 = np.random.randint(a_size)
            action16 = np.random.randint(a_size)
            action17 = np.random.randint(a_size)
            action18 = np.random.randint(a_size)
            action19 = np.random.randint(a_size)
            action20 = np.random.randint(a_size)
            #"""
            action21 = np.random.randint(a_size)
            action22 = np.random.randint(a_size)
            action23 = np.random.randint(a_size)
            action24 = np.random.randint(a_size)
            action25 = np.random.randint(a_size)
            action26 = np.random.randint(a_size)
            action27 = np.random.randint(a_size)
            action28 = np.random.randint(a_size)
            action29 = np.random.randint(a_size)
            action30 = np.random.randint(a_size)
            #"""
            action31 = np.random.randint(a_size)
            action32 = np.random.randint(a_size)
            action33 = np.random.randint(a_size)
            action34 = np.random.randint(a_size)
            action35 = np.random.randint(a_size)
            action36 = np.random.randint(a_size)
            action37 = np.random.randint(a_size)
            action38 = np.random.randint(a_size)
            action39 = np.random.randint(a_size)
            action40 = np.random.randint(a_size)
            #"""
        else:
            # action = np.argmax(model.predict(state)[0])
            # action = player.get_action(state)
            action1 = player1.get_action(state)
            action2 = player2.get_action(state)
            action3 = player3.get_action(state)
            action4 = player4.get_action(state)
            action5 = player5.get_action(state)
            action6 = player6.get_action(state)
            action7 = player7.get_action(state)
            action8 = player8.get_action(state)
            action9 = player9.get_action(state)
            action10 = player10.get_action(state)
            action11 = player11.get_action(state)
            action12 = player12.get_action(state)
            action13 = player13.get_action(state)
            action14 = player14.get_action(state)
            action15 = player15.get_action(state)
            action16 = player16.get_action(state)
            action17 = player17.get_action(state)
            action18 = player18.get_action(state)
            action19 = player19.get_action(state)
            action20 = player20.get_action(state)
            #"""
            action21 = player1.get_action(state)
            action22 = player2.get_action(state)
            action23 = player3.get_action(state)
            action24 = player4.get_action(state)
            action25 = player5.get_action(state)
            action26 = player6.get_action(state)
            action27 = player7.get_action(state)
            action28 = player8.get_action(state)
            action29 = player9.get_action(state)
            action30 = player10.get_action(state)
            #"""
            action31 = player1.get_action(state)
            action32 = player2.get_action(state)
            action33 = player3.get_action(state)
            action34 = player4.get_action(state)
            action35 = player5.get_action(state)
            action36 = player6.get_action(state)
            action37 = player7.get_action(state)
            action38 = player8.get_action(state)
            action39 = player9.get_action(state)
            action40 = player10.get_action(state)
            #"""

        # Step
        # next_state, reward, done, info = env.step(action)
        actionvec = [action1, action2, action3, action4, action5, action6, action7, action8, action9, action10,
                     action11, action12, action13, action14, action15, action16, action17, action18, action19, action20,
                     action21, action22, action23, action24, action25, action26, action27, action28, action19, action30
                     ,action31, action32, action33, action34, action35, action36, action37, action38, action39, action40
                     ]
        next_state, reward, done, info = env.step(actionvec)

        # rewardsum.append(reward)
        # avg_reward=sum(rewardsum)/len(rewardsum)

        if done:
            print("episode: {}/{}, time: {}, rew: {}, eps: {:.2}, olr: {}"
                  .format(e, total_episodes, time, rewardsum, epsilon, info))
            break

        next_state = np.reshape(next_state, [1, s_size])

        # Train
        target1 = reward
        target2 = reward
        target3 = reward
        target4 = reward
        target5 = reward
        target6 = reward
        target7 = reward
        target8 = reward
        target9 = reward
        target10 = reward
        target11 = reward
        target12 = reward
        target13 = reward
        target14 = reward
        target15 = reward
        target16 = reward
        target17 = reward
        target18 = reward
        target19 = reward
        target20 = reward
        #"""
        target21 = reward
        target22 = reward
        target23 = reward
        target24 = reward
        target25 = reward
        target26 = reward
        target27 = reward
        target28 = reward
        target29 = reward
        target30 = reward
        #"""
        target31 = reward
        target32 = reward
        target33 = reward
        target34 = reward
        target35 = reward
        target36 = reward
        target37 = reward
        target38 = reward
        target39 = reward
        target40 = reward
        #"""

        if not done:
            # target = (reward + 0.9 * np.amax(model.predict(next_state)[0]))
            target1 = reward + 0.95 * np.amax(player1.predict(next_state))
            target2 = reward + 0.95 * np.amax(player2.predict(next_state))
            target3 = reward + 0.95 * np.amax(player3.predict(next_state))
            target4 = reward + 0.95 * np.amax(player4.predict(next_state))
            target5 = reward + 0.95 * np.amax(player5.predict(next_state))
            target6 = reward + 0.95 * np.amax(player6.predict(next_state))
            target7 = reward + 0.95 * np.amax(player7.predict(next_state))
            target8 = reward + 0.95 * np.amax(player8.predict(next_state))
            target9 = reward + 0.95 * np.amax(player9.predict(next_state))
            target10 = reward + 0.95 * np.amax(player10.predict(next_state))
            target11 = reward + 0.95 * np.amax(player11.predict(next_state))
            target12 = reward + 0.95 * np.amax(player12.predict(next_state))
            target13 = reward + 0.95 * np.amax(player13.predict(next_state))
            target14 = reward + 0.95 * np.amax(player14.predict(next_state))
            target15 = reward + 0.95 * np.amax(player15.predict(next_state))
            target16 = reward + 0.95 * np.amax(player16.predict(next_state))
            target17 = reward + 0.95 * np.amax(player17.predict(next_state))
            target18 = reward + 0.95 * np.amax(player18.predict(next_state))
            target19 = reward + 0.95 * np.amax(player19.predict(next_state))
            target20 = reward + 0.95 * np.amax(player20.predict(next_state))
            #"""
            target21 = reward + 0.95 * np.amax(player21.predict(next_state))
            target22 = reward + 0.95 * np.amax(player22.predict(next_state))
            target23 = reward + 0.95 * np.amax(player23.predict(next_state))
            target24 = reward + 0.95 * np.amax(player24.predict(next_state))
            target25 = reward + 0.95 * np.amax(player25.predict(next_state))
            target26 = reward + 0.95 * np.amax(player26.predict(next_state))
            target27 = reward + 0.95 * np.amax(player27.predict(next_state))
            target28 = reward + 0.95 * np.amax(player28.predict(next_state))
            target29 = reward + 0.95 * np.amax(player29.predict(next_state))
            target30 = reward + 0.95 * np.amax(player30.predict(next_state))
            #"""
            target31 = reward + 0.95 * np.amax(player31.predict(next_state))
            target32 = reward + 0.95 * np.amax(player32.predict(next_state))
            target33 = reward + 0.95 * np.amax(player33.predict(next_state))
            target34 = reward + 0.95 * np.amax(player34.predict(next_state))
            target35 = reward + 0.95 * np.amax(player35.predict(next_state))
            target36 = reward + 0.95 * np.amax(player36.predict(next_state))
            target37 = reward + 0.95 * np.amax(player37.predict(next_state))
            target38 = reward + 0.95 * np.amax(player38.predict(next_state))
            target39 = reward + 0.95 * np.amax(player39.predict(next_state))
            target40 = reward + 0.95 * np.amax(player40.predict(next_state))
            #"""

        # target_f = model.predict(state)
        # target_f[0][action] = target
        # model.fit(state, target_f, epochs=1, verbose=0)

        player1.fit(state, target1, action1)
        player2.fit(state, target2, action2)
        player3.fit(state, target3, action3)
        player4.fit(state, target4, action4)
        player5.fit(state, target5, action5)
        player6.fit(state, target6, action6)
        player7.fit(state, target7, action7)
        player8.fit(state, target8, action8)
        player9.fit(state, target9, action9)
        player10.fit(state, target10, action10)
        player11.fit(state, target11, action11)
        player12.fit(state, target12, action12)
        player13.fit(state, target13, action13)
        player14.fit(state, target14, action14)
        player15.fit(state, target15, action15)
        player16.fit(state, target16, action16)
        player17.fit(state, target17, action17)
        player18.fit(state, target18, action18)
        player19.fit(state, target19, action19)
        player20.fit(state, target20, action20)
        #"""
        player21.fit(state, target21, action21)
        player22.fit(state, target22, action22)
        player23.fit(state, target23, action23)
        player24.fit(state, target24, action24)
        player25.fit(state, target25, action25)
        player26.fit(state, target26, action26)
        player27.fit(state, target27, action27)
        player28.fit(state, target28, action28)
        player29.fit(state, target29, action29)
        player30.fit(state, target30, action30)
        #"""
        player31.fit(state, target31, action31)
        player32.fit(state, target32, action32)
        player33.fit(state, target33, action33)
        player34.fit(state, target34, action34)
        player35.fit(state, target35, action35)
        player36.fit(state, target36, action36)
        player37.fit(state, target37, action37)
        player38.fit(state, target38, action38)
        player39.fit(state, target39, action39)
        player40.fit(state, target40, action40)
        #"""

        state = next_state
        # rewardsum.append(reward)
        # avg_reward=sum(rewardsum)/len(rewardsum)
        rewardsum += reward
        if epsilon > epsilon_min: epsilon *= epsilon_decay

    time_history.append(time)

    rew_history.append(rewardsum)
    olr.append(info)
"""
player1.model.save("model_1.h5")
player2.model.save("model_2.h5")
player3.model.save("model_3.h5")
player4.model.save("model_4.h5")
player5.model.save("model_5.h5")
player6.model.save("model_6.h5")
player7.model.save("model_7.h5")
player8.model.save("model_8.h5")
player9.model.save("model_9.h5")
player10.model.save("model_10.h5")
player11.model.save("model_11.h5")
player12.model.save("model_12.h5")
player13.model.save("model_13.h5")
player14.model.save("model_14.h5")
player15.model.save("model_15.h5")
player16.model.save("model_16.h5")
player17.model.save("model_17.h5")
player18.model.save("model_18.h5")
player19.model.save("model_19.h5")
player20.model.save("model_20.h5")
player21.model.save("model_21.h5")
player22.model.save("model_22.h5")
player23.model.save("model_23.h5")
player24.model.save("model_24.h5")
player25.model.save("model_25.h5")
player26.model.save("model_26.h5")
player27.model.save("model_27.h5")
player28.model.save("model_28.h5")
player29.model.save("model_29.h5")
player30.model.save("model_30.h5")

player31.model.save("model_31.h5")
player32.model.save("model_32.h5")
player33.model.save("model_33.h5")
player34.model.save("model_34.h5")
player35.model.save("model_35.h5")
player36.model.save("model_36.h5")
player37.model.save("model_37.h5")
player38.model.save("model_38.h5")
player39.model.save("model_39.h5")
player40.model.save("model_40.h5")
"""

#import matplotlib.pyplot as plt

print("Plot Learning Performance")
mpl.rcdefaults()
mpl.rcParams.update({'font.size': 16})

#shows the rewards and olr of every episodes (to calculate rewards)
print('rewards')
print(rew_history)
print('average olr', sum(olr)/len(olr))
print(olr)

#shows rewards graph
fig, ax = plt.subplots(figsize=(10,4))
plt.grid(True, linestyle='--')
plt.title('Learning Performance')
#plt.plot(range(len(time_history)), time_history, label='Steps', marker="^", linestyle=":")#, color='k')
plt.plot(range(len(rew_history)), rew_history, linestyle="-")#, color='red')
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.legend(prop={'size': 12})

plt.savefig('learning.pdf', bbox_inches='tight')
plt.show()

#shows olr graphs
fig, ax = plt.subplots(figsize=(10,4))
plt.grid(True, linestyle='--')
#plt.title('OLR')
plt.plot(range(len(olr)), olr, linestyle="-")#, color='red')
plt.xlabel('Episode')
plt.ylabel('OLR')
plt.legend(prop={'size': 12})

plt.savefig('olr.pdf', bbox_inches='tight')
plt.show()




