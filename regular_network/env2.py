from gym import Env
from gym.spaces import Discrete, Box, Dict
import numpy as np
import random
import math


class WMSEnvironment(Env):
    def __init__(self, coordinate, playerchannel):
        # parameters
        self.node = 10
        self.r = 4  # radio per node

        #length of path
        #self.rt = 150 #sparse
        #self.rt=200 #regular
        self.rt = 350  #dense

        self.transmit_power = (10 ** (15 / 10)) / (1000)  # tp = 15
        self.reference_distance = 1
        self.path_loss_reff = 35  # ptr = 35
        self.background_noise = (10 ** (-95 / 10)) / (1000)  # bgn = -95
        self.path_loss_exponent = 3
        self.xg = 0  # fading
        self.sinr_thres = 1
        self.pi = math.pi
        # self.numchannel = self.r * 2 - 1
        self.numchannel = 12
        self.numplayer = self.node * self.r
        self.channel = list(range(1, self.numchannel + 1))
        self.player = list(range(1, self.numplayer + 1))

        self.beta = 50
        self.gamma = 30

        self._max_episode_steps = 100

        # set node and link
        # self.coordinate=self.node_coordinate()
        self.coordinate = coordinate
        self.designated_link, self.designated_distance = self.des_link()
        while len(self.designated_link) == 0:
            self.coordinate = self.node_coordinate()
            self.designated_link, self.designated_distance = self.des_link()

        self.reward = 0.0 #initial rewards

        # initial channel assignments
        self.initial_channel = playerchannel
        self.playerchannel = self.initial_channel
        self.current_channel = self.initial_channel
        # self.chosen_player=random.choice(self.player)
        # self.chosen_player=self.player[chosenplayer]

        # assigning radio-channel pairs to link
        # divide radio's channels based on nodes
        self.channel_player = []
        # add = 0
        for i in range(0, len(self.current_channel), self.r):
            temp_channel = []
            for j in range(0, self.r):
                temp_channel.append(self.current_channel[i + j])

            self.channel_player.append(temp_channel)
            # add += (self.r - 1)

        self.utility = self.initial_utility()

        # search common channel between nodes in a designated link
        self.common_channel = []
        self.working_link = []
        self.working_distance = []
        for i in range(len(self.designated_link)):
            a = self.designated_link[i]
            com_chan = set(self.channel_player[a[0]]).intersection(self.channel_player[a[1]])
            if len(com_chan) != 0:
                self.common_channel.append(com_chan)
                self.working_link.append(self.designated_link[i])
                self.working_distance.append(self.designated_distance[i])

        self.state = self.getobservation()

        # Actions we can take: choose channel
        self.action_space = Discrete(self.numchannel)

        self.observation_space = Box(low=1, high=self.numchannel, shape=self.state.shape)

        # episode length
        self.steps_left = 100

    def getobservation(self):
        # state = np.zeros((self.numplayer, 3))  ## channel, utility, node
        state = np.zeros(self.numplayer)
        for i in range(self.numplayer):
            # state[i,0]=i+1
            state[i] = self.playerchannel[i]

            # x = (i-(i% self.r))/self.r
            # print(x)
            # state[i,2]=self.coordinate[int((i-(i% self.r))/self.r)]

            # state[i,1]=self.utility[i]
            # state[i,2]= int((i-(i% self.r))/self.r)
        return state

    def step(self, action):

        #stage 1: channel allocation to radios
        self.previous_utility = self.utility
        # self.chosen_player_old_channel=self.playerchannel[self.chosen_player-1]
        self.previous_channel = self.current_channel
        self.prev_working_link=self.working_link

        # apply action
        self.current_channel = action

        # assigning radio-channel pairs to link
        # divide radio's channels based on nodes
        self.channel_player = []
        # add = 0
        for i in range(0, len(self.current_channel), self.r):
            temp_channel = []
            for j in range(0, self.r):
                temp_channel.append(self.current_channel[i + j])

            self.channel_player.append(temp_channel)
            # add += (self.r - 1)

        self.utility = self.initial_utility()
        self.playerchannel = self.current_channel

        step_reward = 0

        # search common channel between nodes in a designated link
        self.common_channel = []
        self.working_link = []
        self.working_distance = []
        for i in range(len(self.designated_link)):
            a = self.designated_link[i]
            com_chan = set(self.channel_player[a[0]]).intersection(self.channel_player[a[1]])
            if len(com_chan) != 0:
                self.common_channel.append(com_chan)
                self.working_link.append(self.designated_link[i])
                self.working_distance.append(self.designated_distance[i])

        #rewards calculations
        if len(self.prev_working_link)<len(self.working_link):
            step_reward+=30 #if committed link increase
        if len(self.prev_working_link)>len(self.working_link):
            step_reward-=30 #if committed link decrease
        if len(self.prev_working_link)==len(self.working_link):
            for i in range(len(self.current_channel)):
                # check if the new channel is better
                if self.utility[i] > self.previous_utility[i] and self.utility[i] - self.previous_utility[i] >= 40:
                    step_reward += 5 #if utility increase significantly (radios in the same node use the same channel -> use different channels)
                if self.utility[i] > self.previous_utility[i] and self.utility[i] - self.previous_utility[i] < 40:
                    step_reward += 1 #if utility increase not significantly
                if self.utility[i] == self.previous_utility[i]:
                    step_reward += 0 #utility stays the same
                if self.utility[i] < self.previous_utility[i] and self.utility[i] - self.previous_utility[i] <= -40:
                    step_reward += -5 #if utility decrease significantly (use different channels -> radios in the same node use the same channel)
                if self.utility[i] < self.previous_utility[i] and self.utility[i] - self.previous_utility[i] > -40:
                    step_reward += -1 #if utility decrease not significantly


        #update reward
        self.reward = step_reward

        # update state
        self.state = self.current_channel

        #check termination condition
        if self.steps_left <= 0 or self.reward == 0:
            done = True
        else:
            done = False

        """
        previous_player=self.chosen_player
        self.chosen_player=random.choice(self.player)
        if (self.chosen_player==previous_player):
          self.chosen_player=random.choice(self.player)
        """

        info = {}

        if done:

            #stage 2: assign radio-channel pairs to links
            # check if there are more than 2 common channel
            multiple_channel = []
            multiple_channel_link = []
            for i in range(len(self.working_link)):
                if len(self.common_channel[i]) > 1:
                    multiple_channel_link.append(self.working_link[i])
                    multiple_channel.append(self.common_channel[i])

            if len(multiple_channel_link) != 0:
                # compute neighbor count
                multiple_channel_util = []
                for i in range(len(multiple_channel_link)):
                    neighbour = []
                    a = multiple_channel_link[i]
                    for j in range(len(a)):
                        neighbour_count = 0
                        for k in range(len(self.working_link)):
                            b = self.working_link[k]
                            if a != b:
                                if len(set(a).intersection(b)) != 0:
                                    if (a[j] in b):
                                        neighbour_count -= 1
                        neigh_util = neighbour_count
                        neighbour.append(neigh_util)
                    multiple_channel_util.append(neighbour)

                # assign common channel based on neighbor count
                # print('multiple channel',multiple_channel)
                # print('utility',multiple_channel_util)
                for i in range(len(multiple_channel_link)):
                    a = list(multiple_channel[i])
                    a_util = multiple_channel_util[i]
                    # print('a',a)
                    # print('a_util',a_util)
                    chosen_common_channel = a[0]
                    chosen_common_util = a_util[0]
                    for j in range(1, len(a_util)):
                        if chosen_common_util < a_util[j]:
                            chosen_common_channel = a[j]
                            chosen_common_util = a_util[j]
                    x = self.working_link.index(multiple_channel_link[i])
                    self.common_channel[x] = chosen_common_channel

            # compute OLR
            # compute sinr of both end of link
            sinr_list = []
            for i in range(len(self.working_link)):
                fik_sum_1 = 0
                fik_sum_2 = 0
                if self.working_distance[i] < 0:
                    print('working distance [i]', self.working_distance[i])
                link_path_loss = self.path_loss_reff + 10 * self.path_loss_exponent * math.log(
                    (self.working_distance[i] / self.reference_distance), 10) + self.xg
                link_rss = (10 ** (-link_path_loss / 10)) * self.transmit_power
                a = self.working_link[i]
                for j in range(len(self.player)):
                    if isinstance(self.common_channel[i], np.int64) or isinstance(self.common_channel[i], np.int32) or isinstance(
                            self.common_channel[i], int):
                        com_chan = self.common_channel[i]
                    else:
                        com_chan = list(self.common_channel[i])
                        com_chan = com_chan[0]

                    x = int((j - (j % self.r)) / self.r)  # node j

                    if x != a[0] and x != a[1] and self.playerchannel[j] == com_chan:
                        # a[0]
                        d_1 = np.sqrt(
                            (self.coordinate[a[0]][0] - self.coordinate[x][0]) ** 2 + (
                                    self.coordinate[a[0]][1] - self.coordinate[x][1]) ** 2)  # distance
                        other_path_loss_1 = self.path_loss_reff + 10 * self.path_loss_exponent * math.log(
                            (d_1 / self.reference_distance), 10) + self.xg
                        other_rss_1 = (10 ** (-other_path_loss_1 / 10)) * self.transmit_power
                        fik_sum_1 += other_rss_1
                        # a[1]
                        d_2 = np.sqrt(
                            (self.coordinate[a[1]][0] - self.coordinate[x][0]) ** 2 + (
                                    self.coordinate[a[1]][1] - self.coordinate[x][1]) ** 2)  # distance

                        other_path_loss_2 = self.path_loss_reff + 10 * self.path_loss_exponent * math.log(
                            (d_2 / self.reference_distance), 10) + self.xg
                        other_rss_2 = (10 ** (-other_path_loss_2 / 10)) * self.transmit_power
                        fik_sum_2 += other_rss_2

                #if sinr >1, then it's working link
                sinr_1 = 20 * math.log(link_rss / (fik_sum_1 + self.background_noise))
                if sinr_1 > 1:
                    sinr_2 = 20 * math.log(link_rss / (fik_sum_2 + self.background_noise))
                    if sinr_2 > 1:
                        sinr_list.append(self.working_link[i])

            # calculate OLR
            olr = len(sinr_list) / len(self.designated_link)

            # print('olr',olr)
            info = olr

        # reduce step left
        self.steps_left -= 1

        return self.state, self.reward, done, info

    def _action_player_(self, action):

        """
        previous_channel=self.playerchannel[self.chosen_player-1]
        current_channel=action+1

        #utility of chosen player
        total_util=0
        for i in range(0, self.numplayer):
          if i !=self.chosen_player-1:
            if current_channel == self.playerchannel[i]:
              a = int(((self.chosen_player-1)-((self.chosen_player-1)% self.r))/self.r)
              b = int((i-(i% self.r))/self.r)
              if a != b:
                d = np.sqrt(
                    (self.coordinate[a][0] - self.coordinate[b][0]) ** 2 + (
                        self.coordinate[a][1] - self.coordinate[b][1]) ** 2)  # distance
                chosen_player_utility = -(1 / ((d) ** self.path_loss_exponent))
              else:
                chosen_player_utility = -50
            else:
              chosen_player_utility = 0

            total_util+=chosen_player_utility
        """
        self.current_channel = action
        new_utility = self.initial_utility()

        return new_utility

    def reset(self):
        """
        #set node and link
        self.coordinate=self.node_coordinate()
        self.designated_link, self.designated_distance = self.des_link()
        while len(self.designated_link)==0:
          self.coordinate = self.node_coordinate()
          self.designated_link, self.designated_distance = self.des_link()
        """

        # reward
        self.reward = 0.0
        """
        #set starting channel allocation (random)
        self.playerchannel=[]
        for x in range(self.numplayer):
          assign = random.choice(self.channel)
          self.playerchannel.append(assign)

        """
        self.playerchannel = self.initial_channel

        # self.chosen_player=random.choice(self.player)
        self.utility = self.initial_utility()

        self.state = self.getobservation()

        # episode length
        self.steps_left = 10

        return self.state

    """
    def node_coordinate(self):
      coordinate = []
      while len(coordinate) < self.node:
        i_cord=[]
        x = np.random.randint(0, 1000)
        y = np.random.randint(0, 1000)
        i_cord.append(x)
        i_cord.append(y)      
        while i_cord in coordinate:
          x = np.random.randint(0, 1000)
          y = np.random.randint(0, 1000)
          i_cord.append(x)
          i_cord.append(y) 

        coordinate.append(i_cord)

      return coordinate
    """

    def des_link(self):
        #calculate and determine the designated links
        designated_link = []
        designated_distance = []

        for i in range(0, self.node):
            for j in range(i + 1, self.node):
                if i != j:
                    d = np.sqrt(
                        (self.coordinate[i][0] - self.coordinate[j][0]) ** 2 + (
                                self.coordinate[i][1] - self.coordinate[j][1]) ** 2)  # calculate distance
                    # print(d)

                    if d <= self.rt:  # if distance less or equal than rt
                        cord_temp = []
                        cord_temp.append(i)
                        cord_temp.append(j)
                        # print(i,j)
                        designated_link.append(cord_temp)  # link is operative
                        designated_distance.append(d)
        return designated_link, designated_distance

    def initial_utility(self):
        #calculate every agents' utility
        player_utility = []
        for i in range(0, self.numplayer):
            u_i = 0
            for j in range(0, self.numplayer):
                if i != j:
                    # u=utility(playerchannel, node_player, coordinate,path_loss_exponent, i, j)
                    a = int((i - (i % self.r)) / self.r)  # node i
                    b = int((j - (j % self.r)) / self.r)  # node j
                    """
                    if [a, b] in self.designated_link or [b, a] in self.designated_link:
                        # print(self.channel_player[a], self.channel_player[b])
                        com_chan = set(self.channel_player[a]).intersection(self.channel_player[b])
                        if len(com_chan) != 0:
                            # x=self.channel_player[a]
                            # y=self.channel_player[b]

                            u = self.gamma
                        if len(com_chan) == 0:
                            u = -self.beta
                    else:
                    """
                    if self.current_channel[i] == self.current_channel[j]:
                        # print('a',a,'b',b)
                        if a != b:
                            d = np.sqrt(
                                (self.coordinate[a][0] - self.coordinate[b][0]) ** 2 +
                                (self.coordinate[a][1] - self.coordinate[b][1]) ** 2)  # distance
                            u = -1 / ((d) ** self.path_loss_exponent)
                        else:
                                u = -self.beta
                    else:
                        u = 0

                    u_i += u

            player_utility.append(u_i)
        return player_utility

    def changed_utility(self):
        #calculate agents' utility if only 1 agent change channel at a time
        new_utility = []
        for i in range(len(self.player)):
            if i != self.chosen_player:
                # new
                if self.playerchannel[i - 1] == self.playerchannel[self.chosen_player - 1]:
                    a = int((i - (i % self.r)) / self.r)  # node i
                    b = int(
                        ((self.chosen_player - 1) - ((self.chosen_player - 1) % self.r)) / self.r)  # node chosen player
                    if a != b:
                        d = np.sqrt(
                            (self.coordinate[a][0] - self.coordinate[b][0]) ** 2 + (
                                    self.coordinate[a][1] - self.coordinate[b][1]) ** 2)  # distance
                        fc_new = 1 / ((d) ** self.path_loss_exponent)
                    else:
                        fc_new = 50
                else:
                    fc_new = 0
                # old
                if self.playerchannel[i - 1] == self.chosen_player_old_channel:
                    a = int((i - (i % self.r)) / self.r)  # node i
                    b = int(
                        ((self.chosen_player - 1) - ((self.chosen_player - 1) % self.r)) / self.r)  # node chosen player
                    if a != b:
                        d = np.sqrt(
                            (self.coordinate[a][0] - self.coordinate[b][0]) ** 2 + (
                                    self.coordinate[a][1] - self.coordinate[b][1]) ** 2)  # distance
                        fc_old = 1 / ((d) ** self.path_loss_exponent)
                    else:
                        fc_old = 50
                else:
                    fc_old = 0

                u_new = self.previous_utility[i - 1] + fc_old - fc_new

            else:
                u_new = self.current_utility

            new_utility.append(u_new)
        return new_utility