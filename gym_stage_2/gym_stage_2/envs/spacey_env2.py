

# coding: utf-8

# In[1]:


import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
import numpy as np
import random


# In[2]:


class FooEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):

        self.screen_width = 600
        self.screen_height = 400

        self.min_position = 0
        self.max_position = self.screen_width
        self.max_speed = 10
        self.min_speed = 0

        self.goal_position = self.screen_width

        self.min_y_position = 0
        self.max_y_position = self.screen_height

        self.force=1
        self.gravity=0.0025

        self.step_size = 10

        ################################################## 40 Episode for each location of the planets ########################
        self.steady_episodes = 40
        self.episodes_count = 0
        self.previous_target_planet_center = [0,0]
        self.previous_origin_planet_center = [0,0]
        ######################################################################################################################

        ################################ Specific number of Locations ########################################################
        #self.possible_origin_locations = [[150, 150], [200, 200],[50, 50]]#, [400, 300]]
        #self.possible_target_locations = [[350, 350], [300, 300],[400, 300]]#, [100, 170]]

        self.defined_set_of_points = [[100, 100], [200, 200], [500, 100]]#, [100, 200], [300, 200], [500, 200], [100, 300], [300, 300], [500, 300]]
        self.possible_origin_locations = []#[[400, 200], [200, 200],[300, 300], [300, 100], [450, 350], [450, 50], [150, 50], [150, 350]]
        self.possible_target_locations = []#[[300, 200], [300, 200],[300, 200], [300, 200], [300, 200], [300, 200], [300, 200], [300, 200]]
        self.inc_test_1 = 0
        self.inc_test_2 = 0
        for i in range(len(self.defined_set_of_points)):
            for j in range(len(self.defined_set_of_points)):
                if j != i:
                    self.possible_origin_locations.append(self.defined_set_of_points[i])
                    self.possible_target_locations.append(self.defined_set_of_points[j])

        self.inc_test_1 = len(self.possible_origin_locations)
        self.inc_test_2 = len(self.possible_target_locations)

        self.inc = 0
        self.possible_episodes = 1000
        self.possible_episodes_count = 0
        self.possible_previous_origin_center = [0, 0]
        self.possible_previous_target_center = [0, 0]
        self.test = 0
        #############################################################################################################3##

        #self.origin_planet_center = [random.uniform(-1.2, 0.6), random.uniform(0.0, 1.2)]
        #self.target_planet_center = [random.uniform(-1.2, 0.6), random.uniform(0.0, 1.2)]
        #self.planet_radius = 10
        #while math.sqrt((self.origin_planet_center[0] - self.target_planet_center[0])**2 + (self.origin_planet_center[1] - self.target_planet_center[1])**2) < 5*self.planet_radius:
        #    self.target_planet_center = [random.uniform(-1.2, 0.6), random.uniform(0.0, 1.2)]

        self.low = np.array([self.min_position, self.min_y_position, 0])#, 0, 0])
        self.high = np.array([self.max_position, self.max_y_position, 6.3])#, self.screen_width, self.screen_height])

        self.viewer = None

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.seed()
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        posx, posy, rot = self.state
        position, position_y = posx + self.target_planet_center[0], posy + self.target_planet_center[1]
        #position, position_y = posx, posy
        velocity = 0
        velocity_y = 0
        """
        velocity += (action-1)*self.force + math.cos(3*position)*(-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        """
        if action==0:
            velocity += self.force
            velocity = np.clip(velocity, -self.max_speed, self.max_speed)
            position += self.step_size * math.cos(rot)
            position_y += self.step_size * math.sin(rot)
            position = np.clip(position, self.min_position, self.max_position)
        elif action==1:
            velocity_y += self.force
            velocity_y = np.clip(velocity_y, -self.max_speed, self.max_speed)
            position -= self.step_size * math.sin(rot)
            position_y += self.step_size * math.cos(rot)
            position_y = np.clip(position_y, self.min_y_position, self.max_y_position)
        elif action==2:
            velocity -= self.force
            velocity = np.clip(velocity, -self.max_speed, self.max_speed)
            position -= self.step_size * math.cos(rot)
            position_y -= self.step_size * math.sin(rot)

            position = np.clip(position, self.min_position, self.max_position)
        elif action==3:
            velocity_y -= self.force
            velocity_y = np.clip(velocity_y, -self.max_speed, self.max_speed)
            position += self.step_size * math.sin(rot)
            position_y -= self.step_size * math.cos(rot)
            position_y = np.clip(position_y, self.min_y_position, self.max_y_position)
        elif action ==4:
            rot += 0.1
        elif action ==5:
            rot -= 0.1

        if rot >= 6.28319:
            rot = 0
        if rot <0:
            rot = 6.28319 + rot

        if (position==self.min_position and velocity<0) or (position==self.max_position and velocity>0):
            velocity = 0

        if (position_y==self.min_y_position and velocity_y<0) or (position_y==self.max_y_position and velocity_y>0):
            velocity_y = 0

        alpha = math.atan2((self.target_planet_center[0] - position), (position_y - self.target_planet_center[1]))
        done = False
        dist = math.sqrt((position - self.target_planet_center[0])**2 + (position_y - self.target_planet_center[1])**2)
        reward = - 0.002#((dist / self.screen_width) / 10) + (abs(alpha - rot) / 10)
        succeeded = False
        if self.rocket_in_planet_range(position, position_y):
            alpha = math.atan2((self.target_planet_center[0] - position), (position_y - self.target_planet_center[1]))
            if alpha < 0 :
                alpha = 3.14 + (3.14 + alpha)

            #print("Alpha", alpha, math.degrees(alpha))
            #print("Rot", rot)
            #print("Bounds", rot + 0.78, rot - 0.78)
            first_condition = rot <= (alpha + 0.78)
            second_condition = rot >= (alpha - 0.78)
            condition = first_condition and second_condition
            if alpha < 0.78:
                #print("Modified bounds", (alpha + 0.78), " ", (6.28319 - (0.78 - alpha)))
                condition = (rot <= (alpha + 0.78) or (rot >= (6.28319 - (0.78 - alpha))))
            if alpha > (6.28319 - 0.78):
                #print("Modified Bounds", (alpha - 0.78), " ", (0.78 - (6.28319 - alpha)))
                condition = (rot >= alpha - 0.78) or (rot <= (0.78 - (6.28319 - alpha)))

            if condition:
                done = True
                reward = 1
                succeeded = True
            else:
                done = True
                reward = -0.5
        else:
            done = bool(position <= self.min_position) or bool(position >= self.max_position) or bool(position_y >= self.max_y_position) or bool(position_y <= self.min_y_position)
            if done:
                reward = -1
        self.state = ((position - self.target_planet_center[0]), (position_y - self.target_planet_center[1]), rot)
        normalized_state = ((position - self.target_planet_center[0]) / self.screen_width , (position_y - self.target_planet_center[1]) / self.screen_height, rot / 6.28319)
        test_bool = False
        if bool(self.possible_episodes_count % self.possible_episodes == 0):
            if self.test == 0:
                test_bool = True
                self.test += 1
        else:
                self.test = 0
        if done:
            print(succeeded)
        return np.array(normalized_state), reward, done, succeeded, test_bool

    def reset(self):
        screen_width = self.screen_width
        screen_height = self.screen_height

        index = self.inc % len(self.possible_origin_locations)#random.randint(0, len(self.possible_origin_locations)-1)

        #############Possible Locations##############
        if self.possible_episodes_count % self.possible_episodes == 0:
            self.origin_planet_center= self.possible_origin_locations[index]
            self.target_planet_center = self.possible_target_locations[index]
            self.inc += 1
        else:
            self.origin_planet_center = self.possible_previous_origin_center
            self.target_planet_center = self.possible_previous_target_center
        ##############################################

        self.possible_episodes_count += 1
        self.possible_previous_target_center = self.target_planet_center
        self.possible_previous_origin_center = self.origin_planet_center
        ###############################################
        #print(self.origin_planet_center)
        #print(self.target_planet_center)
        #print(len(self.possible_origin_locations))
        #print(len(self.possible_target_locations))
        # #########################Planets################################################3
        # if self.episodes_count % self.steady_episodes == 0:
        #     self.origin_planet_center = [random.randint(0, screen_width), random.randint(0, screen_height)]
        #     self.target_planet_center = [random.randint(0, screen_width), random.randint(0, screen_height)]
        # else:
        #     self.origin_planet_center = self.previous_origin_planet_center
        #     self.target_planet_center = self.previous_target_planet_center
        #
        # self.episodes_count +=1
        # self.previous_origin_planet_center = self.origin_planet_center
        # self.previous_target_planet_center = self.target_planet_center
        # #####################################################################################
        #

        self.planet_radius = 20

        ##################################################HardCoded for now. Remove the comments when you do it randomly 1!######
        #while math.sqrt((self.origin_planet_center[0] - self.target_planet_center[0]) ** 2 + (
        #        self.origin_planet_center[1] - self.target_planet_center[1]) ** 2) < 6 * self.planet_radius:
        #    self.target_planet_center = [random.randint(0, screen_width), random.randint(0, screen_height)]
        #
        #################################################################################

        #self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), self.np_random.uniform(low=0, high=0.2),0, 0])

        self.state = np.array([self.origin_planet_center[0] - self.target_planet_center[0], self.origin_planet_center[1] + self.planet_radius - self.target_planet_center[1] ,0])
        normalized_state = np.array([(self.origin_planet_center[0] - self.target_planet_center[0])/screen_width, (self.origin_planet_center[1] + self.planet_radius - self.target_planet_center[1]) / screen_height ,0])
        return np.array(normalized_state)

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55
    def rocket_in_planet_range(self, posx, posy):
        inX = bool(posx >= (self.target_planet_center[0] - self.planet_radius) and posx <= (self.target_planet_center[0] + self.planet_radius))
        inY = bool(posy >= (self.target_planet_center[1] - self.planet_radius) and posy <= (self.target_planet_center[1] + self.planet_radius))
        return inX and inY

    def render(self, mode='human'):
        screen_width = self.screen_width
        screen_height = self.screen_height

        #world_width = self.max_position - self.min_position
        #scale = screen_width/world_width
        carwidth=20
        carheight=80




        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            #xs = np.linspace(self.min_position, self.max_position, 100)
            #ys = self._height(xs)
            #xys = list(zip((xs-self.min_position)*scale, ys*scale))

            #self.track = rendering.make_polyline(xys)
            #self.track.set_linewidth(4)
            #self.viewer.add_geom(self.track)




            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0#self.planet_radius+carheight, self.planet_radius
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.car_transform = rendering.Transform(translation=(0, 0))
            car.add_attr(self.car_transform)
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            
            frontwheel = rendering.FilledPolygon([(-carwidth/4,0), (-carwidth/2,carheight/2), (-carwidth, - carheight/2), (-(3*carwidth)/4,-carheight)])
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            #frontwheel.add_attr(self.cartrans)
            #self.viewer.add_geom(frontwheel)
            
            
            backwheel = rendering.FilledPolygon([(-carwidth/4,carheight), (-carwidth/2,carheight/2), (-carwidth,  (3*carheight)/2), (-(3*carwidth)/4,(2*carheight))])
            backwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            #backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            #self.viewer.add_geom(backwheel)


            #Planets rendering
            #self.planetsTrans = rendering.Transform()
            origin_planet = rendering.make_circle(self.planet_radius)
            origin_planet.set_color(.7, .7, .3)
            #origin_planet.add_attr(self.planetsTrans)
            self.origin_planet_transform = rendering.Transform(translation=(self.origin_planet_center[0], self.origin_planet_center[1] + self.planet_radius))
            origin_planet.add_attr(self.origin_planet_transform)
            self.viewer.add_geom(origin_planet)

            target_planet = rendering.make_circle(self.planet_radius)
            target_planet.set_color(.6, .2, .1)
            #target_planet.add_attr(self.planetsTrans)
            self.target_planet_transform = rendering.Transform(translation=(self.target_planet_center[0], self.target_planet_center[1] + self.planet_radius))
            target_planet.add_attr(self.target_planet_transform)
            self.viewer.add_geom(target_planet)


        pos = self.state[0] + self.target_planet_center[0]
        posY = self.state[1] + self.target_planet_center[1]

        rot = self.state[2]
        self.cartrans.set_translation(pos, posY)#self._height(pos)
        self.cartrans.set_rotation(rot)# math.cos(3 * pos)

        self.target_planet_transform.set_translation(self.target_planet_center[0], self.target_planet_center[1])
        self.origin_planet_transform.set_translation(self.origin_planet_center[0], self.origin_planet_center[1])

        ##Planets
        #self.planetsTrans.set_translation(self.min_position * scale, self.min_y_position*scale)
        #self.planetsTrans.set_rotation(0)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
    
    def get_keys_to_action(self):
        return {():1,(276,):0,(275,):2,(275,276):1} #control with left and right arrow keys 
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

