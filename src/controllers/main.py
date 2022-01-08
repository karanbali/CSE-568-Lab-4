#!/usr/bin/env python

# Importing all dependencies
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import tf
import random
import math
from tf.transformations import euler_from_quaternion
from heapq import *


# Map.txt
map = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,
       0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,
       0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
       1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
       0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
       0,0,1,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,
       0,0,1,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,
       0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,
       0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,
       0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,
       0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,
       0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,
       0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,
       0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,
       0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
       0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,
       0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1,0,
       0,0,0,0,0,0,0,0,1,1,1,0,0,1,1,1,1,0,
       0,0,0,0,0,0,0,1,1,1,0,0,0,1,1,1,1,0,
       0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1,1]

# Copy of map for later use
map_txt = map

# Conversion into a numpy array
map = np.asarray(map).reshape((20,18))

# Conversion from world frame to map frame
def world_to_map(ind):
    x = ind[0]
    y = ind[1]
    new_x = 10 - y
    new_y = 9 + x
    return [new_x,new_y]

# Conversion from map frame to world frame
def map_to_world(ind):
    x = ind[0]
    y = ind[1]
    new_x = y - 9
    new_y = 10 - x
    return [new_x,new_y]

# Heuristic function (using Euclidean distance)
def h(p1, p2):

    # coefficient of heuristic function
    E = 1

    # Return Euclidean distance b/w point 'p1' & 'p2'
    return E*(math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2))

# Main 'A*' Function
def a_star(array, start, goal):

    # Defining 4 neighbours to search w.r.t to current element (i.e. up,down,left,right)
    neighbors = [(0,1),(0,-1),(1,0),(-1,0)]

    #  Initializing parameters (i.e. close set, parent node, open list, f,g)
    close_set = set()
    came_from = {}
    g = {start:0}
    f = {start:h(start, goal)}
    oheap = []
    heappush(oheap, (f[start], start))
    
    # Main loop
    while oheap:
        
        # Get first node/element
        current = heappop(oheap)[1]

        # If goal is found than break away from loop
        if current == goal:
            data = []

            # Get the path by recollecting parent node
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        # Add to close set (i.e. mark as visited/traversed)
        close_set.add(current)

        # Loop to traverse all possibe 4 neighbours defined above
        for i, j in neighbors:

            # Get a new neighbour
            neighbor = current[0] + i, current[1] + j

            # Calculate new 'g' for the neighbour
            g_new = g[current] + h(current, neighbor)

            # Check limits of map so that iteration doesn't get out of limits
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:

                    # skip if element is a 'wall'               
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    continue
            else:
                continue
                
            # Check if neighbour has been already traversed or 'g_new' is not the optimal one
            if neighbor in close_set and g_new >= g.get(neighbor, 0):
                continue
                
            # Check if 'g_new' is the optimal one or neighbour is not accounted for in oheap
            if  g_new < g.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                
                # Add current node as parent to the neighbour
                came_from[neighbor] = current

                # Update 'f' & 'g' score
                g[neighbor] = g_new
                f[neighbor] = g_new + h(neighbor, goal)
                heappush(oheap, (f[neighbor], neighbor))

    # If the algorithm Fails       
    return "A* Failed"



# Main function for node
def main():

    # Initializing Global variable
    global init_x
    global init_y
    global goal_x
    global goal_y

    # Getting initial position of bug
    init_x = -8.0
    init_y = -2.0

    # Getting final goal position
    goal_x = rospy.get_param("/lab4/goal_x")
    goal_y = rospy.get_param("/lab4/goal_y")

    # Getting conversions for initial & goal positions from world frame to map frame
    init_w2m = world_to_map([int(init_x),int(init_y)])
    goal_w2m = world_to_map([int(goal_x),int(goal_y)])
    
    # Getting map value for initial & goal positions
    init_v = map[init_w2m[0],init_w2m[1]]
    goal_v = map[goal_w2m[0],goal_w2m[1]]


    # Check if initial position is a wall
    if int(init_v) == 1:
        print("INVALID Starting Position")
        return "INVALID Starting Position" 

    # Check if goal position is a wall
    if int(goal_v) == 1:
        print("INVALID GOAL Position")
        return "INVALID GOAL Position"
    
    # Calculate optimal route b/w initial position & goal position using 'A*' algorithm 
    path = a_star(map, (init_w2m[0],init_w2m[1]), (goal_w2m[0],goal_w2m[1]))
    
    # Get a new numpy array using copy of map.txt
    mp = np.asarray(map_txt, dtype=object).reshape((20,18))

    # Loop through all nodes & modify to get the required pattern
    for i in range(20):
        for j in range(18):

            # If the particular node is in A* path then change to '*'
            if (i,j) in path:
                mp[i,j] = '*'

            # If the particular node is '0' then change to ' '
            elif mp[i,j] == 0:
                mp[i,j] = ' '

            # If the particular node is '1' then change to '1'
            elif mp[i,j] == 1:
                mp[i,j] = '1'

    # Print the pattern as asked in the assignment
    print(mp)

    
    # Convert all nodes in 'A*' path to their world coordinates respectively
    npath = []
    for i in path:
        new_i = map_to_world(i)
        npath.append(new_i)
    
    
    # Initiating lab4 node
    rospy.init_node('lab4')

    # Setting the rate of cycles
    rate = rospy.Rate(10)

    # Initializing some global constants
    global cnt
    global npx
    global npy
    global ords
    global s
    s = 0

    # Counter to keep track of next milestone/point to reach
    cnt = 0

    # Callback to '/base_pose_ground_truth', which saves the pose to a global variable
    def bug(ord):
        global ords
        ords = ord        

    # Get next milestone/point (x & y) in a global variable
    npx = npath[-1-cnt][0]
    npy = npath[-1-cnt][1]
            
                        


    # Subscribing to '/base_pose_ground_truth' to get its odometry readings with a callback function to 'bug'
    pos = rospy.Subscriber('/base_pose_ground_truth', Odometry, bug, queue_size=20)
    

    while not rospy.is_shutdown():
        try:

            # Initiating a publisher to '/cmd_vel' so that we can publish new Twist
            pos = rospy.Subscriber('/base_pose_ground_truth', Odometry, bug, queue_size=20)

            # Injecting Global variables
            global cnt
            global npx
            global npy
            global ords
            global s

            # Get next milestone/point (x & y) in a global variable
            npx = npath[-1-cnt][0]
            npy = npath[-1-cnt][1]

            # Current pose of the bug
            pos =  ords.pose.pose.position
            
            # Distance b/w next milestone/point & current position of bug
            p_dist = math.sqrt((pos.x - npx)**2 + (pos.y - npy)**2)
    
              
            # Initiating a publisher to '/cmd_vel' so that we can publish new Twist
            pub  =rospy.Publisher('/cmd_vel', Twist, queue_size=1)
            cmd_vel = Twist()        


            # Get Euler transformation for the current position of bug
            ang = ords.pose.pose.orientation
            (a_,b_,ang_eu) = euler_from_quaternion([ang.x,ang.y,ang.z,ang.w])
    
            # Calculate 'Turn' angle for bug     
            angle_delta = math.atan2(npy - pos.y,npx - pos.x)
            angle_delta1 = angle_delta - ang_eu

            # Angular velocity for bug
            cmd_vel.angular.z = angle_delta1
            
            
            # If Turn is below a threshold than move linearly only
            if abs(angle_delta1) < 0.01:
                cmd_vel.angular.z = 0
                cmd_vel.linear.x = 0.5

            # Else rotate only
            else:
                cmd_vel.angular.z = angle_delta1
                cmd_vel.linear.x = 0

            # Publish Twist
            pub.publish(cmd_vel)

            # If bug is too close to next milestone/point than update counter in order to update next milestone/point
            if p_dist < 0.35:
                cnt += 1
                
            
        except:
            pass	
    

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass