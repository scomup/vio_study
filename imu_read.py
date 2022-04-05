#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import matplotlib.pyplot as plt
import rosbag
import numpy as np

def main():



    gps = []
    imu = []

    bag = rosbag.Bag('/home/liu/bag/imu_pose.bag', "r")
    count = 0

    for topic, msg, t in bag.read_messages():
        if topic=="imu":
            imu.append([msg.header.stamp.to_sec(),msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z,\
                msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z,])
        if topic=="current_pose":
            gps.append([msg.header.stamp.to_sec(),msg.pose.position.x,msg.pose.position.y,msg.pose.position.z,\
                msg.pose.orientation.w, msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z])

        count += 1

    bag.close()

    gps = np.array(gps)
    imu = np.array(imu)
    np.save('/home/liu/bag/gps.npy', gps)
    np.save('/home/liu/bag/imu.npy', imu)

    plt.plot(gps[:,0],gps[:,1])
    plt.show()

    return

if __name__ == '__main__':
    main()
