"""
Run the whole stack with the EUFS simulator.

    ros2 launch bringup trackdrive.launch.py
    ros2 launch bringup trackdrive.launch.py planner:=rrt_planner controller:=stanley

planner    : delaunay_planner (default) | midpoint_planner | rrt_planner
controller : pure_pursuit (default) | stanley
gui        : true -> EUFS launcher window opens, pick the track there
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory("bringup"), "config", "params.yaml")
    launcher_config = os.path.join(get_package_share_directory("eufs_launcher"), "config", "eufs_launcher.yaml")

    planner = LaunchConfiguration("planner")
    controller = LaunchConfiguration("controller")

    return LaunchDescription([
        DeclareLaunchArgument("planner", default_value="delaunay_planner"),
        DeclareLaunchArgument("controller", default_value="pure_pursuit"),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("mapping", default_value="true"),

        Node(package="eufs_launcher", executable="eufs_launcher", name="eufs_launcher", output="screen",
             parameters=[{"config": launcher_config, "gui": LaunchConfiguration("gui")}]),
        Node(package="state_estimation", executable="ekf_node", name="state_estimation",
             output="screen", parameters=[params]),
        Node(package="slam", executable="cone_mapping_node", name="cone_mapping", output="screen",
             parameters=[params], condition=IfCondition(LaunchConfiguration("mapping"))),
        Node(package="planning", executable=planner, name="planner", output="screen",
             parameters=[params]),
        Node(package="control", executable=controller, name="controller", output="screen",
             parameters=[params]),
    ])
