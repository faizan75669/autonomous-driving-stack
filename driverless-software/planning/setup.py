from setuptools import setup

package_name = 'planning'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='faizan75669',
    maintainer_email='149727349+faizan75669@users.noreply.github.com',
    description='Local path planners: Delaunay gate planner, midpoint planner, RRT planner',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'delaunay_planner = planning.delaunay_planner_node:main',
            'midpoint_planner = planning.midpoint_planner_node:main',
            'rrt_planner = planning.rrt_planner_node:main',
        ],
    },
)
