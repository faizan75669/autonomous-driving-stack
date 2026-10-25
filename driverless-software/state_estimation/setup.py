from setuptools import setup

package_name = 'state_estimation'

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
    description='EKF state estimation (odometry + IMU + GPS) for the FERN driverless car',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ekf_node = state_estimation.ekf_node:main',
        ],
    },
)
