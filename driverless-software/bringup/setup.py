from glob import glob
from setuptools import setup

package_name = 'bringup'

setup(
    name=package_name,
    version='1.0.0',
    packages=[],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='faizan75669',
    maintainer_email='149727349+faizan75669@users.noreply.github.com',
    description='Launch files and parameters for the FERN driverless stack',
    license='MIT',
)
