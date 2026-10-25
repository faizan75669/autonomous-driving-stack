from setuptools import setup

package_name = 'control'

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
    description='Pure pursuit and Stanley controllers with PID speed control',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pure_pursuit = control.pure_pursuit_node:main',
            'stanley = control.stanley_node:main',
        ],
    },
)
