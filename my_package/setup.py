from setuptools import find_packages, setup
import os
import glob

package_name = 'my_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['tests*', 'numpy*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob.glob('launch/*.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='maruyama',
    maintainer_email='u451328k@ecs.osaka-u.ac.jp',
    description='TODO: Package description',
    license='TODO: License declaration',
    # tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'coordinator_node = my_package.coordinator_node:main',
        'multi_node_runner = my_package.multi_node_runner:main',
    ],
}
)
