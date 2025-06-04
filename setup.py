from setuptools import find_packages, setup

package_name = 'dual_arm_control_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cstar',
    maintainer_email='kuldeeplakhansons@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    #tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fpc = dual_arm_control_py.fpc:main',
            'home_fpc = dual_arm_control_py.home_fpc:main',
            'hwc = dual_arm_control_py.hwc:main',
            'js_sub = dual_arm_control_py.js_sub:main',
            'gripper_control_effort = dual_arm_control_py.gripper_control_effort:main',
            'gripper_control_position = dual_arm_control_py.gripper_control_position:main',
            'gripper_hwc = dual_arm_control_py.gripper_hwc:main',
            'gripper_test = dual_arm_control_py.gripper_test:main',
        ],
    },
)
