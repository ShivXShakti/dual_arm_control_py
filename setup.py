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
    maintainer='scg',
    maintainer_email='kuldeeplakhansons@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'test3 = dual_arm_control_py.test3:main',
            'js_control = dual_arm_control_py.js_control:main',
            'fpc = dual_arm_control_py.fpc:main',
            'home_fpc = dual_arm_control_py.home_fpc:main',
        ],
    },
)
