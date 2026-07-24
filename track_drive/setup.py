from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'track_drive'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # launch 파일을 설치
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # PolyLaneNet 체크포인트(track_drive/weights/*) 설치 -- lane_util.py의
        # DEFAULT_WEIGHTS_PATH가 ament_index_python으로 이 share 경로를 찾는다.
        (os.path.join('share', package_name, 'weights'), glob(os.path.join(package_name, 'weights', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'track_drive = track_drive.track_drive:main',
        ],
    },
)
