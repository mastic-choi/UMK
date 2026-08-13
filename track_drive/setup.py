from setuptools import setup, find_packages
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

        # launch 파일 설치
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),

        # 모델 가중치 설치 (perception/dl_lane.py _default_model_path()가
        # share/track_drive/models/<파일명>을 1순위로 찾음)
        (os.path.join('share', package_name, 'models'),
            glob(os.path.join(package_name, 'models', '*'))),
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
