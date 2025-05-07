from .base_data import Rigid3d, TimedPose, ImuData, Time
import numpy as np
from typing import List
from collections import deque
from scipy.spatial.transform import Rotation as R

class PoseExtrapolator:
    def __init__(self, pose_queue_duration: float, imu_gravity_time_constant: float):
        self.pose_queue_duration_ = pose_queue_duration
        self.gravity_time_constant_ = imu_gravity_time_constant
        self.cached_extrapolated_pose_ = TimedPose(time=Time.min(), pose=Rigid3d.identity())
        self.imu_data_ = deque()
        self.timed_pose_queue_ = deque()

        self.last_pose_time_ = None
        self.last_linear_acceleration_time_ = Time.min()
        self.gravity_vector_ = np.array([0.0, 0.0, 9.81])
        self.orientation_ = R.from_quat([0, 0, 0, 1])
        self.linear_velocity_from_poses_ = np.zeros(3)
        self.angular_velocity_from_poses_ = np.zeros(3)

        self.initial_acceleration_data_ = []
        self.initial_acceleration_mean_ = np.zeros(3)
        self.bool_find_z_ = False
        self.bool_correct_z_ = False
        self.bool_pose_10_frame_=True
    @classmethod
    def initialize_with_imu(cls, imu_data: 'ImuData'):
        try:
            pose_queue_duration = 0.01
            imu_gravity_time_constant = 10.0
            extrapolator = cls(pose_queue_duration, imu_gravity_time_constant)
            extrapolator.add_imu_data(imu_data)

            translation = np.zeros(3)
            z_down_rotation = R.from_quat([0, 0, 0, 1])
            rigid_transform = Rigid3d(translation, z_down_rotation)
            extrapolator.add_pose(imu_data.time, rigid_transform)
            return extrapolator
        except Exception as e:
            print("Initialization failed:", str(e))
            return None

    def add_imu_data(self, imu_data: ImuData):
        if self.timed_pose_queue_ and imu_data.time < self.timed_pose_queue_[-1].time:
            raise ValueError("IMU data timestamp must be greater than or equal to the last pose timestamp")
        self.imu_data_.append(imu_data)
        self.trim_imu_data()

    def add_pose(self, time: Time, pose: Rigid3d):
        self.timed_pose_queue_.append(TimedPose(time, pose))
        while (len(self.timed_pose_queue_) > 2 and
               self.timed_pose_queue_[1].time.timestamp <= time.timestamp - self.pose_queue_duration_):
            self.timed_pose_queue_.popleft()
        self.last_pose_time_ = time
        self.update_velocities_from_poses()
        self.trim_imu_data()

    def trim_imu_data(self):
        while (len(self.imu_data_) > 1 and
               self.timed_pose_queue_ and
               self.imu_data_[1].time <= self.timed_pose_queue_[-1].time):
            self.imu_data_.popleft()

    def get_last_extrapolated_time(self) -> Time:
        return self.last_pose_time_ if self.last_pose_time_ is not None else Time.min()

    def update_velocities_from_poses(self):
        if len(self.timed_pose_queue_) < 2:
            return

        newest_timed_pose = self.timed_pose_queue_[-1]
        oldest_timed_pose = self.timed_pose_queue_[0]

        time_delta = newest_timed_pose.time - oldest_timed_pose.time
        if time_delta <= 0:
            return

        translation_delta = newest_timed_pose.pose.translation - oldest_timed_pose.pose.translation
        self.linear_velocity_from_poses_ = translation_delta / time_delta

        rotation_diff = oldest_timed_pose.pose.rotation.inv() * newest_timed_pose.pose.rotation
        angular_velocity = rotation_diff.as_rotvec() / time_delta
        self.angular_velocity_from_poses_ = angular_velocity

    def ExtrapolatePose(self, time: Time) -> Rigid3d:
        newest_timed_pose = self.timed_pose_queue_[-1]
        if time < newest_timed_pose.time:
            raise ValueError("Extrapolation time must be greater than or equal to the last pose time")

        if self.cached_extrapolated_pose_ is None or self.cached_extrapolated_pose_.time != time:
            self.AddImuLinearAccelerationObservation()
            rotation = self.ExtrapolateRotation(time)
            translation = self.ExtrapolateTranslation(time) + newest_timed_pose.pose.translation
            self.cached_extrapolated_pose_ = TimedPose(time, Rigid3d(translation, rotation))
        return self.cached_extrapolated_pose_.pose

    def process_initial_imu_data(self,global_linear_acceleration):
        """
        处理前十五帧IMU数据，计算初始加速度的均值，并更新相关标志变量。
        """
        if len(self.initial_acceleration_data_) <=15:
            # 如果尚未收集到足够的数据，直接返回
            self.initial_acceleration_data_.append(global_linear_acceleration)
            return

        # 计算前十五帧IMU数据的加速度均值（去掉前4帧和后1帧，取中间10帧的均值）
        self.initial_acceleration_mean_ = np.mean(self.initial_acceleration_data_[4:14], axis=0)
        self.bool_find_z_ = True  # 标志变量，表示已经找到初始加速度的均值

        if self.bool_pose_10_frame_:
            # 如果是第一次处理，记录初始姿态
            self.init_pose_10_frame_ = self.orientation_
            self.bool_pose_10_frame_ = False

    def ExtrapolateTranslation(self, time: Time) -> np.ndarray:
        newest_timed_pose = self.timed_pose_queue_[-1]
        extrapolation_delta = time - newest_timed_pose.time

        current_linear_acceleration = self.imu_data_[-1].linear_acceleration
        global_linear_acceleration = self.orientation_.apply(current_linear_acceleration)

        self.process_initial_imu_data(global_linear_acceleration)

        if self.bool_find_z_:
            adjusted_linear_acceleration = global_linear_acceleration - self.initial_acceleration_mean_
            adjusted_linear_acceleration[np.abs(adjusted_linear_acceleration) < 0.015] = 0
            translation = extrapolation_delta * self.linear_velocity_from_poses_ + 0.5 * adjusted_linear_acceleration * extrapolation_delta**2
        else:
            translation = extrapolation_delta * self.linear_velocity_from_poses_
        return translation

    def ExtrapolateRotation(self, time: Time) -> R:
        return self.orientation_

    def AddImuLinearAccelerationObservation(self):
        delta_t = self.imu_data_[-1].time - self.last_linear_acceleration_time_ if self.last_linear_acceleration_time_ > Time.min() else np.inf
        self.last_linear_acceleration_time_ = self.imu_data_[-1].time

        alpha = 1.0 - np.exp(-delta_t / self.gravity_time_constant_)
        self.gravity_vector_ = (1.0 - alpha) * self.gravity_vector_ + alpha * self.imu_data_[-1].linear_acceleration

        rotation = self.from_two_vectors(self.gravity_vector_, self.orientation_.apply([0, 0, 1]))
        self.orientation_ = self.orientation_ * rotation

        if not self.bool_find_z_:
            rotated_gravity_vector = self.orientation_.apply(self.gravity_vector_)
            normalized_rotated_gravity_vector = rotated_gravity_vector / np.linalg.norm(rotated_gravity_vector)
            if normalized_rotated_gravity_vector[2] < 0.995:
                self.recompute_pose()
            else:
                self.bool_find_z_ = True

    def recompute_pose(self):
        rotation = self.from_two_vectors(self.gravity_vector_, self.orientation_.apply([0, 0, 1]))
        self.orientation_ = self.orientation_ * rotation

    def from_two_vectors(self, v1: np.ndarray, v2: np.ndarray) -> R:
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        axis = np.cross(v1, v2)
        angle = np.arccos(np.dot(v1, v2))
        return R.from_rotvec(angle * axis / np.linalg.norm(axis))

    def ExtrapolatePosesWithGravity(self, times: List[Time]) -> Rigid3d:
        new_pose = self.ExtrapolatePose(times[-1])
        return Rigid3d(new_pose.translation, self.orientation_)