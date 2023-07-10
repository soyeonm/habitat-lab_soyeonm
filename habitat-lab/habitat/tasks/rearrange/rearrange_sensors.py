#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import cv2
import numpy as np
from gym import spaces

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.nav.nav import PointGoalSensor
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import (
    CollisionDetails,
    UsesArticulatedAgentInterface,
    batch_transform_point,
    get_angle_to_pos,
    rearrange_logger,
)
from habitat.tasks.utils import cartesian_to_polar
import os
import pickle


class MultiObjSensor(PointGoalSensor):
    """
    Abstract parent class for a sensor that specifies the locations of all targets.
    """

    def __init__(self, *args, task, **kwargs):
        self._task = task
        self._sim: RearrangeSim
        super().__init__(*args, task=task, **kwargs)

    def _get_observation_space(self, *args, **kwargs):
        n_targets = self._task.get_n_targets()
        return spaces.Box(
            shape=(n_targets * 3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )


@registry.register_sensor
class TargetCurrentSensor(UsesArticulatedAgentInterface, MultiObjSensor):
    """
    This is the ground truth object position sensor relative to the robot end-effector coordinate frame.
    """

    cls_uuid: str = "obj_goal_pos_sensor"

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        self._sim: RearrangeSim
        T_inv = (
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
            .inverted()
        )

        idxs, _ = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()
        pos = scene_pos[idxs]

        for i in range(pos.shape[0]):
            pos[i] = T_inv.transform_point(pos[i])

        return pos.reshape(-1)


@registry.register_sensor
class TargetStartSensor(UsesArticulatedAgentInterface, MultiObjSensor):
    """
    Relative position from end effector to target object
    """

    cls_uuid: str = "obj_start_sensor"

    def get_observation(self, *args, observations, episode, **kwargs):
        self._sim: RearrangeSim
        global_T = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.ee_transform()
        T_inv = global_T.inverted()
        pos = self._sim.get_target_objs_start()
        return batch_transform_point(pos, T_inv, np.float32).reshape(-1)


class PositionGpsCompassSensor(UsesArticulatedAgentInterface, Sensor):
    def __init__(self, *args, sim, task, **kwargs):
        self._task = task
        self._sim = sim
        super().__init__(*args, task=task, **kwargs)

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        n_targets = self._task.get_n_targets()
        self._polar_pos = np.zeros(n_targets * 2, dtype=np.float32)
        return spaces.Box(
            shape=(n_targets * 2,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def _get_positions(self) -> np.ndarray:
        raise NotImplementedError("Must override _get_positions")

    def get_observation(self, task, *args, **kwargs):
        pos = self._get_positions()
        articulated_agent_T = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.base_transformation

        rel_pos = batch_transform_point(
            pos, articulated_agent_T.inverted(), np.float32
        )
        self._polar_pos = np.zeros_like(self._polar_pos)
        for i, rel_obj_pos in enumerate(rel_pos):
            rho, phi = cartesian_to_polar(rel_obj_pos[0], rel_obj_pos[1])
            self._polar_pos[(i * 2) : (i * 2) + 2] = [rho, -phi]

        return self._polar_pos


@registry.register_sensor
class TargetStartGpsCompassSensor(PositionGpsCompassSensor):
    cls_uuid: str = "obj_start_gps_compass"

    def _get_uuid(self, *args, **kwargs):
        return TargetStartGpsCompassSensor.cls_uuid

    def _get_positions(self) -> np.ndarray:
        return self._sim.get_target_objs_start()


@registry.register_sensor
class TargetGoalGpsCompassSensor(PositionGpsCompassSensor):
    cls_uuid: str = "obj_goal_gps_compass"

    def _get_uuid(self, *args, **kwargs):
        return TargetGoalGpsCompassSensor.cls_uuid

    def _get_positions(self) -> np.ndarray:
        _, pos = self._sim.get_targets()
        return pos


@registry.register_sensor
class AbsTargetStartSensor(MultiObjSensor):
    """
    Relative position from end effector to target object
    """

    cls_uuid: str = "abs_obj_start_sensor"

    def get_observation(self, observations, episode, *args, **kwargs):
        pos = self._sim.get_target_objs_start()
        return pos.reshape(-1)


@registry.register_sensor
class GoalSensor(UsesArticulatedAgentInterface, MultiObjSensor):
    """
    Relative to the end effector
    """

    cls_uuid: str = "obj_goal_sensor"

    def get_observation(self, observations, episode, *args, **kwargs):
        global_T = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.ee_transform()
        T_inv = global_T.inverted()

        _, pos = self._sim.get_targets()
        return batch_transform_point(pos, T_inv, np.float32).reshape(-1)


@registry.register_sensor
class AbsGoalSensor(MultiObjSensor):
    cls_uuid: str = "abs_obj_goal_sensor"

    def get_observation(self, *args, observations, episode, **kwargs):
        _, pos = self._sim.get_targets()
        return pos.reshape(-1)


@registry.register_sensor
class JointSensor(UsesArticulatedAgentInterface, Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return "joint"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(config.dimensionality,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        joints_pos = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.arm_joint_pos
        return np.array(joints_pos, dtype=np.float32)


@registry.register_sensor
class HumanoidJointSensor(UsesArticulatedAgentInterface, Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return "humanoid_joint_sensor"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(config.dimensionality,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        curr_agent = self._sim.get_agent_data(self.agent_id).articulated_agent
        if hasattr(curr_agent, "get_joint_transform"):
            joints_pos = curr_agent.get_joint_transform()[0]
            return np.array(joints_pos, dtype=np.float32)
        else:
            return np.zeros(self.observation_space.shape, dtype=np.float32)


@registry.register_sensor
class JointVelocitySensor(UsesArticulatedAgentInterface, Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return "joint_vel"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(config.dimensionality,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        joints_pos = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.arm_velocity
        return np.array(joints_pos, dtype=np.float32)


@registry.register_sensor
class EEPositionSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "ee_pos"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EEPositionSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        trans = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.base_transformation
        ee_pos = (
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
            .translation
        )
        local_ee_pos = trans.inverted().transform_point(ee_pos)

        return np.array(local_ee_pos, dtype=np.float32)


@registry.register_sensor
class RelativeRestingPositionSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "relative_resting_position"

    def _get_uuid(self, *args, **kwargs):
        return RelativeRestingPositionSensor.cls_uuid

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, task, *args, **kwargs):
        base_trans = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.base_transformation
        ee_pos = (
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
            .translation
        )
        local_ee_pos = base_trans.inverted().transform_point(ee_pos)

        relative_desired_resting = task.desired_resting - local_ee_pos

        return np.array(relative_desired_resting, dtype=np.float32)


@registry.register_sensor
class RestingPositionSensor(Sensor):
    """
    Desired resting position in the articulated_agent coordinate frame.
    """

    cls_uuid: str = "resting_position"

    def _get_uuid(self, *args, **kwargs):
        return RestingPositionSensor.cls_uuid

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, task, *args, **kwargs):
        return np.array(task.desired_resting, dtype=np.float32)


@registry.register_sensor
class LocalizationSensor(UsesArticulatedAgentInterface, Sensor):
    """
    The position and angle of the articulated_agent in world coordinates.
    """

    cls_uuid = "localization_sensor"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return LocalizationSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(4,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        articulated_agent = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent
        T = articulated_agent.base_transformation
        forward = np.array([1.0, 0, 0])
        heading_angle = get_angle_to_pos(T.transform_vector(forward))
        return np.array(
            [*articulated_agent.base_pos, heading_angle], dtype=np.float32
        )


@registry.register_sensor
class NavigationTargetPositionSensor(UsesArticulatedAgentInterface, Sensor):
    """
    To check if the agent is in the goal or not
    """

    cls_uuid = "navigation_target_position_sensor"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return NavigationTargetPositionSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(1,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        action_name = "oracle_nav_with_backing_up_action"
        task = kwargs["task"]
        if (
            "agent_"
            + str(self.agent_id)
            + "_oracle_nav_with_backing_up_action"
            in task.actions
        ):
            action_name = (
                "agent_"
                + str(self.agent_id)
                + "_oracle_nav_with_backing_up_action"
            )
        at_goal = task.actions[action_name].at_goal
        return np.array([at_goal])


@registry.register_sensor
class IsHoldingSensor(UsesArticulatedAgentInterface, Sensor):
    """
    Binary if the robot is holding an object or grasped onto an articulated object.
    """

    cls_uuid: str = "is_holding"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return IsHoldingSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(shape=(1,), low=0, high=1, dtype=np.float32)

    def get_observation(self, observations, episode, *args, **kwargs):
        return np.array(
            int(self._sim.get_agent_data(self.agent_id).grasp_mgr.is_grasped),
            dtype=np.float32,
        ).reshape((1,))


@registry.register_measure
class ObjectToGoalDistance(Measure):
    """
    Euclidean distance from the target object to the goal.
    """

    cls_uuid: str = "object_to_goal_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ObjectToGoalDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, **kwargs):
        idxs, goal_pos = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()
        target_pos = scene_pos[idxs]
        distances = np.linalg.norm(target_pos - goal_pos, ord=2, axis=-1)
        self._metric = {str(idx): dist for idx, dist in zip(idxs, distances)}


@registry.register_measure
class GfxReplayMeasure(Measure):
    cls_uuid: str = "gfx_replay_keyframes_string"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._enable_gfx_replay_save = (
            self._sim.sim_config.sim_cfg.enable_gfx_replay_save
        )
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return GfxReplayMeasure.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self._gfx_replay_keyframes_string = None
        self.update_metric(*args, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        if not task._is_episode_active and self._enable_gfx_replay_save:
            self._metric = (
                self._sim.gfx_replay_manager.write_saved_keyframes_to_string()
            )
        else:
            self._metric = ""

    def get_metric(self, force_get=False):
        if force_get and self._enable_gfx_replay_save:
            return (
                self._sim.gfx_replay_manager.write_saved_keyframes_to_string()
            )
        return super().get_metric()


@registry.register_measure
class ObjAtGoal(Measure):
    """
    Returns if the target object is at the goal (binary) for each of the target
    objects in the scene.
    """

    cls_uuid: str = "obj_at_goal"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._config = config
        self._succ_thresh = self._config.succ_thresh
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ObjAtGoal.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                ObjectToGoalDistance.cls_uuid,
            ],
        )
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        obj_to_goal_dists = task.measurements.measures[
            ObjectToGoalDistance.cls_uuid
        ].get_metric()

        self._metric = {
            str(idx): dist < self._succ_thresh
            for idx, dist in obj_to_goal_dists.items()
        }


@registry.register_measure
class EndEffectorToGoalDistance(UsesArticulatedAgentInterface, Measure):
    cls_uuid: str = "ee_to_goal_distance"

    def __init__(self, sim, *args, **kwargs):
        self._sim = sim
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EndEffectorToGoalDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, observations, **kwargs):
        ee_pos = (
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
            .translation
        )

        idxs, goals = self._sim.get_targets()

        distances = np.linalg.norm(goals - ee_pos, ord=2, axis=-1)

        self._metric = {str(idx): dist for idx, dist in zip(idxs, distances)}


@registry.register_measure
class EndEffectorToObjectDistance(UsesArticulatedAgentInterface, Measure):
    """
    Gets the distance between the end-effector and all current target object COMs.
    """

    cls_uuid: str = "ee_to_object_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EndEffectorToObjectDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, **kwargs):
        ee_pos = (
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
            .translation
        )

        idxs, _ = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()
        target_pos = scene_pos[idxs]

        distances = np.linalg.norm(target_pos - ee_pos, ord=2, axis=-1)

        self._metric = {str(idx): dist for idx, dist in zip(idxs, distances)}


@registry.register_measure
class EndEffectorToRestDistance(Measure):
    """
    Distance between current end effector position and position where end effector rests within the robot body.
    """

    cls_uuid: str = "ee_to_rest_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EndEffectorToRestDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, task, observations, **kwargs):
        to_resting = observations[RelativeRestingPositionSensor.cls_uuid]
        rest_dist = np.linalg.norm(to_resting)

        self._metric = rest_dist


@registry.register_measure
class ReturnToRestDistance(UsesArticulatedAgentInterface, Measure):
    """
    Distance between end-effector and resting position if the articulated agent is holding the object.
    """

    cls_uuid: str = "return_to_rest_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ReturnToRestDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, task, observations, **kwargs):
        to_resting = observations[RelativeRestingPositionSensor.cls_uuid]
        rest_dist = np.linalg.norm(to_resting)

        snapped_id = self._sim.get_agent_data(self.agent_id).grasp_mgr.snap_idx
        abs_targ_obj_idx = self._sim.scene_obj_ids[task.abs_targ_idx]
        picked_correct = snapped_id == abs_targ_obj_idx

        if picked_correct:
            self._metric = rest_dist
        else:
            T_inv = (
                self._sim.get_agent_data(self.agent_id)
                .articulated_agent.ee_transform()
                .inverted()
            )
            idxs, _ = self._sim.get_targets()
            scene_pos = self._sim.get_scene_pos()
            pos = scene_pos[idxs][0]
            pos = T_inv.transform_point(pos)

            self._metric = np.linalg.norm(task.desired_resting - pos)


@registry.register_measure
class RobotCollisions(UsesArticulatedAgentInterface, Measure):
    """
    Returns a dictionary with the counts for different types of collisions.
    """

    cls_uuid: str = "robot_collisions"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._sim = sim
        self._config = config
        self._task = task
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RobotCollisions.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._accum_coll_info = CollisionDetails()
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        cur_coll_info = self._task.get_cur_collision_info(self.agent_id)
        self._accum_coll_info += cur_coll_info
        self._metric = {
            "total_collisions": self._accum_coll_info.total_collisions,
            "robot_obj_colls": self._accum_coll_info.robot_obj_colls,
            "robot_scene_colls": self._accum_coll_info.robot_scene_colls,
            "obj_scene_colls": self._accum_coll_info.obj_scene_colls,
        }


@registry.register_measure
class RobotForce(UsesArticulatedAgentInterface, Measure):
    """
    The amount of force in newton's accumulatively applied by the robot.
    """

    cls_uuid: str = "articulated_agent_force"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._sim = sim
        self._config = config
        self._task = task
        self._count_obj_collisions = self._task._config.count_obj_collisions
        self._min_force = self._config.min_force
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RobotForce.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._accum_force = 0.0
        self._prev_force = None
        self._cur_force = None
        self._add_force = None
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    @property
    def add_force(self):
        return self._add_force

    def update_metric(self, *args, episode, task, observations, **kwargs):
        articulated_agent_force, _, overall_force = self._task.get_coll_forces(
            self.agent_id
        )
        if self._count_obj_collisions:
            self._cur_force = overall_force
        else:
            self._cur_force = articulated_agent_force

        if self._prev_force is not None:
            self._add_force = self._cur_force - self._prev_force
            if self._add_force > self._min_force:
                self._accum_force += self._add_force
                self._prev_force = self._cur_force
            elif self._add_force < 0.0:
                self._prev_force = self._cur_force
            else:
                self._add_force = 0.0
        else:
            self._prev_force = self._cur_force
            self._add_force = 0.0

        self._metric = {
            "accum": self._accum_force,
            "instant": self._cur_force,
        }


@registry.register_measure
class NumStepsMeasure(Measure):
    """
    The number of steps elapsed in the current episode.
    """

    cls_uuid: str = "num_steps"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return NumStepsMeasure.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._metric = 0

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._metric += 1


@registry.register_measure
class ForceTerminate(Measure):
    """
    If the accumulated force throughout this episode exceeds the limit.
    """

    cls_uuid: str = "force_terminate"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._sim = sim
        self._config = config
        self._max_accum_force = self._config.max_accum_force
        self._max_instant_force = self._config.max_instant_force
        self._task = task
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ForceTerminate.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                RobotForce.cls_uuid,
            ],
        )

        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        force_info = task.measurements.measures[
            RobotForce.cls_uuid
        ].get_metric()
        accum_force = force_info["accum"]
        instant_force = force_info["instant"]
        if self._max_accum_force > 0 and accum_force > self._max_accum_force:
            rearrange_logger.debug(
                f"Force threshold={self._max_accum_force} exceeded with {accum_force}, ending episode"
            )
            self._task.should_end = True
            self._metric = True
        elif (
            self._max_instant_force > 0
            and instant_force > self._max_instant_force
        ):
            rearrange_logger.debug(
                f"Force instant threshold={self._max_instant_force} exceeded with {instant_force}, ending episode"
            )
            self._task.should_end = True
            self._metric = True
        else:
            self._metric = False


@registry.register_measure
class DidViolateHoldConstraintMeasure(UsesArticulatedAgentInterface, Measure):
    cls_uuid: str = "did_violate_hold_constraint"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return DidViolateHoldConstraintMeasure.cls_uuid

    def __init__(self, *args, sim, **kwargs):
        self._sim = sim

        super().__init__(*args, sim=sim, **kwargs)

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, **kwargs):
        self._metric = self._sim.get_agent_data(
            self.agent_id
        ).grasp_mgr.is_violating_hold_constraint()


class RearrangeReward(UsesArticulatedAgentInterface, Measure):
    """
    An abstract class defining some measures that are always a part of any
    reward function in the Habitat 2.0 tasks.
    """

    def __init__(self, *args, sim, config, task, **kwargs):
        self._sim = sim
        self._config = config
        self._task = task
        self._force_pen = self._config.force_pen
        self._max_force_pen = self._config.max_force_pen
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                RobotForce.cls_uuid,
                ForceTerminate.cls_uuid,
            ],
        )

        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        reward = 0.0

        reward += self._get_coll_reward()

        if self._sim.get_agent_data(
            self.agent_id
        ).grasp_mgr.is_violating_hold_constraint():
            reward -= self._config.constraint_violate_pen

        force_terminate = task.measurements.measures[
            ForceTerminate.cls_uuid
        ].get_metric()
        if force_terminate:
            reward -= self._config.force_end_pen

        self._metric = reward

    def _get_coll_reward(self):
        reward = 0

        force_metric = self._task.measurements.measures[RobotForce.cls_uuid]
        # Penalize the force that was added to the accumulated force at the
        # last time step.
        reward -= max(
            0,  # This penalty is always positive
            min(
                self._force_pen * force_metric.add_force,
                self._max_force_pen,
            ),
        )
        return reward


@registry.register_measure
class DoesWantTerminate(Measure):
    """
    Returns 1 if the agent has called the stop action and 0 otherwise.
    """

    cls_uuid: str = "does_want_terminate"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return DoesWantTerminate.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self.update_metric(*args, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        self._metric = task.actions["rearrange_stop"].does_want_terminate


@registry.register_measure
class BadCalledTerminate(Measure):
    """
    Returns 0 if the agent has called the stop action when the success
    condition is also met or not called the stop action when the success
    condition is not met. Returns 1 otherwise.
    """

    cls_uuid: str = "bad_called_terminate"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return BadCalledTerminate.cls_uuid

    def __init__(self, config, task, *args, **kwargs):
        super().__init__(**kwargs)
        self._success_measure_name = task._config.success_measure
        self._config = config

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [DoesWantTerminate.cls_uuid, self._success_measure_name],
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        does_action_want_stop = task.measurements.measures[
            DoesWantTerminate.cls_uuid
        ].get_metric()
        is_succ = task.measurements.measures[
            self._success_measure_name
        ].get_metric()

        self._metric = (not is_succ) and does_action_want_stop


@registry.register_sensor
class HasFinishedOracleNavSensor(UsesArticulatedAgentInterface, Sensor):
    """
    Returns 1 if the agent has finished the oracle nav action. Returns 0 otherwise.
    """

    cls_uuid: str = "has_finished_oracle_nav"

    def __init__(self, sim, config, *args, task, **kwargs):
        self._task = task
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return HasFinishedOracleNavSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(shape=(1,), low=0, high=1, dtype=np.float32)

    def get_observation(self, observations, episode, *args, **kwargs):
        if self.agent_id is not None:
            use_k = f"agent_{self.agent_id}_oracle_nav_action"
            if (
                f"agent_{self.agent_id}_oracle_nav_with_backing_up_action"
                in self._task.actions
            ):
                use_k = (
                    f"agent_{self.agent_id}_oracle_nav_with_backing_up_action"
                )
        else:
            use_k = "oracle_nav_action"
            if "oracle_nav_with_backing_up_action" in self._task.actions:
                use_k = "oracle_nav_with_backing_up_action"

        nav_action = self._task.actions[use_k]

        return np.array(nav_action.skill_done, dtype=np.float32)[..., None]


@registry.register_measure
class ContactTestStats(Measure):
    """
    Did agent collide with objects?
    """

    cls_uuid: str = "contact_test_stats"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(**kwargs)
        self._sim = sim
        self._config = config

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ContactTestStats.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        self._contact_flag = []
        self._metric = 0

    def update_metric(self, *args, episode, task, observations, **kwargs):
        flag = self._sim.contact_test(
            self._sim.articulated_agent.get_robot_sim_id()
        )
        self._contact_flag.append(flag)
        self._metric = np.average(self._contact_flag)


@registry.register_measure
class PanopticCalculator(UsesArticulatedAgentInterface, Measure):
    """
    Ratio of episodes the robot is able to find the person.
    """

    cls_uuid: str = "panoptic_calculator"

    def __init__(self, sim, config, config_entire, *args, **kwargs):
        #breakpoint()
        super().__init__(**kwargs)
        self._sim = sim
        self._config = config
        self.entire_config = config_entire
        #self.save_dir = self.entire_config['save_dir']
        self.episode_idx =-1

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return PanopticCalculator.cls_uuid

    def get_obj_id_2_handle(self):
        id2handle_dict = {}
        rom = self._sim.get_rigid_object_manager()
        handles = rom.get_object_handles()
        for handle in handles:
            # if (
            #     "agent_0_oracle_nav_with_backing_up_action"
            #     in self.step_action_set
            # ):  # This means it's floorplanner
            # if True:
            any_target_handle = handle  # remove the '_:0000'
            if any_target_handle in [
                self.any_target0_handle,
                self.any_target1_handle,
            ]:
                # In floorplanner, handles look like '004_sugar_box_:0000' or '0164a753999c91217e819b52f8f354b3f60ded96_:0000' (seems inconsistent)
                obj = rom.get_object_by_handle(handle)
                objid = (
                    obj.object_id
                    + self._sim.habitat_config.object_ids_start
                )
                id2handle_dict[objid] = any_target_handle
                if any_target_handle == self.any_target1_handle:
                    self.target_1_entry = objid
        #Add human
        self.human_entry = 102
        id2handle_dict[self.human_entry] = "human"

        return id2handle_dict

    def reset_metric(self, *args, task, **kwargs):
        #save to pickle before updating
        if self.episode_idx >=0:
            pickle.dump(self.stats_dict, open(self.save_dir + '/stats_dict.p', 'wb'))
            breakpoint()

        self.episode_idx +=1
        self.step_count = 0
        self.human_poses = []
        # self.human_holding = False
        # self.prev_human_holding = False

        self.state = 'beginning'
        self.robot_state = 'beginning'

        #beginning list
        self.state_list = []
        self.gt_human_visible_list = []
        self.gt_target_1_visible_list = []


        #start log

        self.update_metric(*args, task=task, **kwargs)


    def set_before_middle_end(self, agent_1_holding):
        # self.prev_human_holding = self.human_holding
        # self.human_holding = agent_1_holding

        #if not(self.prev_human_holding) and not(self.human_holding):
            #beginning

        if self.state == 'beginning' and agent_1_holding:
            self.state = 'middle'
        elif self.state == 'middle' and not(agent_1_holding):
            self.state = 'end'

    def agent0_set_before_middle_end(self, agent_0_holding):
        # self.prev_human_holding = self.human_holding
        # self.human_holding = agent_1_holding

        #if not(self.prev_human_holding) and not(self.human_holding):
            #beginning

        if self.robot_state == 'beginning' and agent_0_holding:
            self.robot_state  = 'middle'
        elif self.robot_state == 'middle' and not(agent_0_holding):
            self.robot_state  = 'end'

    def save_rgb(self, rgb):
        file_name = os.path.join(self.save_dir, 'rgb', str(self.step_count) + ".png")
        if not os.path.exists(os.path.join(self.save_dir, 'rgb')):
            os.makedirs(os.path.join(self.save_dir, 'rgb'))
        cv2.imwrite(file_name, cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))

        #if cur targ visible, save again 
        if self.cur_human_visible:
            file_name = os.path.join(self.save_dir, 'targ_viz', str(self.step_count) + ".png")
            if not os.path.exists(os.path.join(self.save_dir, 'targ_viz')):
                os.makedirs(os.path.join(self.save_dir, 'targ_viz'))
            cv2.imwrite(file_name, cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))

        viz_mask = self.target_1_mask #bool
        if np.sum(viz_mask) >0:
            file_name = os.path.join(self.save_dir, 'targ_viz', str(self.step_count) + "_mask.png")
            if not os.path.exists(os.path.join(self.save_dir, 'targ_viz')):
                os.makedirs(os.path.join(self.save_dir, 'targ_viz'))
            cv2.imwrite(file_name, viz_mask.astype(np.uint8)*255)

        #if cur targ visible, save again 
        if self.cur_human_visible:
            file_name = os.path.join(self.save_dir, 'hum_viz', str(self.step_count) + ".png")
            if not os.path.exists(os.path.join(self.save_dir, 'hum_viz')):
                os.makedirs(os.path.join(self.save_dir, 'hum_viz'))
            cv2.imwrite(file_name, cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))

        viz_mask = self.human_mask  #bool
        if np.sum(viz_mask) >0:
            file_name = os.path.join(self.save_dir, 'hum_viz', str(self.step_count) + "_mask.png")
            if not os.path.exists(os.path.join(self.save_dir, 'hum_viz')):
                os.makedirs(os.path.join(self.save_dir, 'hum_viz'))
            cv2.imwrite(file_name, viz_mask.astype(np.uint8)*255)


    def save_rgb_human(self, rgb):
        file_name = os.path.join(self.save_dir, 'rgb_human', str(self.step_count) + ".png")
        if not os.path.exists(os.path.join(self.save_dir, 'rgb_human')):
            os.makedirs(os.path.join(self.save_dir, 'rgb_human'))
        cv2.imwrite(file_name, cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))

    def any_target_1_visible_gt(self, panoptic):
        # see if anytarget 1 is visible in the current frame
        #any_target1_id = self.any_target_ids_to_names[self.any_target1_handle]
        self.target_1_mask = panoptic[:,:,0]==self.target_1_entry
        # if np.sum(self.target_1_mask) >0:
        #     print("Step ", self.step_count, ", target 1 visible!")
        self.cur_targ_visible = np.sum(self.target_1_mask) >0
        return np.sum(self.target_1_mask) >0

    def human_visible_gt(self, panoptic):
        # see if anytarget 1 is visible in the current frame
        self.human_mask = panoptic[:,:,0]==self.human_entry
        # if np.sum(self.human_mask) >0:
        #     print("Step ", self.step_count, ", human visible!")
        self.cur_human_visible = np.sum(self.human_mask) >0
        return np.sum(self.human_mask) >0

    #Don't do fancy 
    def visualize_gt_seg(self):
        viz_mask = self.human_mask + self.target_1_mask #bool
        if np.sum(viz_mask) >0:
            file_name = os.path.join(self.save_dir, 'viz_human_target_1', str(self.step_count) + ".png")
            if not os.path.exists(os.path.join(self.save_dir, 'viz_human_target_1')):
                os.makedirs(os.path.join(self.save_dir, 'viz_human_target_1'))
            cv2.imwrite(file_name, viz_mask.astype(np.uint8)*255)

    #Don't do fancy
    def visualize_detectron(self):
        pass

    def log(self):
        self.file = open(self.save_dir + '/log.txt', 'a')
        self.file.write('step: ' + str(self.step_count) + "\n")
        self.file.write("state: " + str(self.state)+ "\n")
        if np.sum(self.target_1_mask) >0:
            self.file.write("target 1 visible!"+ "\n")
        if np.sum(self.human_mask) >0:
            self.file.write("human visible!"+ "\n")
        self.file.write(str(self.stats_dict) + "\n")
        self.file.close()
        
    def _divide(self, a,b):
        if b ==0:
            return 0
        else:
            return a/b

    def _get_stat(self, viz_obj, state):
        assert state in ['beginning', 'middle', 'end']
        if viz_obj == 'human':
            list_of_interest = self.gt_human_visible_list
        elif viz_obj == 'targ1':
            list_of_interest = self.gt_target_1_visible_list
        else:
            raise Exception("Wrong")

        stat = self._divide(np.sum((np.array(self.state_list) == state) * np.array(list_of_interest)), np.sum(np.array(self.state_list) == state))
        return stat


    # Also save rgb here
    def update_metric(self, *args, episode, task, observations, **kwargs):
        if self.step_count ==0:
            self.ep_info = self._sim.get_agent_data(0).articulated_agent._sim.ep_info
            for k, v in self.ep_info.info["object_labels"].items():
                if v == "any_targets|0":
                    self.any_target0_handle = k
                elif v == "any_targets|1":
                    self.any_target1_handle = k

            self.any_target_ids_to_names = self.get_obj_id_2_handle()
            self.episode_id = self.ep_info.episode_id
            self.save_dir = os.path.join(self.entire_config['save_dir'], self.episode_id)

            #Start a log
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            self.file = open(self.save_dir + '/log.txt', 'w')
            self.file.close()

        panoptic = self._sim._sensor_suite.get_observations(self._sim.get_sensor_observations())["agent_0_articulated_agent_arm_panoptic"]
        rgb = self._sim._sensor_suite.get_observations(self._sim.get_sensor_observations())["agent_0_articulated_agent_arm_rgb"]
        # ep_info = self._sim.get_agent_data(0).articulated_agent._sim.ep_info

        agent_1_holding = self._sim.get_agent_data(1).grasp_mgr.is_grasped #observations['agent_1_is_holding']
        print("agent 1 holding", agent_1_holding)
        #print("step ",self.step_count, " agent 1 holding ", agent_1_holding)
        #print("snap idx ", self._sim.get_agent_data(1).grasp_mgr.snap_idx)
        self.set_before_middle_end(agent_1_holding) #state beginning, middle, end
        agent_0_holding = self._sim.get_agent_data(0).grasp_mgr.is_grasped 
        self.agent0_set_before_middle_end(agent_0_holding)
        #breakpoint()

        self.state_list.append(self.state)
        self.gt_human_visible_list.append(self.human_visible_gt(panoptic))
        self.gt_target_1_visible_list.append(self.any_target_1_visible_gt(panoptic))
        #breakpoint()

        #Calculate stats
        #Beginning stats
        # beginning_gt_human_visible = self._divide(np.sum((np.array(self.state_list) == 'beginning') * np.array(self.gt_human_visible_list)), np.sum(np.array(self.state_list) == 'beginning'))
        # middle_gt_human_visible = self._divide(np.sum(np.array(self.state_list) == 'middle' * np.array(self.gt_human_visible_list)), np.sum(np.array(self.state_list) == 'middle'))
        # end_gt_human_visible = self._divide(np.sum(np.array(self.state_list == 'end') * np.array(self.gt_human_visible_list)), np.sum(np.array(self.state_list) == 'end'))

        # beginning_gt_target1_visible = self._divide(np.sum(np.array(self.state_list == 'beginning') * np.array(self.gt_target_1_visible_list)), np.sum(np.array(self.state_list) == 'beginning'))
        # middle_gt_target1_visible = self._divide(np.sum(np.array(self.state_list == 'middle') * np.array(self.gt_target_1_visible_list)), np.sum(np.array(self.state_list) == 'middle'))
        # end_gt_target1_visible = self._divide(np.sum(np.array(self.state_list == 'end') * np.array(self.gt_target_1_visible_list)), np.sum(np.array(self.state_list) == 'end'))

        beginning_gt_human_visible = self._get_stat('human', 'beginning')
        middle_gt_human_visible = self._get_stat('human', 'middle') 
        end_gt_human_visible = self._get_stat('human', 'end') 

        beginning_gt_target1_visible = self._get_stat('targ1', 'beginning') 
        middle_gt_target1_visible = self._get_stat('targ1', 'middle') 
        end_gt_target1_visible = self._get_stat('targ1', 'end') 
        

        
        self.stats_dict = {'beginning_gt_human_visible': beginning_gt_human_visible,
                    'middle_gt_human_visible': middle_gt_human_visible,
                    'end_gt_human_visible': end_gt_human_visible,
                    'beginning_gt_target1_visible': beginning_gt_target1_visible,
                    'middle_gt_target1_visible': middle_gt_target1_visible,
                    'end_gt_target1_visible': end_gt_target1_visible,
                    'valid_episod': self.state == 'end' and self.robot_state =='end',
                    'gt_human_visible_list':self.gt_human_visible_list,
                    'gt_target_1_visible_list': self.gt_target_1_visible_list} #Make sure it ended with end

        #print("stats dict is ", stats_dict)
        self.save_rgb(rgb) 
        #breakpoint()
        self.save_rgb_human(self._sim._sensor_suite.get_observations(self._sim.get_sensor_observations())["agent_1_third_rgb"]) 
        self.visualize_gt_seg()
        print("step ", self.step_count)
        self.step_count +=1
        if self.step_count >= 1000:
            task.should_end = True
        
        self.log()
        self._metric = 0
