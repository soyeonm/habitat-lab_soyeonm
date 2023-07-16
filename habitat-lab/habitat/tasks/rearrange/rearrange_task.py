#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os.path as osp
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from gym import spaces

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorSuite
from habitat.tasks.nav.nav import NavigationTask
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import (
    CacheHelper,
    CollisionDetails,
    UsesArticulatedAgentInterface,
    rearrange_collision,
    rearrange_logger,
)


@registry.register_task(name="RearrangeEmptyTask-v0")
class RearrangeTask(NavigationTask):
    """
    Defines additional logic for valid collisions and gripping shared between
    all rearrangement tasks.
    """

    _cur_episode_step: int
    _articulated_agent_pos_start: Dict[str, Tuple[np.ndarray, float]]

    def _duplicate_sensor_suite(self, sensor_suite: SensorSuite) -> None:
        """
        Modifies the sensor suite in place to duplicate articulated agent specific sensors
        between the two articulated agents.
        """

        task_new_sensors: Dict[str, Sensor] = {}
        task_obs_spaces = OrderedDict()
        for agent_idx, agent_id in enumerate(self._sim.agents_mgr.agent_names):
            for sensor_name, sensor in sensor_suite.sensors.items():
                if isinstance(sensor, UsesArticulatedAgentInterface):
                    new_sensor = copy.copy(sensor)
                    new_sensor.agent_id = agent_idx
                    full_name = f"{agent_id}_{sensor_name}"
                    task_new_sensors[full_name] = new_sensor
                    task_obs_spaces[full_name] = new_sensor.observation_space
                else:
                    task_new_sensors[sensor_name] = sensor
                    task_obs_spaces[sensor_name] = sensor.observation_space

        sensor_suite.sensors = task_new_sensors
        sensor_suite.observation_spaces = spaces.Dict(spaces=task_obs_spaces)

    def __init__(
        self,
        *args,
        sim,
        dataset=None,
        should_place_articulated_agent=True,
        **kwargs,
    ) -> None:
        self.n_objs = len(dataset.episodes[0].targets)
        super().__init__(sim=sim, dataset=dataset, **kwargs)
        self.is_gripper_closed = False
        self._sim: RearrangeSim = sim
        self._ignore_collisions: List[Any] = []
        self._desired_resting = np.array(self._config.desired_resting_position)
        self._sim_reset = True
        self._targ_idx: int = 0
        self._episode_id: str = ""
        self._cur_episode_step = 0
        self._should_place_articulated_agent = should_place_articulated_agent
        self._fixed_starting_position: bool = False
        self._seed = self._sim.habitat_config.seed

        habitat_config = self._sim.habitat_config
        if "overfit" in habitat_config and habitat_config["overfit"]:
            self._fixed_starting_position = True

        data_path = dataset.config.data_path.format(split=dataset.config.split)
        fname = data_path.split("/")[-1].split(".")[0]
        cache_path = osp.join(
            osp.dirname(data_path),
            f"{fname}_{self._config.type}_robot_start.pickle",
        )

        if self._config.should_save_to_cache or osp.exists(cache_path):
            self._articulated_agent_init_cache = CacheHelper(
                cache_path,
                def_val={},
                verbose=False,
            )
            self._articulated_agent_pos_start = (
                self._articulated_agent_init_cache.load()
            )
        else:
            self._articulated_agent_pos_start = None

        if len(self._sim.agents_mgr) > 1:
            # Duplicate sensors that handle articulated agents. One for each articulated agent.
            self._duplicate_sensor_suite(self.sensor_suite)

    def overwrite_sim_config(self, config: Any, episode: Episode) -> Any:
        return config

    @property
    def targ_idx(self):
        return self._targ_idx

    @property
    def abs_targ_idx(self):
        if self._targ_idx is None:
            return None
        return self._sim.get_targets()[0][self._targ_idx]

    @property
    def desired_resting(self):
        return self._desired_resting

    def set_args(self, **kwargs):
        raise NotImplementedError("Task cannot dynamically set arguments")

    def set_sim_reset(self, sim_reset):
        self._sim_reset = sim_reset

    def _get_cached_articulated_agent_start(self, agent_idx: int = 0):
        start_ident = self._get_ep_init_ident(agent_idx)
        if (
            self._articulated_agent_pos_start is None
            or start_ident not in self._articulated_agent_pos_start
            or self._config.force_regenerate
        ):
            return None
        else:
            return self._articulated_agent_pos_start[start_ident]

    def _get_ep_init_ident(self, agent_idx):
        return f"{self._episode_id}_{agent_idx}"

    def _cache_articulated_agent_start(self, cache_data, agent_idx: int = 0):
        if (
            self._articulated_agent_pos_start is not None
            and self._config.should_save_to_cache
        ):
            start_ident = self._get_ep_init_ident(agent_idx)
            self._articulated_agent_pos_start[start_ident] = cache_data
            self._articulated_agent_init_cache.save(
                self._articulated_agent_pos_start
            )

    def _set_articulated_agent_start(self, agent_idx: int) -> None:
        # #breakpoint()
        # targ_object_loc = self._sim.scene_pos[agent_idx] 
        # #hum_object_loc = self._sim.scene_pos[1]

        # #sample until within 5 meters for each agent

        #Just fix navmesh error first

        #Get rot
        (_,articulated_agent_rot,) = self._sim.set_articulated_agent_base_to_random_point(agent_idx=agent_idx)

        #Copied from https://github.com/soyeonm/habitat-lab_soyeonm/blob/my_changes_SIRo_socnav_latest/habitat-lab/habitat/tasks/rearrange/actions/oracle_nav_action.py
        navigable_point = self._sim.pathfinder.get_random_navigable_point()
        _navmesh_vertices = np.stack(
            self._sim.pathfinder.build_navmesh_vertices(), axis=0
        )
        _island_sizes = [
            self._sim.pathfinder.island_radius(p) for p in _navmesh_vertices
        ]
        _max_island_size = max(_island_sizes)
        largest_size_vertex = _navmesh_vertices[
            np.argmax(_island_sizes)
        ]
        _largest_island_idx = self._sim.pathfinder.get_island(
            largest_size_vertex
        )

        start_pos = self._sim.pathfinder.get_random_navigable_point(
                island_index=_largest_island_idx
            )
        #self.cur_articulated_agent.sim_obj.translation = start_pos

        articulated_agent = self._sim.get_agent_data(agent_idx).articulated_agent
        articulated_agent.base_pos = start_pos #articulated_agent_pos
        articulated_agent.base_rot = articulated_agent_rot

    # def _set_articulated_agent_start(self, agent_idx: int) -> None:
    #     #oriiginal
    #     articulated_agent_start = self._get_cached_articulated_agent_start(
    #         agent_idx
    #     )
    #     if articulated_agent_start is None:
    #         (
    #             articulated_agent_pos,
    #             articulated_agent_rot,
    #         ) = self._sim.set_articulated_agent_base_to_random_point(
    #             agent_idx=agent_idx
    #         )
    #         self._cache_articulated_agent_start(
    #             (articulated_agent_pos, articulated_agent_rot), agent_idx
    #         )
    #     else:
    #         (
    #             articulated_agent_pos,
    #             articulated_agent_rot,
    #         ) = articulated_agent_start
    #     articulated_agent = self._sim.get_agent_data(
    #         agent_idx
    #     ).articulated_agent
    #     articulated_agent.base_pos = articulated_agent_pos
    #     articulated_agent.base_rot = articulated_agent_rot

    def reset(self, episode: Episode, fetch_observations: bool = True):
        self._episode_id = episode.episode_id
        self._ignore_collisions = []

        if self._sim_reset:
            self._sim.reset()
            for action_instance in self.actions.values():
                action_instance.reset(episode=episode, task=self)
            self._is_episode_active = True

            if self._should_place_articulated_agent:
                if self._fixed_starting_position:
                    np.random.seed(self._seed)
                    self._sim.pathfinder.seed(self._seed)

                for agent_idx in range(self._sim.num_articulated_agents):
                    self._set_articulated_agent_start(agent_idx)

        self.prev_measures = self.measurements.get_metrics()
        self._targ_idx = 0
        self.coll_accum = CollisionDetails()
        self.prev_coll_accum = CollisionDetails()
        self.should_end = False
        self._done = False
        self._cur_episode_step = 0
        if fetch_observations:
            self._sim.maybe_update_articulated_agent()
            return self._get_observations(episode)
        else:
            return None

    def _get_observations(self, episode):
        obs = self._sim.get_sensor_observations()
        obs = self._sim._sensor_suite.get_observations(obs)

        task_obs = self.sensor_suite.get_observations(
            observations=obs, episode=episode, task=self
        )
        obs.update(task_obs)
        return obs

    def _is_violating_safe_drop(self, action_args):
        idxs, goal_pos = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()
        target_pos = scene_pos[idxs]
        min_dist = np.min(
            np.linalg.norm(target_pos - goal_pos, ord=2, axis=-1)
        )
        return (
            self._sim.grasp_mgr.is_grasped
            and action_args.get("grip_action", None) is not None
            and action_args["grip_action"] < 0
            and min_dist < self._config.obj_succ_thresh
        )

    def step(self, action: Dict[str, Any], episode: Episode):
        action_args = action["action_args"]
        if self._config.enable_safe_drop and self._is_violating_safe_drop(
            action_args
        ):
            action_args["grip_action"] = None
        obs = super().step(action=action, episode=episode)

        self.prev_coll_accum = copy.copy(self.coll_accum)
        self._cur_episode_step += 1
        for grasp_mgr in self._sim.agents_mgr.grasp_iter:
            if (
                grasp_mgr.is_violating_hold_constraint()
                and self._config.constraint_violation_drops_object
            ):
                grasp_mgr.desnap(True)

        return obs

    def _check_episode_is_active(
        self,
        *args: Any,
        action: Union[int, Dict[str, Any]],
        episode: Episode,
        **kwargs: Any,
    ) -> bool:
        done = False
        if self.should_end:
            done = True

        # Check that none of the articulated agents are violating the hold constraint
        for grasp_mgr in self._sim.agents_mgr.grasp_iter:
            if (
                grasp_mgr.is_violating_hold_constraint()
                and self._config.constraint_violation_ends_episode
            ):
                done = True
                break

        if done:
            rearrange_logger.debug("-" * 10)
            rearrange_logger.debug("------ Episode Over --------")
            rearrange_logger.debug("-" * 10)

        return not done

    def get_coll_forces(self, articulated_agent_id):
        grasp_mgr = self._sim.get_agent_data(articulated_agent_id).grasp_mgr
        articulated_agent = self._sim.get_agent_data(
            articulated_agent_id
        ).articulated_agent
        snapped_obj = grasp_mgr.snap_idx
        articulated_agent_id = articulated_agent.sim_obj.object_id
        contact_points = self._sim.get_physics_contact_points()

        def get_max_force(contact_points, check_id):
            match_contacts = [
                x
                for x in contact_points
                if (check_id in [x.object_id_a, x.object_id_b])
                and (x.object_id_a != x.object_id_b)
            ]

            max_force = 0
            if len(match_contacts) > 0:
                max_force = max([abs(x.normal_force) for x in match_contacts])

            return max_force

        forces = [
            abs(x.normal_force)
            for x in contact_points
            if (
                x.object_id_a not in self._ignore_collisions
                and x.object_id_b not in self._ignore_collisions
            )
        ]
        max_force = max(forces) if len(forces) > 0 else 0

        max_obj_force = get_max_force(contact_points, snapped_obj)
        max_articulated_agent_force = get_max_force(
            contact_points, articulated_agent_id
        )
        return max_articulated_agent_force, max_obj_force, max_force

    def get_cur_collision_info(self, agent_idx) -> CollisionDetails:
        _, coll_details = rearrange_collision(
            self._sim, self._config.count_obj_collisions, agent_idx=agent_idx
        )
        return coll_details

    def get_n_targets(self) -> int:
        return self.n_objs

    @property
    def should_end(self) -> bool:
        return self._should_end

    @should_end.setter
    def should_end(self, new_val: bool):
        self._should_end = new_val
        ##
        # NB: _check_episode_is_active is called after step() but
        # before metrics are updated. Thus if should_end is set
        # by a metric, the episode will end on the _next_
        # step. This makes sure that the episode is ended
        # on the correct step.
        self._is_episode_active = (
            not self._should_end
        ) and self._is_episode_active
        if new_val:
            rearrange_logger.debug("-" * 40)
            rearrange_logger.debug(
                f"-----Episode {self._episode_id} requested to end after {self._cur_episode_step} steps.-----"
            )
            rearrange_logger.debug("-" * 40)
