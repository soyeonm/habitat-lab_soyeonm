# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import magnum as mn
import numpy as np
from gym import spaces

import habitat_sim
from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.rearrange.actions.actions import (
    BaseVelAction,
    BaseVelNonCylinderAction,
    HumanoidJointAction,
)
from habitat.tasks.rearrange.utils import place_agent_at_dist_from_pos, place_robot_at_closest_point, place_robot_at_closest_point_for_sem_map, place_robot_at_closest_point_for_sem_map_with_navmesh
from habitat.tasks.utils import get_angle
from habitat_sim.physics import VelocityControl
import copy

from habitat.tiffany_utils.navmesh_utils import get_largest_island_index


@registry.register_task_action
class OracleNavAction(BaseVelAction, HumanoidJointAction):
    """
    An action that will convert the index of an entity (in the sense of
    `PddlEntity`) to navigate to and convert this to base/humanoid joint control to move the
    robot to the closest navigable position to that entity. The entity index is
    the index into the list of all available entities in the current scene. The
    config flag motion_type indicates whether the low level action will be a base_velocity or
    a joint control.
    """

    def __init__(self, *args, task, **kwargs):
        config = kwargs["config"]
        self.motion_type = config.motion_control
        if self.motion_type == "base_velocity":
            BaseVelAction.__init__(self, *args, **kwargs)

        elif self.motion_type == "human_joints":
            HumanoidJointAction.__init__(self, *args, **kwargs)
            self.humanoid_controller = self.lazy_inst_humanoid_controller(
                task, config
            )

        else:
            raise ValueError("Unrecognized motion type for oracle nav  action")

        self._task = task
        self._poss_entities = (
            self._task.pddl_problem.get_ordered_entities_list()
        )
        self._prev_ep_id = None
        self._targets = {}
        self.skill_done = False
        self._spawn_max_dist_to_obj = self._config.spawn_max_dist_to_obj
        self._num_spawn_attempts = self._config.num_spawn_attempts
        self._dist_thresh = self._config.dist_thresh
        self._ori_dist_thresh = self._config.dist_thresh
        self._turn_thresh = self._config.turn_thresh
        self._turn_velocity = self._config.turn_velocity
        self._forward_velocity = self._config.forward_velocity
        self.ep_num = -1

    @staticmethod
    def _compute_turn(rel, turn_vel, robot_forward):
        is_left = np.cross(robot_forward, rel) > 0
        if is_left:
            vel = [0, -turn_vel]
        else:
            vel = [0, turn_vel]
        return vel

    def lazy_inst_humanoid_controller(self, task, config):
        # Lazy instantiation of humanoid controller
        # We assign the task with the humanoid controller, so that multiple actions can
        # use it.

        if (
            not hasattr(task, "humanoid_controller")
            or task.humanoid_controller is None
        ):
            # Initialize humanoid controller
            agent_name = self._sim.habitat_config.agents_order[
                self._agent_index
            ]
            walk_pose_path = self._sim.habitat_config.agents[
                agent_name
            ].motion_data_path

            humanoid_controller = HumanoidRearrangeController(walk_pose_path)
            humanoid_controller.set_framerate_for_linspeed(
                config["lin_speed"], config["ang_speed"], self._sim.ctrl_freq
            )
            task.humanoid_controller = humanoid_controller
        return task.humanoid_controller

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "oracle_nav_action": spaces.Box(
                    shape=(1,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        if self._task._episode_id != self._prev_ep_id:
            self._targets = {}
            self._prev_ep_id = self._task._episode_id
        self.skill_done = False
        self.prev_obj_targ_pos = np.array([np.nan, np.nan, np.nan])
        self.prev_final_nav_targ = np.array([np.nan, np.nan, np.nan])
        self.at_goal_prev = False
        self.last_vel = [0.0, 0.0, 0.0]
        self.timestep = 0
        self.last_rot = None
        self.forward_side_step_try = -1
        self.spot_init_failed = False
        self.ep_num +=1

    def _get_target_for_idx(self, nav_to_target_idx: int):
        nav_to_obj = self._poss_entities[nav_to_target_idx]
        if (
            nav_to_target_idx not in self._targets
            or "robot" in nav_to_obj.name
        ):
            obj_pos = self._task.pddl_problem.sim_info.get_entity_pos(
                nav_to_obj
            )
            if "robot" in nav_to_obj.name:
                # Safety margin between the human and the robot
                sample_distance = 1.0
            else:
                sample_distance = self._spawn_max_dist_to_obj
            start_pos, _, _ = place_agent_at_dist_from_pos(
                np.array(obj_pos),
                0.0,
                sample_distance,
                self._sim,
                self._num_spawn_attempts,
                1,
                self.cur_articulated_agent,
            )

            if self.motion_type == "human_joints":
                self.humanoid_controller.reset(
                    self.cur_articulated_agent.base_transformation
                )
            self._targets[nav_to_target_idx] = (start_pos, np.array(obj_pos))
        return self._targets[nav_to_target_idx]

    def found_path(self, point):
        agent_pos = self.cur_articulated_agent.base_pos

        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = point
        found_path = self._sim.pathfinder.find_path(path)
        return found_path

    def _path_to_point(self, point):
        """
        Obtain path to reach the coordinate point. If agent_pos is not given
        the path starts at the agent base pos, otherwise it starts at the agent_pos
        value
        :param point: Vector3 indicating the target point
        """
        agent_pos = self.cur_articulated_agent.base_pos

        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = point
        found_path = self._sim.pathfinder.find_path(path)
        if not found_path:
            return [agent_pos, point]
        return path.points

    def _update_controller_to_navmesh(self):
        base_offset = self.cur_articulated_agent.params.base_offset
        prev_query_pos = self.cur_articulated_agent.base_pos
        target_query_pos = (
            self.humanoid_controller.obj_transform_base.translation
            + base_offset
        )

        filtered_query_pos = self._sim.step_filter(
            prev_query_pos, target_query_pos
        )
        fixup = filtered_query_pos - target_query_pos
        self.humanoid_controller.obj_transform_base.translation += fixup

    def step(self, *args, is_last_action, **kwargs):
        # if (self._action_arg_prefix + "just_rotate" in kwargs) and kwargs[self._action_arg_prefix + "just_rotate"] != 0.0:
        #     self.dist_thresh = 1.35 #Just set it to this
        #     breakpoint()
        # else:
        #     self.dist_thresh = self._ori_dist_thresh


        self.skill_done = False
        nav_to_target_idx = kwargs[
            self._action_arg_prefix + "oracle_nav_action"
        ]
        if nav_to_target_idx <= 0 or nav_to_target_idx > len(
            self._poss_entities
        ):
            if is_last_action:
                return self._sim.step(HabitatSimActions.base_velocity)
            else:
                return {}

        if isinstance(nav_to_target_idx, np.ndarray) and nav_to_target_idx[0] != np.inf: #original
            move_freely = False
            nav_to_target_idx = int(nav_to_target_idx[0]) - 1
            final_nav_targ, obj_targ_pos = self._get_target_for_idx(
                nav_to_target_idx
            )
            # Get the current path
            curr_path_points = self._path_to_point(final_nav_targ)
        elif isinstance(nav_to_target_idx, np.ndarray) and nav_to_target_idx[0]==np.inf:
            return
        else: #move freely with ogn
            move_freely = True
            action_to_take = kwargs[self._action_arg_prefix + "oracle_nav_action"]

        # nav_to_target_idx = int(nav_to_target_idx[0]) - 1

        # final_nav_targ, obj_targ_pos = self._get_target_for_idx(
        #     nav_to_target_idx
        # )
        base_T = self.cur_articulated_agent.base_transformation
        #curr_path_points = self._path_to_point(final_nav_targ)
        robot_pos = np.array(self.cur_articulated_agent.base_pos)

        if move_freely: 
            if action_to_take == 1:  # forward
                vel = [self._config.forward_velocity, 0]
            elif action_to_take == 2:  # turn left #Just changed for oGN
                vel = [0, self._config.turn_velocity] #[0, -self._config.turn_velocity]
            elif action_to_take == 3:  # turn right
                vel = [0, -self._config.turn_velocity] #[0, self._config.turn_velocity]
            else:  # stop
                vel = [0, 0]
            # else:
            #     vel = [0, 0]
            #self.skill_done = True
            kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
            return BaseVelAction.step(
                self, *args, is_last_action=is_last_action, **kwargs
            )

        else:
            if curr_path_points is None:
                raise Exception
            else:
                # Compute distance and angle to target
                if len(curr_path_points) == 1:
                    curr_path_points += curr_path_points
                cur_nav_targ = curr_path_points[1]
                forward = np.array([1.0, 0, 0])
                robot_forward = np.array(base_T.transform_vector(forward))

                # Compute relative target.
                rel_targ = cur_nav_targ - robot_pos

                # Compute heading angle (2D calculation)
                robot_forward = robot_forward[[0, 2]]
                rel_targ = rel_targ[[0, 2]]
                rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]

                angle_to_target = get_angle(robot_forward, rel_targ)
                angle_to_obj = get_angle(robot_forward, rel_pos)

                dist_to_final_nav_targ = np.linalg.norm(
                    (final_nav_targ - robot_pos)[[0, 2]]
                )
                at_goal = (
                    dist_to_final_nav_targ < self._dist_thresh
                    and angle_to_obj < self._turn_thresh
                ) or dist_to_final_nav_targ < self._dist_thresh / 10.0

                if self.motion_type == "base_velocity":
                    if not at_goal:
                        if dist_to_final_nav_targ < self._dist_thresh:
                            # Look at the object
                            vel = OracleNavAction._compute_turn(
                                rel_pos, self._turn_velocity, robot_forward
                            )
                        elif angle_to_target < self._turn_thresh:
                            # Move towards the target
                            vel = [self._forward_velocity, 0]
                        else:
                            # Look at the target waypoint.
                            vel = OracleNavAction._compute_turn(
                                rel_targ, self._turn_velocity, robot_forward
                            )
                    else:
                        vel = [0, 0]
                        self.skill_done = True
                    kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)

                    return BaseVelAction.step(
                        self, *args, is_last_action=is_last_action, **kwargs
                    )

                elif self.motion_type == "human_joints":
                    # Update the humanoid base
                    self.humanoid_controller.obj_transform_base = base_T
                    if not at_goal:
                        if dist_to_final_nav_targ < self._dist_thresh:
                            # Look at the object
                            self.humanoid_controller.calculate_turn_pose(
                                mn.Vector3([rel_pos[0], 0.0, rel_pos[1]])
                            )
                        else:
                            # Move towards the target
                            self.humanoid_controller.calculate_walk_pose(
                                mn.Vector3([rel_targ[0], 0.0, rel_targ[1]])
                            )
                    else:
                        self.humanoid_controller.calculate_stop_pose()
                        self.skill_done = True
                    self._update_controller_to_navmesh()
                    base_action = self.humanoid_controller.get_pose()
                    kwargs[
                        f"{self._action_arg_prefix}human_joints_trans"
                    ] = base_action

                    return HumanoidJointAction.step(
                        self, *args, is_last_action=is_last_action, **kwargs
                    )
                else:
                    raise ValueError(
                        "Unrecognized motion type for oracle nav action"
                    )


class SimpleVelocityControlEnv:
    """
    Simple velocity control environment for moving agent
    """

    def __init__(self, sim_freq=120.0):
        # the velocity control
        self.vel_control = VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True
        self._sim_freq = sim_freq

    def act(self, trans, vel):
        linear_velocity = vel[0]
        angular_velocity = vel[1]
        # Map velocity actions
        self.vel_control.linear_velocity = mn.Vector3(
            [linear_velocity, 0.0, 0.0]
        )
        self.vel_control.angular_velocity = mn.Vector3(
            [0.0, angular_velocity, 0.0]
        )
        # Compute the rigid state
        rigid_state = habitat_sim.RigidState(
            mn.Quaternion.from_matrix(trans.rotation()), trans.translation
        )
        # Get the target rigit state based on the simulation frequency
        target_rigid_state = self.vel_control.integrate_transform(
            1 / self._sim_freq, rigid_state
        )
        # Get the ending pos of the agent
        end_pos = target_rigid_state.translation
        # Offset the height
        end_pos[1] = trans.translation[1]
        # Construct the target trans
        target_trans = mn.Matrix4.from_(
            target_rigid_state.rotation.to_matrix(),
            target_rigid_state.translation,
        )

        return target_trans


@registry.register_task_action
class OracleNavWithBackingUpAction(BaseVelNonCylinderAction, OracleNavAction):  # type: ignore
    """
    Oracle nav action with backing-up. This function allows the robot to move
    backward to avoid obstacles.
    """

    def __init__(self, *args, task, **kwargs):
        OracleNavAction.__init__(self, *args, task=task, **kwargs)
        if self.motion_type == "base_velocity":
            BaseVelNonCylinderAction.__init__(self, *args, **kwargs, task=task)

        # Define the navigation target
        self.at_goal = False
        self.skill_done = False
        self._navmesh_offset_for_agent_placement = (
            self._config.navmesh_offset_for_agent_placement
        )
        self._navmesh_offset = self._config.navmesh_offset
        #breakpoint()
        #self._navmesh_offset_for_agent_placement = ([[0.0, 0.0], [0.15, 0.0], [-0.15, 0.0]])

        self._nav_pos_3d = [
            np.array([xz[0], 0.0, xz[1]]) for xz in self._navmesh_offset
        ]

        # Initialize the velocity controller
        self._vc = SimpleVelocityControlEnv(self._config.sim_freq)

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "oracle_nav_with_backing_up_action": spaces.Box(
                    shape=(1,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )

    def _get_target_for_idx(self, nav_to_target_idx: int):
        if nav_to_target_idx not in self._targets:
            nav_to_obj = self._poss_entities[nav_to_target_idx]
            obj_pos = self._task.pddl_problem.sim_info.get_entity_pos(
                nav_to_obj
            )
            start_pos, _, _ = place_agent_at_dist_from_pos(
                np.array(obj_pos),
                0.0,
                self._spawn_max_dist_to_obj,
                self._sim,
                self._num_spawn_attempts,
                1,
                self.cur_articulated_agent,
                self._navmesh_offset_for_agent_placement,
            )

            if self.motion_type == "human_joints":
                self.humanoid_controller.reset(
                    self.cur_articulated_agent.base_transformation
                )
            self._targets[nav_to_target_idx] = (start_pos, np.array(obj_pos))
        return self._targets[nav_to_target_idx]

    def is_collision(self, trans) -> bool:
        """
        The function checks if the agent collides with the object
        given the navmesh
        """
        cur_pos = [trans.transform_point(xyz) for xyz in self._nav_pos_3d]
        cur_pos = [
            np.array([xz[0], self.cur_articulated_agent.base_pos[1], xz[2]])
            for xz in cur_pos
        ]

        for pos in cur_pos:  # noqa: SIM110
            # Return true if the pathfinder says it is not navigable
            if not self._sim.pathfinder.is_navigable(pos):
                return True

        return False


    def forward_collision_check(
        self,
        next_pos,
    ):
        """
        This function checks if the robot needs to do backing-up action
        """
        # Make a copy of agent trans
        trans = mn.Matrix4(self.cur_articulated_agent.sim_obj.transformation)
        angle = float("inf")
        # Get the current location of the agent
        cur_pos = self.cur_articulated_agent.base_pos
        # Set the trans to be agent location
        trans.translation = self.cur_articulated_agent.base_pos

        #while abs(angle) > self._turn_thresh:
        # Compute the robot facing orientation
        rel_pos = (next_pos - cur_pos)[[0, 2]]
        forward = np.array([1.0, 0, 0])
        robot_forward = np.array(trans.transform_vector(forward))
        robot_forward = robot_forward[[0, 2]]
        angle = get_angle(robot_forward, rel_pos)
        # vel = OracleNavAction._compute_turn(
        #     rel_pos, self._turn_velocity, robot_forward
        # )
        vel = [self._forward_velocity, 0]
        trans = self._vc.act(trans, vel)
        cur_pos = trans.translation

        if self.is_collision(trans):
            return True

        return False

    def rotation_collision_check(
        self,
        next_pos,
    ):
        """
        This function checks if the robot needs to do backing-up action
        """
        # Make a copy of agent trans
        trans = mn.Matrix4(self.cur_articulated_agent.sim_obj.transformation)
        angle = float("inf")
        # Get the current location of the agent
        cur_pos = self.cur_articulated_agent.base_pos
        # Set the trans to be agent location
        trans.translation = self.cur_articulated_agent.base_pos

        while abs(angle) > self._turn_thresh:
            # Compute the robot facing orientation
            rel_pos = (next_pos - cur_pos)[[0, 2]]
            forward = np.array([1.0, 0, 0])
            robot_forward = np.array(trans.transform_vector(forward))
            robot_forward = robot_forward[[0, 2]]
            angle = get_angle(robot_forward, rel_pos)
            vel = OracleNavAction._compute_turn(
                rel_pos, self._turn_velocity, robot_forward
            )
            trans = self._vc.act(trans, vel)
            cur_pos = trans.translation

            if self.is_collision(trans):
                return True

        return False

    def step(self, *args, is_last_action, **kwargs):
        self.skill_done = False
        self.collided_rot = False
        self.collided_for = False
        self.collided_lat = False

        if self._action_arg_prefix == 'agent_1_':
            self.cur_articulated_agent.at_goal_ona = False

        # if (self._action_arg_prefix + "just_rotate" in kwargs) and kwargs[self._action_arg_prefix + "just_rotate"] != 0.0:
        #     vel = [0, 0, self._config.turn_velocity]
        #     kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
        #     #breakpoint()
        #     return BaseVelNonCylinderAction.step(
        #         self, *args, is_last_action=is_last_action, **kwargs
        #     )
        if (self._action_arg_prefix + "search_rotate" in kwargs) and kwargs[self._action_arg_prefix + "search_rotate"] != 0.0:
            vel = [0, 0, self._config.turn_velocity]
            kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
            #breakpoint()
            return BaseVelNonCylinderAction.step(
                self, *args, is_last_action=is_last_action, **kwargs
            )

        if (self._action_arg_prefix + "just_rotate" in kwargs) and kwargs[self._action_arg_prefix + "just_rotate"] != 0.0:
            self.dist_thresh = 1.35 #Just set it to this
        else:
            self.dist_thresh = self._ori_dist_thresh

        # if (self._action_arg_prefix + "human_follow" in kwargs) and kwargs[self._action_arg_prefix + "human_follow"] != 0.0:
        #     self.dist_thresh = 0.75 #0.5 #0.0 #Just set it to this
        # else:
        #     self.dist_thresh = self._ori_dist_thresh


        nav_to_target_idx = kwargs[
            self._action_arg_prefix + "oracle_nav_with_backing_up_action"
        ] #something like array([3.], dtype=float32) if original (not move freely, not human)
        if nav_to_target_idx.shape == (1,):
            if  nav_to_target_idx <= 0 or nav_to_target_idx > len(
                self._poss_entities
            ):
                if is_last_action:
                    return self._sim.step(HabitatSimActions.base_velocity)
                else:
                    return {}

        #print("here 1")
            #Just retyrn

        #Make human move
        if isinstance(nav_to_target_idx, np.ndarray) and nav_to_target_idx[0]==np.inf:
            return

        else:
            if isinstance(nav_to_target_idx, np.ndarray) and nav_to_target_idx[0] != np.inf and nav_to_target_idx.shape == (1,): #original
                move_freely = False
            elif isinstance(nav_to_target_idx, np.ndarray) and nav_to_target_idx[0] != np.inf and nav_to_target_idx.shape == (3,): #move freely with ogn
                move_freely = True
                #action_to_take = kwargs[self._action_arg_prefix + "oracle_nav_with_backing_up_action"]
            else:
                raise Exception("what is this")
        if move_freely:
            #same as self._get_target_for_idx
            obj_targ_pos = kwargs[self._action_arg_prefix + "oracle_nav_with_backing_up_action"] #goal_pose or stg pose from objectgoal_env  really
            # final_nav_targ, _, _ = place_agent_at_dist_from_pos(
            #         np.array(obj_targ_pos),
            #         0.0,
            #         -1.0, #None, #-1.0, #self._spawn_max_dist_to_obj,
            #         self._sim,
            #         self._num_spawn_attempts,
            #         1,
            #         self.cur_articulated_agent,
            #         self._navmesh_offset_for_agent_placement,
            #         #None,
            #     )
            # while not(self.found_path(final_nav_targ)):
            #     final_nav_targ, _, _ = place_agent_at_dist_from_pos(
            #         np.array(obj_targ_pos),
            #         0.0,
            #         -1.0, #None, #-1.0, #self._spawn_max_dist_to_obj,
            #         self._sim,
            #         self._num_spawn_attempts,
            #         1,
            #         self.cur_articulated_agent,
            #         self._navmesh_offset_for_agent_placement,
            #         #None,
            #     )
            #Replace to just sample until 
            #don't replan!
            #breakpoint()
            # if self.ep_num==0:
            #     self.spot_init_failed = True
            #breakpoint()
            if np.linalg.norm((obj_targ_pos - self.prev_obj_targ_pos)[[0, 2]])<=0.1: 
                #print("here 2")
                final_nav_targ = self.prev_final_nav_targ
            else:
                final_nav_targ = obj_targ_pos
                counter = 0
                found_path = self.found_path(final_nav_targ)
                if not(self.spot_init_failed):
                    while not(found_path):
                        #print("here 3")
                        #print("counter ", counter)
                        #print("timestep ", self.timestep)
                        # final_nav_targ, _, _ = place_robot_at_closest_point(
                        # obj_targ_pos, self._sim, agent=self.cur_articulated_agent)
                        #final_nav_targ, _, _ = place_robot_at_closest_point_for_sem_map(obj_targ_pos, self._sim, agent=self.cur_articulated_agent)
                        final_nav_targ, _, _ = place_robot_at_closest_point_for_sem_map_with_navmesh(obj_targ_pos, self._sim, self._navmesh_offset_for_agent_placement,agent=self.cur_articulated_agent)
                        counter +=1
                        found_path = self.found_path(final_nav_targ)
                        if counter >20 and self.timestep==0:
                            #breakpoint()
                            #just sample again
                            # _largest_island_idx = get_largest_island_index(
                            #     self._sim.pathfinder, self._sim, allow_outdoor=False
                            # )
                            # start_pos0 = self._sim.pathfinder.get_random_navigable_point(island_index=_largest_island_idx)
                            # self.cur_articulated_agent.base_pos = start_pos0
                            #breakpoint()
                            #self._sim._sensor_suite.get_observations(self._sim.get_sensor_observations())
                            #self.failed_init = True
                            #self.cur_articulated_agent.base_pos = np.array([0.0, 0.0, 0.0])
                            #return 
                            found_path=True
                            self.spot_init_failed = True
                            #return
                            #counter = 0
                            #self.cur_articulated_agent.base_pos = obj_targ_pos
                            #final_nav_targ = obj_targ_pos
                            #found_path = True
                        if counter >20 and self.timestep>0:
                            #breakpoint()
                            found_path=True
                # place_robot_at_closest_point_with_navmesh(
                final_nav_targ = np.array(final_nav_targ)
            #print("Placed!")
            #print("here 4")
            robot_rot = float(self.cur_articulated_agent.base_rot)
            robot_pos = np.array(self.cur_articulated_agent.base_pos)
            if self.timestep>=0:
                # if self.timestep>=30:
                #     breakpoint()
                if self.last_vel[0] >0 and np.linalg.norm(self.last_pos[[0,2]] - robot_pos[[0,2]])<=0.001:
                    self.collided_for = True
                    self.forward_side_step_try = 0
                    #breakpoint()
                elif self.last_vel[2] !=0.0 and abs(self.last_rot - robot_rot)<0.05: #was rotation and robot pos no change
                    self.collided_rot = True
                elif self.last_vel[1] != 0.0 and np.linalg.norm(self.last_pos[[0,2]] - robot_pos[[0,2]])<=0.001:
                    self.collided_lat = True
                

        else:
            nav_to_target_idx = int(nav_to_target_idx[0]) - 1
            final_nav_targ, obj_targ_pos = self._get_target_for_idx(
                nav_to_target_idx
            )
        # Get the base transformation
        base_T = self.cur_articulated_agent.base_transformation
        #print("here 5")
        # Get the current path
        curr_path_points = self._path_to_point(final_nav_targ)
        # Get the robot position
        robot_pos = np.array(self.cur_articulated_agent.base_pos)
        robot_rot = float(self.cur_articulated_agent.base_rot)
        self.last_rot = robot_rot
        self.last_pos = robot_pos

        # print("robot_pos",robot_pos)
        # print("obj_targ_pos", obj_targ_pos)
        # print("final nav targ", final_nav_targ)
        # print("curr_path_points", curr_path_points)

        # Get the current robot/human pos assuming human is agent 1
        robot_human_dis = None
        self.prev_obj_targ_pos = copy.deepcopy(obj_targ_pos)
        self.prev_final_nav_targ = copy.deepcopy(final_nav_targ)
        self.timestep +=1
        if self._sim.num_articulated_agents > 1:
            # This is very specific to SIRo. Careful merging
            _robot_pos = np.array(
                self._sim.get_agent_data(
                    0
                ).articulated_agent.base_transformation.translation
            )[[0, 2]]
            _human_pos = np.array(
                self._sim.get_agent_data(
                    1
                ).articulated_agent.base_transformation.translation
            )[[0, 2]]
            # Compute the distance
            robot_human_dis = np.linalg.norm(_robot_pos - _human_pos)

        # if move_freely: 
        #     if action_to_take == 1:  # forward
        #         vel = [self._config.forward_velocity, 0, 0]
        #     elif action_to_take == 2:  # turn left #Just changed for oGN
        #         vel = [0, 0, self._config.turn_velocity] #[0, -self._config.turn_velocity]
        #     elif action_to_take == 3:  # turn right
        #         vel = [0, 0, -self._config.turn_velocity] #[0, self._config.turn_velocity]
        #     elif action_to_take == 4:  # lateral right
        #         vel = [0, -self._config.forward_velocity, 0] #[0, -self._config.turn_velocity]
        #     elif action_to_take == 5:  # lateral left
        #         vel = [0, self._config.forward_velocity, 0] #[0, self._config.turn_velocity]
            
            
        #     else:  # stop
        #         vel = [0, 0]
        #     # else:
        #     #     vel = [0, 0]
        #     #self.skill_done = True
        #     kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
        #     #kwargs[f"{self._action_arg_prefix}rotation_collision"] = 
        #     return BaseVelNonCylinderAction.step(
        #         self, *args, is_last_action=is_last_action, **kwargs
        #     )

        # else:
        if curr_path_points is None:
            raise RuntimeError("Pathfinder returns empty list")
        else:
            # Compute distance and angle to target
            if len(curr_path_points) == 1:
                curr_path_points += curr_path_points

            cur_nav_targ = curr_path_points[1]
            forward = np.array([1.0, 0, 0])
            robot_forward = np.array(base_T.transform_vector(forward))

            # Compute relative target
            rel_targ = cur_nav_targ - robot_pos

            # Compute heading angle (2D calculation)
            robot_forward = robot_forward[[0, 2]]
            rel_targ = rel_targ[[0, 2]]
            rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]
            # Get the angles
            angle_to_target = get_angle(robot_forward, rel_targ)
            angle_to_obj = get_angle(robot_forward, rel_pos)
            # Compute the distance
            dist_to_final_nav_targ = np.linalg.norm((final_nav_targ - robot_pos)[[0, 2]])

            at_goal = (
                dist_to_final_nav_targ < self.dist_thresh
                and angle_to_obj < self._config.turn_thresh
            ) or (self.at_goal_prev and dist_to_final_nav_targ < self._config.dist_thresh)

            if self._action_arg_prefix == 'agent_1_':
                print("at goal ", at_goal)
                print("dist_to_final_nav_targ", dist_to_final_nav_targ)
                #self.cur_articulated_agent.at_goal_ona = False
            # print("angle_to_obj", angle_to_obj)
            # print("angle_to_target", angle_to_target)
            #breakpoint()

            # Planning to see if the robot needs to do back-up
            need_move_backward = False
            # if (
            #     dist_to_final_nav_targ >= self._config.dist_thresh
            #     and angle_to_target >= self._config.turn_thresh
            #     and not at_goal
            # ):
            #     # check if there is a collision caused by rotation
            #     # if it does, we should block the rotation, and
            #     # only move backward
            #     need_move_backward = self.rotation_collision_check(cur_nav_targ,) 

            #move_backward_col = False
            #rot_col_check = self.rotation_collision_check(cur_nav_targ,)  
            #print("rot col check is ", rot_col_check)

            if need_move_backward:
                # Backward direction
                forward = np.array([-1.0, 0, 0])
                robot_forward = np.array(base_T.transform_vector(forward))
                # Compute relative target
                rel_targ = cur_nav_targ - robot_pos
                # Compute heading angle (2D calculation)
                robot_forward = robot_forward[[0, 2]]
                rel_targ = rel_targ[[0, 2]]
                rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]
                # Get the angles
                angle_to_target = get_angle(robot_forward, rel_targ)
                angle_to_obj = get_angle(robot_forward, rel_pos)
                # Compute the distance
                dist_to_final_nav_targ = np.linalg.norm(
                    (final_nav_targ - robot_pos)[[0, 2]]
                )
                at_goal = (
                    dist_to_final_nav_targ < self._config.dist_thresh
                    and angle_to_obj < self._config.turn_thresh
                )
            # print("turn thresh is ",  self._config.turn_thresh)
            if self.motion_type == "base_velocity":
                self.cur_articulated_agent.at_goal_ona = at_goal
                if not at_goal:
                    self.at_goal = False
                    self.at_goal_prev = False
                    if self.collided_for or self.forward_side_step_try>=1: #side step in the direction that makes sense
                        #if self.forward_side_step_try ==0:
                        vel = [0, 0, 0]
                        #self.forward_side_step_try +=1
                        self.cur_articulated_agent.base_pos = cur_nav_targ
                        # elif self.forward_side_step_try ==1:
                        #     vel = [0, 0, 0]
                        #     self.forward_side_step_try = -1
                        #breakpoint()
                    elif self.collided_rot:
                        #vel = -1. * self.last_vel
                        if self.last_vel[2] >0 : #if rotated left in the last vel, do lateral right
                            vel = [0, self._config.forward_velocity, 0] #[0, -self._config.turn_velocity]
                        else:  # lateral left
                            vel = [0, -self._config.forward_velocity, 0]
                    elif self.collided_lat:
                        #move backward and lateral at the same time
                        #vel = [-self._config.forward_velocity, self.last_vel[1], 0]
                        vel = [0, 0, 0]
                        #self.forward_side_step_try +=1
                        self.cur_articulated_agent.base_pos = cur_nav_targ
                    elif dist_to_final_nav_targ < self._config.dist_thresh:
                        # Look at the object
                        vel = OracleNavAction._compute_turn(
                            rel_pos, self._config.turn_velocity, robot_forward
                        )
                    elif angle_to_target < self._config.turn_thresh:
                        #breakpoint()
                        # Move towards the target
                        vel = [self._config.forward_velocity, 0]
                    else:
                        # Look at the target waypoint.
                        vel = OracleNavAction._compute_turn(
                            rel_targ, self._config.turn_velocity, robot_forward
                        )
                    # if move_backward_col:
                    #     #just find an action that won't cause collision
                    #     forward_col = self.virtual_collision_check([1, 0])
                    #     right_col =
                    #     #Tried to move forward 
                    #     if vel == [self._config.forward_velocity, 0]:


                else:
                    self.at_goal = False
                    self.at_goal_prev = True
                    # self.at_goal = True
                    # self.skill_done = True
                    # #vel = [0, 0]
                    # #Turn 
                    #breakpoint()
                    vel = [0, self._config.turn_velocity]

                # if need_move_backward:
                #     vel[0] = -1 * vel[0]
                if self.spot_init_failed:
                    vel = [0,0,0]
                    self.cur_articulated_agent.base_pos = np.array([0.0, 0.0, 0.0])

                #For reading action sequences
                if self._config.enable_lateral_move and len(vel)==2:
                    vel = [vel[0], 0, vel[1]]


                # if self.timestep >=30:
                #     breakpoint()
                self.last_vel = vel
                #breakpoint()
                #print("vel was ", vel)
                kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
                kwargs[f"{self._action_arg_prefix}tele"] = final_nav_targ #obj_targ_pos
                return BaseVelNonCylinderAction.step(
                    self, *args, is_last_action=is_last_action, **kwargs
                )

            elif self.motion_type == "human_joints":
                # Update the humanoid base
                self.cur_articulated_agent.at_goal_ona = at_goal
                self.humanoid_controller.obj_transform_base = base_T
                if not at_goal:
                    self.at_goal = False
                    if dist_to_final_nav_targ < self._config.dist_thresh:
                        # Look at the object
                        self.humanoid_controller.calculate_turn_pose(
                            mn.Vector3([rel_pos[0], 0.0, rel_pos[1]])
                        )
                    else:
                        # Move towards the target
                        self.humanoid_controller.calculate_walk_pose(
                            mn.Vector3([rel_targ[0], 0.0, rel_targ[1]])
                        )
                else:
                    self.at_goal = True
                    self.skill_done = True
                    self.humanoid_controller.calculate_stop_pose()

                self._update_controller_to_navmesh()
                base_action = self.humanoid_controller.get_pose()
                kwargs[
                    f"{self._action_arg_prefix}human_joints_trans"
                ] = base_action

                return HumanoidJointAction.step(
                    self, *args, is_last_action=is_last_action, **kwargs
                )

            else:
                raise ValueError(
                    "Unrecognized motion type for oracle nav action"
                )