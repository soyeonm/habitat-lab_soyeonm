# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Tuple

import magnum as mn
import numpy as np
from gym import spaces

import habitat_sim
from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat.core.registry import registry

# from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.rearrange.actions.actions import (
    BaseVelAction,
    HumanoidJointAction,
)
from habitat.tasks.rearrange.utils import place_agent_at_dist_from_pos
from habitat.tasks.utils import get_angle
import pickle

@registry.register_task_action
class OracleNavSocAction(BaseVelAction, HumanoidJointAction):
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
            self.humanoid_controller = self.lazy_inst_humanoid_controller(task)

        else:
            raise ValueError("Unrecognized motion type for oracle nav  action")

        self._task = task
        self._poss_entities = (
            self._task.pddl_problem.get_ordered_entities_list()
        )
        self._prev_ep_id = None
        self._targets = {}

        self.skill_done = False

        # Just wrote this
        self._counter = 0
        self._waypoint_count = 20
        print("Oracle nav soc action is called!")

        self.poses: List[np.ndarray] = []
        self.waypoints: List[np.ndarray] = []
        self.waypoint_pointer: int = 0
        self.prev_navigable_point: np.ndarray = np.array([])
        self.prev_pose: Tuple[np.ndarray, np.ndarray]

    @staticmethod
    def _compute_turn(rel, turn_vel, robot_forward):
        is_left = np.cross(robot_forward, rel) > 0
        if is_left:
            vel = [0, -turn_vel]
        else:
            vel = [0, turn_vel]
        return vel

    def lazy_inst_humanoid_controller(self, task):
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
            task.humanoid_controller = humanoid_controller
        return task.humanoid_controller

    # def find_short_path_from_two_points(
    #     self, sample1=None, sample2=None
    # ) -> habitat_sim.ShortestPath():
    #     """
    #     Finds two random points on the NavMesh, calculates a shortest path between
    #     the two, and creates a trajectory object to visualize the path.
    #     """
    #     # if self.spline_path_traj_obj_id >= 0:
    #     #     self._sim.get_rigid_object_manager().remove_object_by_id(
    #     #         self.spline_path_traj_obj_id
    #     #     )
    #     self.spline_path_traj_obj_id = -1

    #     found_path = False
    #     # while not found_path:
    #         #sample1 = None
    #         #sample2 = None
    #     while sample1 is None or sample2 is None:
    #         sample1 = sample1 or self._sim.pathfinder.get_random_navigable_point()
    #         sample2 = sample2 or self._sim.pathfinder.get_random_navigable_point()

    #         # constraint points to be on first floor
    #         if sample1[1] != sample2[1] or sample1[1] > 2:
    #             logger.warn(
    #                 "Warning: points are out of acceptable area, replacing with randoms"
    #             )
    #             sample1, sample2 = None, None
    #     path = habitat_sim.ShortestPath()
    #     path.requested_start = sample1
    #     path.requested_end = sample2
    #     #found_path = self._sim.pathfinder.find_path(path)
    #     #print("Found path ", found_path) #This is always false
    #     self.path_points = [sample1, sample2] #path.points #[sample1, sample2] #path.points

    #     spline_points = habitat_sim.geo.build_catmull_rom_spline(self.path_points, 10, 0.75)
    #     self.path_points = spline_points
    #     # print("THis worked")

    #     colors_spline = [mn.Color3.blue(), mn.Color3.green()]

    #     self.spline_path_traj_obj_id = self._sim.add_gradient_trajectory_object(
    #         traj_vis_name=f"spline_{time.strftime('%Y-%m-%d_%H-%M-%S')}",
    #         colors=colors_spline,
    #         points=self.path_points,
    #         radius=0.01,
    #     )
    #     # print("Not this")

    #     return path

    def find_short_path_from_two_points(
        self, sample1=None, sample2=None
    ) -> habitat_sim.ShortestPath():
        import time
        """
        Finds two random points on the NavMesh, calculates a shortest path between
        the two, and creates a trajectory object to visualize the path.
        """
        # if self.spline_path_traj_obj_id >= 0:
        #     self._sim.get_rigid_object_manager().remove_object_by_id(
        #         self.spline_path_traj_obj_id
        #     )
        self.spline_path_traj_obj_id = -1

        found_path = False
        while not found_path:
            # sample1 = None
            # sample2 = None
            while sample1 is None or sample2 is None:
                sample1 = (
                    sample1
                    or self._sim.pathfinder.get_random_navigable_point()
                )
                sample2 = (
                    sample2
                    or self._sim.pathfinder.get_random_navigable_point()
                )

                # constraint points to be on first floor
                if sample1[1] != sample2[1] or sample1[1] > 2:
                    logger.warn(
                        "Warning: points are out of acceptable area, replacing with randoms"
                    )
                    sample1, sample2 = None, None

            # breakpoint()
            # if sample1[1] != sample2[1] or sample1[1] > 2:
            #     print(
            #         "Warning: points are out of acceptable area, replacing with randoms"
            #     )
            #     sample1, sample2 = None, None

            path = habitat_sim.ShortestPath()
            path.requested_start = sample1
            path.requested_end = sample2
            found_path = self._sim.pathfinder.find_path(path)
            print("found path is ", found_path)
            self.path_points = path.points

        spline_points = habitat_sim.geo.build_catmull_rom_spline(
            path.points, 10, 0.75
        )
        print("Built spline!")
        path.points = spline_points
        colors_spline = [mn.Color3.blue(), mn.Color3.green()]
        print("Colors spline!")

        self.spline_path_traj_obj_id = (
            self._sim.add_gradient_trajectory_object(
                traj_vis_name=f"spline_{time.strftime('%Y-%m-%d_%H-%M-%S')}",
                colors=colors_spline,
                points=spline_points,
                radius=0.01,
            )
        )
        print("Drew spline!")
        return path

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "oracle_nav_soc_action": spaces.Box(
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
        self._counter = 0
        self.poses = []
        #self.robot_forward_list = pickle.load(open('robot_forward_list.p', 'rb'))

    def get_waypoints(self):
        # When resetting, decide 5 navigable points
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

        self._largest_island_idx = _largest_island_idx
        self.cur_articulated_agent.sim_obj.translation = start_pos
        self.initial_height = start_pos[1]


        self.waypoints = []
        self.waypoint_pointer = 0
        self.prev_navigable_point = np.array(
            self.cur_articulated_agent.base_pos
        )
        while len(self.waypoints) < self._waypoint_count:
            # if self._get_distance(self._get_current_pose(), self.prev_navigable_point)<=0.3 :# stop condition
            final_nav_targ, _ = self._get_random_waypoint()
            self.prev_navigable_point = final_nav_targ
            self.waypoints.append(final_nav_targ)
        print("Initialized waypoints are ", self.waypoints)

    def _get_random_waypoint(self):
        # Just sample a new point
        # print("Getting waypoint")
        #base_T = self.cur_articulated_agent.base_transformation

        navigable_point = self._sim.pathfinder.get_random_navigable_point(island_index=self._largest_island_idx)
        
        #self.cur_articulated_agent.base_transformation.translation = start_pos
        #breakpoint()


        found_path = False
        # while abs(navigable_point[1] - self.prev_navigable_point[1]) >= 0.1 or self._get_distance(self.prev_navigable_point, navigable_point) <=7: #add distance measure too
        while (
            #abs(navigable_point[1] - self.prev_navigable_point[1]) >= 0.1  or self._get_distance(self.prev_navigable_point, navigable_point) <= 3 or not(found_path)
            self._get_distance(self.prev_navigable_point, navigable_point) <= 5
        ):
            navigable_point = self._sim.pathfinder.get_random_navigable_point(
                island_index=self._largest_island_idx
            )
            # path = habitat_sim.ShortestPath()
            # path.requested_start = self.prev_navigable_point
            # path.requested_end = navigable_point
            # found_path = self._sim.pathfinder.find_path(path)

            
        # print("navigable point is ", navigable_point)
        # print("dist is ", self._get_distance(self.prev_navigable_point, navigable_point))
        return navigable_point, navigable_point

    def _get_target_for_idx(self, nav_to_target_idx: int):
        nav_to_obj = self._poss_entities[nav_to_target_idx]
        if (
            nav_to_target_idx not in self._targets
            or "robot" in nav_to_obj.name
        ):
            obj_pos = self._task.pddl_problem.sim_info.get_entity_pos(
                nav_to_obj
            )  # obj_pos is Vector(3.44619, 0.48436, 4.73188)
            if "robot" in nav_to_obj.name:
                # Safety margin between the human and the robot
                sample_distance = 1.0
            else:
                sample_distance = (
                    self._config.spawn_max_dist_to_obj
                )  # this is -1.0
            start_pos, _, _ = place_agent_at_dist_from_pos(
                np.array(obj_pos),
                0.0,
                sample_distance,
                self._sim,
                self._config.num_spawn_attempts,
                1,
                self.cur_articulated_agent,
            )
            if np.isnan(start_pos).any():
                print("start_pos contains NaN @oracle_nav_soc_action.py.")

            if self.motion_type == "human_joints":
                self.humanoid_controller.reset(
                    self.cur_articulated_agent.base_transformation
                )
            self._targets[nav_to_target_idx] = (start_pos, np.array(obj_pos))
        # import ipdb; ipdb.set_trace()
        return self._targets[nav_to_target_idx]

    def _path_to_point(self, point):
        """
        Obtain path to reach the coordinate point. If agent_pos is not given
        the path starts at the agent base pos, otherwise it starts at the agent_pos
        value
        :param point: Vector3 indicating the target point
        """
        agent_pos = self.cur_articulated_agent.base_pos
        # if self._counter == 0:
        #     agent_pos = self.cur_articulated_agent.base_pos
        # else:
        #     agent_pos = self.temp_human_pos #self.cur_articulated_agent.base_pos

        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = point
        found_path = self._sim.pathfinder.find_path(path)
        #breakpoint()
        #print("Found path: ", found_path)
        if not found_path:
            #breakpoint()
            return [agent_pos, point]
        return path.points

    def _put_offset_back(self): #make the agent sink again
        # print("before adding ", self.humanoid_controller.obj_transform_base.translation)
        # trans = self.cur_articulated_agent.sim_obj.transformation
        # rigid_state = habitat_sim.RigidState(
        #     mn.Quaternion.from_matrix(trans.rotation()), trans.translation
        # )
        # target_rigid_state_trans = (
        #     self.humanoid_controller.obj_transform_base.translation
        # )
        # end_pos = self._sim.step_filter(
        #     rigid_state.translation, target_rigid_state_trans
        # )

        # end_pos += self.cur_articulated_agent.params.base_offset
        #self.humanoid_controller.obj_transform_base.translation = end_pos
        self.cur_articulated_agent.sim_obj.translation += self.cur_articulated_agent.params.base_offset

        # #end_pos += self.cur_articulated_agent.params.base_offset
        # print("after adding ", self.humanoid_controller.obj_transform_base.translation)
        # pass

    def _update_controller_to_navmesh(self):
        trans = self.cur_articulated_agent.sim_obj.transformation
        rigid_state = habitat_sim.RigidState(
            mn.Quaternion.from_matrix(trans.rotation()), trans.translation
        )
        target_rigid_state_trans = (
            self.humanoid_controller.obj_transform_base.translation
        )
        end_pos = self._sim.step_filter(
            rigid_state.translation, target_rigid_state_trans
        )
        #print("original end pos is ", end_pos)

        # Offset the base
        #end_pos -= self.cur_articulated_agent.params.base_offset
        # if self._counter == 1:
        #     end_pos -= self.cur_articulated_agent.params.base_offset

        # Offset the base
        #end_pos -= self.cur_articulated_agent.params.base_offset
        # import copy
        # if self._counter == 0:
        #     breakpoint()
        #     print("counter was 0")
        # if self._counter> 1:
        #     end_pos += self.cur_articulated_agent.params.base_offset
        # self.temp_human_pos = copy.deepcopy(np.array(end_pos))
        # print("temp human pos is ", self.temp_human_pos)
        # #if self._counter == 1:
        # end_pos -= self.cur_articulated_agent.params.base_offset
        # if self._counter> 1:
        #     end_pos -= self.cur_articulated_agent.params.base_offset
        # #end_pos -= 2*self.cur_articulated_agent.params.base_offset
        # print("final end pos is ", end_pos)
        #end_pos -= self.cur_articulated_agent.params.base_offset
        self.humanoid_controller.obj_transform_base.translation = end_pos
        # print("viz human pos is ", self.humanoid_controller.obj_transform_base.translation)

    def _get_current_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        base_T = self.cur_articulated_agent.base_transformation
        robot_pos = np.array(self.cur_articulated_agent.base_pos)
        forward = np.array([1.0, 0, 0])
        robot_forward = np.array(base_T.transform_vector(forward))
        # Compute heading angle (2D calculation)
        robot_forward = robot_forward[[0, 2]]
        return (robot_pos, robot_forward)

    def _get_distance(self, prev_nav_target, final_nav_targ):
        dist_to_final_nav_targ = np.linalg.norm(
            (final_nav_targ - prev_nav_target)  # [[0, 2]]
        )
        return dist_to_final_nav_targ

    def _decide_if_stuck(self, prev_pos, cur_pos):
        stuck_xy = np.sqrt(((prev_pos[0] - cur_pos[0]) ** 2).sum()) < 0.001
        stuck_angle = np.sqrt(((prev_pos[1] - cur_pos[1]) ** 2).sum()) < 0.001
        # temporary fix
        return stuck_xy and stuck_angle

    def compute_geodesc_dist_for_path(self, path_points, start_pose):
        # euclidean distance between points
        dist = 0.0
        for p in path_points:
            dist += np.linalg.norm((p - start_pose)[[0, 2]]).item()
        return dist

    def compute_opt_trajectory_len_until_found(self, robot_start_pose):
        # Compute geodesic distance to all the human_poses at step_i
        geo_dists = []
        for p in self.poses:
            path = habitat_sim.ShortestPath()
            path.requested_start = robot_start_pose
            path.requested_end = p  # oint
            found_path = self._sim.pathfinder.find_path(path)
            if found_path:
                geo_dist = self.compute_geodesc_dist_for_path(
                    path.points, robot_start_pose
                )
                geo_dists.append(geo_dist)
            else:
                # raise Exception("path not found!")
                geo_dist = np.inf
                geo_dists.append(geo_dist)
        # get the argmin among geo_dists
        optimal_dist = np.min(geo_dists)
        return optimal_dist, np.argmin(geo_dists)

    #Just directly do it here
    def get_humanoid_controller_pose(self):
        """
        Obtains the controller joints, offset and base transform in a vectorized form so that it can be passed
        as an argument to HumanoidJointAction
        """
        obj_trans_offset = np.asarray(
            self.humanoid_controller.obj_transform_offset.transposed()
        ).flatten()
        #print("humanoid controller base before", self.humanoid_controller.obj_transform_base)
        base = mn.Matrix4(np.array(self.humanoid_controller.obj_transform_base))
        base.translation -= self.cur_articulated_agent.params.base_offset
        obj_trans_base = np.asarray(
            #self.humanoid_controller.obj_transform_base.transposed()
            base.transposed()
        ).flatten()
        #print("humanoid controller base ", self.humanoid_controller.obj_transform_base)
        return self.humanoid_controller.joint_pose + list(obj_trans_offset) + list(obj_trans_base)

    def step(self, *args, is_last_action, **kwargs):
        # nav_to_target_idx = kwargs[
        #     self._action_arg_prefix + "oracle_nav_action"
        # ]

        # if nav_to_target_idx <= 0 or nav_to_target_idx > len(
        #     self._poss_entities
        # ):
        #     if is_last_action:
        #         return self._sim.step(HabitatSimActions.base_velocity)
        #     else:
        #         return {}
        # import ipdb; ipdb.set_trace()
        # nav_to_target_idx = int(nav_to_target_idx[0]) - 1

        # final_nav_targ, obj_targ_pos = self._get_target_for_idx(
        #    nav_to_target_idx
        #
        # if self._counter ==0:
        #     self.prev_navigable_point = np.array(self.cur_articulated_agent.base_pos)
        self.skill_done = False
        # print("Step called! ", self._counter)
        if self._counter == 0:
            self.get_waypoints()
            #breakpoint()
            self.waypoint_increased_step = self._counter
        else:
            self._put_offset_back()

        #self.cur_articulated_agent.sim_obj.translation[1] = self.initial_height
        # self.cur_articulated_agent.sim_obj.translation = np.array(self.cur_articulated_agent.sim_obj.translation)
        # self.cur_articulated_agent.sim_obj.translation[1] = self.initial_height
        # self.cur_articulated_agent.sim_obj.translation = mn.Vector3(self.cur_articulated_agent.sim_obj.translation)

        print(
            "step ",
            str(self._counter),
            ": dist is ",
            self._get_distance(
                self._get_current_pose()[0],
                self.waypoints[self.waypoint_pointer],
            ),
        )
        # print("pointer is ", self.waypoint_pointer)
        # print("cur pose is ",self._get_current_pose() )
        print(
            "step ",
            str(self._counter),
            ": cur pose is ",
            self._get_current_pose()[0],
        )
        # print("prev navigable point is ", self.prev_navigable_point)
        # if self._counter %20==0:
        # If almost there, resample
        # TODO: change this to at_goal
        if self._counter > 0:
            stuck = self._decide_if_stuck(
                self.prev_pose, self._get_current_pose()
            )
            reached_waypoint = (
                self._get_distance(
                    self._get_current_pose()[0],
                    self.waypoints[self.waypoint_pointer],
                )
                <= 0.01
            )  # _config.stop_thresh is 0.001 #stop condition
            if self.waypoint_pointer + 1 < len(self.waypoints) and (
                stuck or reached_waypoint
            ):
                self.waypoint_pointer += 1
                print("step ", str(self._counter), ": NEW WAYPOINT!")
                self.waypoint_increased_step = self._counter

        final_nav_targ, obj_targ_pos = (
            self.waypoints[self.waypoint_pointer],
            self.waypoints[self.waypoint_pointer],
        )

        # print("received nav_to_target_idx", nav_to_target_idx)
        # print("final_nav_targ ", final_nav_targ, " obj_targ_pos, ", obj_targ_pos)
        # import ipdb; ipdb.set_trace()
        # print("Get target for idx called!")
        base_T = self.cur_articulated_agent.base_transformation
        curr_path_points = self._path_to_point(final_nav_targ)
        robot_pos = np.array(self.cur_articulated_agent.base_pos)
        self.poses.append(robot_pos)
        #self.find_short_path_from_two_points(robot_pos,self.waypoints[self.waypoint_pointer])
        # Visualize waypoint pointer and my pose
        # self.find_short_path_from_two_points(self.waypoints[self.waypoint_pointer], robot_pos)

        self._counter += 1
        self.prev_pose = self._get_current_pose()
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
            #robot_forward = self.robot_forward_list[self._counter-1]
            rel_targ = rel_targ[[0, 2]]
            rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]

            angle_to_target = get_angle(robot_forward, rel_targ)
            angle_to_obj = get_angle(robot_forward, rel_pos)

            dist_to_final_nav_targ = np.linalg.norm(
                (final_nav_targ - robot_pos)[[0, 2]]
            )
            at_goal = (
                dist_to_final_nav_targ < self._config.dist_thresh
                and angle_to_obj < self._config.turn_thresh
            )

            if self.motion_type == "base_velocity":
                if not at_goal:
                    if dist_to_final_nav_targ < self._config.dist_thresh:
                        # Look at the object
                        vel = OracleNavSocAction._compute_turn(
                            rel_pos, self._config.turn_velocity, robot_forward
                        )
                    elif angle_to_target < self._config.turn_thresh:
                        # Move towards the target
                        vel = [self._config.forward_velocity, 0]
                    else:
                        # Look at the target waypoint.
                        vel = OracleNavSocAction._compute_turn(
                            rel_targ, self._config.turn_velocity, robot_forward
                        )
                else:
                    vel = [0, 0]
                    if self.waypoint_pointer == len(self.waypoints) - 1:
                        self.skill_done = True
                        print("Completed!")
                kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
                return BaseVelAction.step(
                    self, *args, is_last_action=is_last_action, **kwargs
                )

            elif self.motion_type == "human_joints":
                # Update the humanoid base
                #breakpoint()
                self.humanoid_controller.obj_transform_base = base_T
                #print("self.humanoid_controller.obj_transform_base is ", self.humanoid_controller.obj_transform_base)
                # if self._counter == 1:
                #     self.humanoid_controller.obj_transform_base = base_T
                # else:
                #     self.humanoid_controller.obj_transform_base = self.temp_human_pos
                if not at_goal:
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
                    self.humanoid_controller.calculate_stop_pose()
                    # if at_goal and at the end of the pointer
                    if self.waypoint_pointer == len(self.waypoints) - 1:
                        self.skill_done = True
                        print("Completed!")

                self._update_controller_to_navmesh()
                base_action = self.get_humanoid_controller_pose() #self.humanoid_controller.get_pose()
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
