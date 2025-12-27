
import time
import numpy as np
from functools import partial
from scipy.spatial import ConvexHull
from pydrake.all import (
    namedview,
    MathematicalProgram,
    AddUnitQuaternionConstraintOnPlant,
    OrientationConstraint,
    RotationMatrix,
    eq,
    PositionConstraint,
    SnoptSolver,
    Solve,
    PiecewisePolynomial,
    RollPitchYaw, 
    Quaternion,
    InverseKinematics,   
)
from utils.utils_trajopt import (
    velocity_matching_constraints,
    momentum_matching_constraints,
    angular_momentum_constraints,
    contact_constraints,
)
from math import atan2



class OneStepPlanner:
    def __init__(
            self, 
            plant, 
            context, 
            model, 
            T=0.5, 
            N=10, 
            min_clearence_low=0.01,
            min_clearence_high=0.1,
            target_height=0.0,
            init_stabilization_time = 1,
            target_z=0.5,
            move_distance=0.8, 
            step_length=0.2,
            gait_sequence=[0, 3, 1, 2],
            init_stab_xyz=np.array([0,0,0.52]),
            p_FootFootTip=np.array([0.0, 0.0, -0.379]),
        ):

        self.plant = plant
        self.model = model
        self.context = context

        # state size
        self.n_q = plant.num_positions()
        self.n_v = plant.num_velocities()
        self.n_u = plant.num_actuators()

        # some constants
        self.friction_constant = 0.75  # rubber on rubber
        self.total_mass = plant.CalcTotalMass(context, [model])
        self.gravity = plant.gravity_field().gravity_vector()
        self.foot_frames = [
            plant.GetFrameByName("front_left_lower_leg"),
            plant.GetFrameByName("front_right_lower_leg"),
            plant.GetFrameByName("rear_left_lower_leg"),
            plant.GetFrameByName("rear_right_lower_leg"),
        ]
        # self.leg_qslice = {
        #     'fl': slice(7, 10),
        #     'fr': slice(10, 13),
        #     'bl': slice(13, 16),
        #     'br': slice(16, 19),
        # }
        self.body_frame = plant.GetFrameByName('body')

        # AUTODIFF plant
        self.plant_ad = plant.ToAutoDiffXd()
        self.foot_frames_ad = [
            self.plant_ad.GetFrameByName("front_left_lower_leg"),
            self.plant_ad.GetFrameByName("front_right_lower_leg"),
            self.plant_ad.GetFrameByName("rear_left_lower_leg"),
            self.plant_ad.GetFrameByName("rear_right_lower_leg"),
        ]

        # named view
        self.PositionView = namedview(
            "Positions", plant.GetPositionNames(model, always_add_suffix=False)
        )
        self.VelocityView = namedview(
            "Velocities",
            plant.GetVelocityNames(model, always_add_suffix=False),
        )

        # ROBOT CONSTANTS
        self.p_FootFootTip = p_FootFootTip
        self.target_height = target_height
        self.z_target = target_z
        # STEP parameters
        # duration
        self.init_stabilization_time = init_stabilization_time
        self.T = T
        self.N = N
        self.min_clearence_low = min_clearence_low + target_height
        self.min_clearence_high = min_clearence_high + target_height
        self.move_distance=move_distance
        self.step_length=step_length
        self.gait_sequence = gait_sequence
        self.init_stab_xyz = init_stab_xyz


        # Joint costs weights
        q_cost = self.PositionView(np.ones(self.n_q))
        v_cost = self.VelocityView(np.ones(self.n_v))
        q_cost.body_qw = 0
        q_cost.body_qx = 0
        q_cost.body_qy = 0
        q_cost.body_qz = 0
        q_cost.body_x = 5
        q_cost.body_y = 1
        q_cost.body_z = 5

        # hip angles x
        q_cost.front_left_hip_x = 10
        q_cost.front_right_hip_x = 10
        q_cost.rear_left_hip_x = 10
        q_cost.rear_right_hip_x= 10

        # hip angle pitch, y
        q_cost.front_left_hip_y = 10
        q_cost.front_right_hip_y = 10
        q_cost.rear_left_hip_y = 10
        q_cost.rear_right_hip_y= 10

        # knee angle
        q_cost.front_left_knee = 10
        q_cost.front_right_knee = 10
        q_cost.rear_left_knee = 10
        q_cost.rear_right_knee = 10

        v_cost.body_wx = 0
        v_cost.body_wy = 0
        v_cost.body_wz = 0
        v_cost.body_vx = 0
        v_cost.body_vy = 0
        v_cost.body_vz = 0

        self.Q_v = np.diag(v_cost)
        self.Q_q = np.diag(q_cost)

        # desired velocity
        self.v_desired = np.zeros(self.n_v)

    
    def get_q_end_from_ik(self, q_curr, p_target):
        """
        q_curr current configuration/states
        p_target[i,:] : foot i xyz positions
        """

        # body pose in xy is mean of feet
        p_mean = np.mean(p_target, axis=0)

        # heigh remain same
        p_mean[-1] = q_curr[6]
        # p_mean[0] += 0.05

        # compute body orientation in z (yaw)
        left_feet_dir = p_target[0,:2] - p_target[2,:2] # FL - BL
        right_feet_dir = p_target[1,:2] - p_target[3,:2] # FR - BR
        left_yaw = atan2(left_feet_dir[1], left_feet_dir[0])
        right_yaw = atan2(right_feet_dir[1], right_feet_dir[0])
        
        # average orientation of left / right feet vectors
        body_yaw_target = (right_yaw + left_yaw)/2 

        # current body angles
        # q_curr[0:4] = [qw, qx, qy, qz] quaternion
        quat_vals = q_curr[:4]
        quat_norm = np.linalg.norm(quat_vals)
        if quat_norm == 0:
            # Fallback if norm is zero (rare, but safe to handle)
            quat_vals = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            quat_vals = quat_vals / quat_norm

        body_quat = Quaternion(quat_vals)
        body_rpy = RollPitchYaw(body_quat)


        # ik
        q_end = self.ik_from_pos(
            p_target, 
            xyz_target=p_mean,
            rpy_target=[body_rpy.roll_angle(), body_rpy.pitch_angle(), body_yaw_target]
        )

        return q_end



    def ik_from_pos(self, foot_p_WF_target, context=None, xyz_target=[0,0,0.4], rpy_target=[0,0,0]):
        """
        Solve IK for desired body position and foot positions.
        
        Args:
            context: Current plant context
            foot_p_WF_target: (4,3) array of foot positions in world frame
            xyz_target: [x, y, z] desired body position
        
        Returns:
            q_solution or None if failed
        """

        # update the context
        if context is not None:
            self.context = context

        # inital position
        q0 = self.plant.GetPositions(self.context)

        # ik object
        ik = InverseKinematics(self.plant, self.plant.CreateDefaultContext())

        # for body
        ik.AddPositionConstraint(
            frameB=self.body_frame,
            p_BQ=[0, 0, 0], # The point on the body to constrain (its origin)
            frameA=self.plant.world_frame(),
            p_AQ_lower=np.array(xyz_target) - 1e-3, # The target box in the world
            p_AQ_upper=np.array(xyz_target) + 1e-3
        )

        # body orientation constraint
        Rmatrix_target = RotationMatrix(RollPitchYaw(*rpy_target))
        ik.AddOrientationConstraint(
            frameAbar=self.plant.world_frame(),
            R_AbarA=Rmatrix_target,
            frameBbar=self.body_frame,
            R_BbarB=RotationMatrix(),  # identity
            theta_bound=0.01  # tolerance in radians
        )

        # for foot
        for foot in range(4):
            # feet
            ik.AddPositionConstraint(
                frameB=self.foot_frames[foot],
                p_BQ=self.p_FootFootTip, # The point on the body to constrain (its origin)
                frameA=self.plant.world_frame(),
                p_AQ_lower=foot_p_WF_target[foot,:], # The target box in the world
                p_AQ_upper=foot_p_WF_target[foot,:],
            )
        # unit quaternion constraint for floating base
        AddUnitQuaternionConstraintOnPlant(self.plant, ik.q(), ik.prog())

        # Get the decision variables from the object to use in the cost.
        q_variables = ik.q()

        # initial guess
        ik.prog().SetInitialGuess(q_variables, q0)

        # Add a cost to find the solution "closest" to the home pose q0.
        ik.prog().AddQuadraticErrorCost(np.identity(len(q_variables)), q0, q_variables)

        # solve
        result = Solve(ik.prog())
        if result.is_success():
            return result.GetSolution(q_variables)
        else:
            print("IK could not find a solution!")
            return None
        

    def get_feet_positions_from_q(self, q):
        context = self.plant.CreateDefaultContext()
        self.plant.SetPositions(context, q)
        p_feet_xyz = np.zeros((4,3))
        for i in range(4):
                p_WF = self.plant.CalcPointsPositions(
                    context, 
                    self.foot_frames[i],     
                    self.p_FootFootTip,         
                    self.plant.world_frame()  
                )
                p_feet_xyz[i,:] = p_WF.flatten()[:3]
        return p_feet_xyz
    
    def get_feet_positions_from_context(self, context=None):
        if not context:
            context = self.context
        p_feet_xyz = np.zeros((4,3))
        for i in range(4):
                p_WF = self.plant.CalcPointsPositions(
                    context, 
                    self.foot_frames[i],     
                    self.p_FootFootTip,         
                    self.plant.world_frame()  
                )
                p_feet_xyz[i,:] = p_WF.flatten()[:3]
        return p_feet_xyz

    def plan_one_step(self, p_feet_target_xy,foot_idx, context=None):

        # update the context
        if context is not None:
            self.context = context

        # inital position
        q0 = self.plant.GetPositions(self.context)

        # inital foot position:
        p_feet0_xy = np.zeros((4,2))
        for i in range(4):
                p_WF = self.plant.CalcPointsPositions(
                    self.context, 
                    self.foot_frames[i],     
                    self.p_FootFootTip,         
                    self.plant.world_frame()  
                )
                p_feet0_xy[i,:] = p_WF.flatten()[:2]
        # print("Planner initial foot placement:")
        # print(p_feet0_xy)

        # get p_feet_target_xyz
        p_feet_xyz = np.hstack([p_feet_target_xy, self.target_height*np.ones((4,1))])
        # print("IK target foot placement (xy):")
        # print(p_feet_xyz)

        # final position
        q_end = self.get_q_end_from_ik(q0, p_feet_xyz)
        # print(q0 - q_end)

        # initial and end for com
        com0 = np.array(q0[4:7])
        com_end = np.array(q_end[4:7]) 
        
        # time constatns
        N, T = self.N, self.T

        # foot stance
        in_stance = np.ones((4, N), dtype=bool)
        if foot_idx > -1:
            in_stance[foot_idx, 1:-1] = False

        # Compute the Convex Hull
        hull = ConvexHull(p_feet0_xy[in_stance[:,1]])


        # optimization
        prog = MathematicalProgram()

        # step size (time intervals between breakpoints)
        h = prog.NewContinuousVariables(N-1, "h")
        prog.SetInitialGuess(h, np.full(N-1, T / (N-1)))
        
    
        # position and velocity
        q = prog.NewContinuousVariables(self.n_q, N, "q")
        v = prog.NewContinuousVariables(self.n_v, N, 'v')

        # contact forces for 4 feet
        contact_forces = [prog.NewContinuousVariables(3,N,f'foot{i}_contact_force') for i in range(4)]

        # Center of Mass (CoM) variables and derivatives for ZMP like implementation
        com = prog.NewContinuousVariables(3, N, 'com')
        comdot = prog.NewContinuousVariables(3, N, 'comdot')
        comddot = prog.NewContinuousVariables(3, N, 'comddot')

        # angular momentum and derivative
        H = prog.NewContinuousVariables(3, N,"H")
        Hdot = prog.NewContinuousVariables(3, N,'Hdot')

        # context init
        context_q = [self.plant.CreateDefaultContext() for i in range(N)]
        context_ad_velocity = [self.plant_ad.CreateDefaultContext() for i in range(N)]
        context_ad_foot_position = [self.plant_ad.CreateDefaultContext() for i in range(N)]
        context_ad_com_momentum = [self.plant_ad.CreateDefaultContext() for i in range(N)]
        context_ad_contact = [self.plant_ad.CreateDefaultContext() for i in range(N)]


        ## constraints on time 
        prog.AddBoundingBoxConstraint(0.5 * T / N, 2 * T / N, h)
        prog.AddLinearConstraint(0.9 * T <= sum(h))
        prog.AddLinearConstraint(1.1 * T >= sum(h))

        # constriants over time span
        sum_force_z = 0.0
        average_force_z = self.total_mass * np.abs(self.gravity[-1]) / 4
        for foot in range(4):
            prog.SetInitialGuess(contact_forces[foot][2, :], np.full(N, average_force_z))
            prog.AddLinearConstraint(contact_forces[foot][0,0] <= self.friction_constant * contact_forces[foot][2,0])
            prog.AddLinearConstraint(-contact_forces[foot][0,0] <= self.friction_constant * contact_forces[foot][2,0])
            prog.AddLinearConstraint(contact_forces[foot][1,0] <= self.friction_constant * contact_forces[foot][2,0])
            prog.AddLinearConstraint(-contact_forces[foot][1,0] <= self.friction_constant * contact_forces[foot][2,0])
            prog.AddLinearConstraint(0.1 * self.total_mass * np.abs(self.gravity[-1]) <= contact_forces[foot][2,0])
            prog.AddLinearConstraint(contact_forces[foot][2,0] <= 2 * self.total_mass * np.abs(self.gravity[-1]))
            # sum_force_z +=  contact_forces[foot][2,0]
        # prog.AddLinearConstraint(self.total_mass * np.abs(self.gravity[-1]) <= sum_force_z)
        

        for n in range(1, N):
            # joint constraints on position
            prog.AddBoundingBoxConstraint(
                self.plant.GetPositionLowerLimits(), 
                self.plant.GetPositionUpperLimits(),
                q[:, n]
            )
            # joint constraints on velocity
            prog.AddBoundingBoxConstraint(
                self.plant.GetVelocityLowerLimits(),
                self.plant.GetVelocityUpperLimits(),
                v[:, n]
            )

            # unit quaternion constraint from drake
            AddUnitQuaternionConstraintOnPlant(
                self.plant, 
                q[:, n], 
                prog
            )

            # body orientation
            prog.AddConstraint(
                OrientationConstraint(
                    self.plant, 
                    self.body_frame, 
                    RotationMatrix(), 
                    self.plant.world_frame(), 
                    RotationMatrix(), 
                    30/180*np.pi, 
                    context_q[n]
                ),
                q[:,n]
            )

            # height constraints
            prog.AddBoundingBoxConstraint(
                self.z_target-0.15, 
                self.z_target+0.15, 
                q[6, n]
            )

            # friction cone
            for foot in range(4):
                prog.AddLinearConstraint(contact_forces[foot][0,n] <= self.friction_constant * contact_forces[foot][2,n])
                prog.AddLinearConstraint(-contact_forces[foot][0,n] <= self.friction_constant * contact_forces[foot][2,n])
                prog.AddLinearConstraint(contact_forces[foot][1,n] <= self.friction_constant * contact_forces[foot][2,n])
                prog.AddLinearConstraint(-contact_forces[foot][1,n] <= self.friction_constant * contact_forces[foot][2,n])
                prog.AddLinearConstraint(0 <= contact_forces[foot][2,n])
                prog.AddLinearConstraint(
                    contact_forces[foot][2,n] <= in_stance[foot, n] * 2 * self.total_mass * np.abs(self.gravity[-1])
                )
            # contact constriants
                if  in_stance[foot,n]:
                    # check if z==0
                    # prog.AddConstraint(
                    #     PositionConstraint(
                    #         self.plant,
                    #         self.plant.world_frame(),
                    #         [-np.inf, -np.inf, 0],
                    #         [np.inf, np.inf, 0],
                    #         self.foot_frames[foot],
                    #         self.p_FootFootTip,
                    #         context_q[n]
                    #     ),
                    #     vars=q[:,n]
                    # )
                    # do not move during stance
                    if n > 0 and in_stance[foot,n-1]:
                        custom_constraint = partial(
                            contact_constraints, 
                            plant=self.plant, 
                            plant_ad=self.plant_ad, 
                            context=context_q,
                            context_ad=context_ad_contact,
                            time_index=n-1,
                            frame=self.foot_frames[foot],
                            frame_ad=self.foot_frames_ad[foot],
                            p_foot=self.p_FootFootTip
                        )
                        prog.AddConstraint(
                            custom_constraint,
                            lb=np.zeros(3),
                            ub=np.zeros(3),
                            vars=np.concatenate([q[:,n-1], q[:,n]])
                        )
                else:#swing foot
                    if n == N//2:
                        min_clearence = self.min_clearence_high
                    else:
                        min_clearence = self.min_clearence_low

                    prog.AddConstraint(
                        PositionConstraint(
                            self.plant,
                            self.plant.world_frame(),
                            [-np.inf, -np.inf, min_clearence],
                            [np.inf, np.inf, np.inf],
                            self.foot_frames[foot],
                            self.p_FootFootTip,
                            context_q[n]
                        ),
                        vars=q[:, n]
                    )
                

            # matching: com and angualr momentum
            custom_constraint = partial(
                momentum_matching_constraints, 
                plant=self.plant, 
                plant_ad=self.plant_ad, 
                context=context_q,
                context_ad=context_ad_com_momentum,
                time_index=n,
                model=self.model
            )
            prog.AddConstraint(
                custom_constraint,
                lb=np.zeros(6),
                ub=np.zeros(6),
                vars=np.concatenate([q[:,n], v[:,n], com[:,n], H[:,n]]),
            )
            

            # miss the last one: not necessary
            if n < N-1:
                # matching: velocity
                custom_constraint = partial(
                    velocity_matching_constraints, 
                    plant=self.plant, 
                    plant_ad=self.plant_ad, 
                    context=context_q,
                    context_ad=context_ad_velocity,
                    time_index=n,
                )
                prog.AddConstraint(
                    custom_constraint,
                    lb=np.zeros(self.n_v),
                    ub=np.zeros(self.n_v),
                    vars=np.concatenate([[h[n]], q[:,n], v[:,n], q[:,n+1]]),
                )
            
                # angular momentum vs force
                custom_constraint = partial(
                    angular_momentum_constraints, 
                    plant=self.plant, 
                    plant_ad=self.plant_ad, 
                    context=context_q,
                    context_ad=context_ad_foot_position,
                    time_index=n,
                    foot_frames=self.foot_frames,
                    foot_frames_ad=self.foot_frames_ad,
                    p_foot=self.p_FootFootTip
                )
                Fn = np.concatenate([contact_forces[i][:, n] for i in range(4)])
                prog.AddConstraint(
                    custom_constraint,
                    lb=np.zeros(3),
                    ub=np.zeros(3),
                    vars=np.concatenate([q[:,n], com[:,n], Hdot[:,n], Fn]),
                )

                # dynamics
                # comdot
                prog.AddConstraint(eq(com[:,n+1], com[:,n] + h[n]*comdot[:,n+1]))
                
                # comddot
                prog.AddConstraint(eq(comdot[:,n+1], comdot[:,n] + h[n]*comddot[:,n+1]))
                
                # angular momentum
                prog.AddConstraint(eq(H[:,n+1],  H[:,n] + h[n]*Hdot[:,n+1]))

                #  total force : ma = F + mg
                prog.AddConstraint(
                    eq(self.total_mass * comddot[:, n], 
                        sum([contact_forces[i][:, n] for i in range(4)]) + self.total_mass * self.gravity
                    )
                )

                #  CONVEX HULL CONSTRIANTS
                for eq_plane in hull.equations:
                        a, b, c_const = eq_plane
                        # Constrain the CoM (x, y) to be on the "inside" of this line
                        prog.AddLinearConstraint(a * com[0, n] + b * com[1, n] + c_const <= 0)

            # interpolate 
            q_interpol = q0 + (q_end - q0) * n / (N-1)
            com_interpol = com0 + (com_end - com0) * n / (N-1)

            # initial guess
            prog.SetInitialGuess(q[:, n], q_interpol)          
            prog.SetInitialGuess(com[:, n], com_interpol)

            # Running costs:
            prog.AddQuadraticErrorCost(
                    self.Q_q, 
                    q_interpol, 
                    q[:, n]
                )
            prog.AddQuadraticErrorCost(
                    self.Q_v, 
                    self.v_desired, 
                    v[:, n]
                )
 
            prog.AddQuadraticErrorCost(
                    np.diag([10,10]), 
                    np.mean(p_feet0_xy[:2],axis=0), 
                    com[:2, n]
                )  
            

        # HARD CONSTRAINTS
        # zero velocity at the begining
        prog.AddBoundingBoxConstraint(0,0,comdot[2,0])
        prog.AddLinearConstraint(eq(v[:, 0], self.v_desired)) 
        prog.AddLinearConstraint(eq(v[:, -1], self.v_desired)) 
        prog.AddLinearConstraint(eq(q[:7, -1], q_end[:7]))

        # inital condition is exact
        prog.AddLinearConstraint(eq(q[:, 0], q0))

        # SOFT CONSTRAINTS
        prog.AddBoundingBoxConstraint(0.2+self.target_height, +self.target_height+1.0, com[2,:])
        # prog.AddBoundingBoxConstraint(0, np.inf, comdot[0,:])

        # Start and Final costs:
        prog.AddQuadraticErrorCost(
                    100*self.Q_q, 
                    q_end, 
                    q[:, -1]
                )
        # prog.AddQuadraticErrorCost(10*np.diag(), q0, q[:, 0])
        # prog.AddQuadraticErrorCost(10*np.diag(q_cost), q_end, q[:, N-1])

        # solve
        snopt = SnoptSolver().solver_id()
        prog.SetSolverOption(snopt, "Iterations Limits", 1e6)
        prog.SetSolverOption(snopt, "Major Iterations Limit", 1e3)
        prog.SetSolverOption(snopt, "Major Feasibility Tolerance", 5e-6)
        prog.SetSolverOption(snopt, "Major Optimality Tolerance", 1e-4)
        prog.SetSolverOption(snopt, "Superbasics limit", 2000)
        prog.SetSolverOption(snopt, "Linesearch tolerance", 0.9)
        # prog.SetSolverOption(snopt, 'Print file', 'snopt.out')

        start_time = time.time()
        result = Solve(prog)
        print("trajopt time = ", time.time() - start_time)
        
        print(result.get_solver_id().name())   
        print(result.is_success())  

        t_sol = np.cumsum(np.concatenate(([0], result.GetSolution(h))))
        q_sol = PiecewisePolynomial.CubicShapePreserving(t_sol, result.GetSolution(q))
        v_sol = PiecewisePolynomial.CubicShapePreserving(t_sol, result.GetSolution(v))

        # Extract Contact Forces
        # contact_forces is a list of 4 (3,N) variables. 
        # We stack them into a (12, N) array: [Force_foot0; Force_foot1; ...]
        f_val = np.zeros((12, len(t_sol)))
        for i in range(4):
            f_val[3*i : 3*i+3, :] = result.GetSolution(contact_forces[i])
        
        # Create a trajectory for forces
        f_sol = PiecewisePolynomial.CubicShapePreserving(t_sol, f_val)

        stance_sol = PiecewisePolynomial.ZeroOrderHold(t_sol, in_stance)



        return t_sol, q_sol, v_sol, q_end, f_sol, stance_sol, f_val
    

    def generate_foot_trajectories(self, context=None):
        """
        Generates the step-by-step trajectory for each foot to advance a specific distance.
        
        Args:
            current_feet: (4, 2) numpy array of current positions [RB, RF, LF, LB]
            move_distance: (float) Distance to move forward from the front feet.
            step_length: (float) Stride length for each step.
            
        Returns:
            step positions Four (N, 4, 2) numpy arrays.
            order in drake: [LF, RF, LB, RB]
        """
        if context is not None:
            self.context = context

        current_feet = np.zeros((4,2))
        for i in range(4):
                p_WF = self.plant.CalcPointsPositions(
                    self.context, 
                    self.foot_frames[i],      # The lower leg frame
                    self.p_FootFootTip,         # The offset to the tip [0, 0, -0.3725]
                    self.plant.world_frame()  # Target frame (World)
                )
                current_feet[i,:] = p_WF.flatten()[:2]


        # Dimensions
        # 0=RB, 1=RF, 2=LF, 3=LB
        # Gait Sequence: LF(2) -> RB(0) -> RF(1) -> LB(3)
        # gait_sequence = [0, 3, 1, 2] 
        
        # Initialize separate lists for each foot's history
        # Start with the initial position
        history_rb = [current_feet[3].copy()]
        history_rf = [current_feet[1].copy()]
        history_lf = [current_feet[0].copy()]
        history_lb = [current_feet[2].copy()]
        
        # Track the current state of the feet to update iteratively
        sim_feet = current_feet.copy()
        
        # Calculate how many full gait cycles are needed
        # Each cycle advances the robot by 1 * step_length
        # We want to move 'move_distance' meters.
        # We assume 'move_distance' is the goal for the FRONT feet.
        
        current_dist = 0.0
        
        # Loop until we have covered the distance
        while current_dist < self.move_distance:
            
            for foot_idx in self.gait_sequence:
                # Update the position of the active foot
                # We assume forward movement is along +X axis
                sim_feet[foot_idx, 0] += self.step_length
                
                # Append the NEW state of all feet to their respective histories
                history_rb.append(sim_feet[3].copy())
                history_rf.append(sim_feet[1].copy())
                history_lf.append(sim_feet[0].copy())
                history_lb.append(sim_feet[2].copy())
                
                # Check if we should stop early?
                # The prompt says "move x meter", usually implies the whole cycle completes 
                # or at least until the front feet reach the target.
                
            current_dist += self.step_length

        # Convert to Numpy Arrays (N, 2)
        traj_rb = np.array(history_rb)
        traj_rf = np.array(history_rf)
        traj_lf = np.array(history_lf)
        traj_lb = np.array(history_lb)

        step_positions = np.stack([traj_lf, traj_rf, traj_lb, traj_rb], axis=1)
        
        return step_positions
    
    def init_stabilization(self, context=None,rpy_target=[0,0,0]):
        if context is not None:
            self.context = context


        self.z_target=self.init_stab_xyz[-1]


        # feet position
        feet_pos_xy = np.zeros((4,2))
        for i in range(4):
                p_WF = self.plant.CalcPointsPositions(
                    self.context, 
                    self.foot_frames[i],     
                    self.p_FootFootTip,         
                    self.plant.world_frame()  
                )
                feet_pos_xy[i,:] = p_WF.flatten()[:2]
        # feet_pos_xyz[0,1] += 0.05
        # feet_pos_xyz[1,1] -= 0.05
        # feet_pos_xyz[2,1] += 0.05
        # feet_pos_xyz[3,1] -= 0.05 
        feet_pos_xyz = np.hstack([feet_pos_xy, self.target_height*np.ones((4,1))])
        print('Init stabilizer current foot placement:')
        print(feet_pos_xyz)
        # stabilize spot
        q_init = self.ik_from_pos(
            feet_pos_xyz,
            self.context,  
            xyz_target=self.init_stab_xyz, 
            rpy_target=rpy_target,
            )
        v_init = np.zeros(self.n_v)

        q_traj = PiecewisePolynomial.ZeroOrderHold(
            [0,self.init_stabilization_time], 
            np.column_stack([q_init, q_init])
        )

        v_traj = PiecewisePolynomial.ZeroOrderHold(
            [0,self.init_stabilization_time], 
            np.column_stack([v_init, v_init])
        )

        return q_traj, v_traj

    def stabilize_current(self, context=None):
        """
        Creates a stabilization trajectory based on the robot's CURRENT state.
        1. Fixes feet at their current measured locations (prevents drift).
        2. Moves Body XY to the centroid of the feet (stability).
        3. Moves Body Z to z_target.
        4. Levels the body (Roll=0, Pitch=0) but preserves current Yaw.
        """
        if context is not None:
            self.context = context
        
        # Get current state and normalize quaternion (Fixing the RuntimeError)
        q_curr = self.plant.GetPositions(self.context)
        # quat_vals = q_curr[:4]
        # quat_norm = np.linalg.norm(quat_vals)
        # if quat_norm > 1e-6:
        #     quat_vals = quat_vals / quat_norm
        # else:
        #     quat_vals = np.array([1.0, 0.0, 0.0, 0.0]) # Fallback
            
        # # Find current Yaw (to preserve heading)
        # rpy_curr = RollPitchYaw(Quaternion(quat_vals))
        # current_yaw = rpy_curr.yaw_angle()

        # 3. Find current Foot Positions in World Frame
        current_feet_pos = np.zeros((4, 3))
        for i in range(4):
            p_WF = self.plant.CalcPointsPositions(
                self.context,
                self.foot_frames[i],
                self.p_FootFootTip,
                self.plant.world_frame()
            ).flatten()
            current_feet_pos[i] = p_WF

        # Compute optimal Body XY (Centroid of feet)
        # This ensures the CoM is exactly in the middle of the support polygon
        feet_centroid = np.mean(current_feet_pos, axis=0)
        # Target Body Position: [Centroid_X, Centroid_Y, Target_Height]
        # Solve IK to find the joints that satisfy this
        q_stable = self.ik_from_pos(
            current_feet_pos, # Pass the full (4,3) array
            xyz_target=[q_curr[4], feet_centroid[1], self.z_target], 
            rpy_target=[0.0, 0.0, 0.0]
        )[7:]
        

        # 6. Create Trajectory (Zero Velocity Hold)
        # Output format: [q; v] stacked
        v_stable = np.zeros_like(q_stable)
        qv_stable = np.vstack([q_stable, v_stable])
        
        # Create a trajectory that simply holds this value constant
        # Duration can be small or matches init_stabilization_time
        qv_traj = PiecewisePolynomial.ZeroOrderHold(
            [0, self.init_stabilization_time], 
            [qv_stable, qv_stable] 
        )

        return qv_traj


