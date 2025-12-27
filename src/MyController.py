import numpy as np
from pydrake.all import LeafSystem, JacobianWrtVariable


class HybridController(LeafSystem):
    def __init__(
            self, 
            plant, 
            Kp=np.eye(3), 
            Kd=np.eye(3),
            init_stabilization_time=0.2,
            Kp_stab=10,
            Kd_stab=1,
            p_FootFootTip=np.array([0.0, 0.0, -0.379]),
            ):
        
        super().__init__()
        self.plant = plant
        
        # context for computations
        self.context = plant.CreateDefaultContext()
        self.context_desired = plant.CreateDefaultContext()

        # num of states
        self.n_q = plant.num_positions()
        self.n_v = plant.num_velocities()
        self.n_u = plant.num_actuators()

        # parameters for cartesion PD
        self.Kp = Kp
        self.Kd = Kd
        self.Kp_stab = Kp_stab
        self.Kd_stab = Kd_stab
        self.init_stabilization_time = init_stabilization_time
        self.p_FootFootTip= p_FootFootTip

        # PORTS
        # input
        self.input_port_state = self.DeclareVectorInputPort('state', self.n_q + self.n_v)
        # desired state
        self.input_port_desired_q = self.DeclareVectorInputPort('desired_q', self.n_q)
        self.input_port_desired_v = self.DeclareVectorInputPort('desired_v', self.n_v)
        # forces
        self.input_port_force = self.DeclareVectorInputPort('force', self.n_u)
        # feet stance/swing
        self.input_port_stance = self.DeclareVectorInputPort('stance', 4)
        # output
        self.output_port_torque = self.DeclareVectorOutputPort('torque', self.n_u, self.CalcOutput)


        self.foot_frames = [
            plant.GetFrameByName("front_left_lower_leg"),
            plant.GetFrameByName("front_right_lower_leg"),
            plant.GetFrameByName("rear_left_lower_leg"),
            plant.GetFrameByName("rear_right_lower_leg"),
        ]
    
    def CalcOutput(self, context, output):
        # read inputs 
        curr_state = self.input_port_state.Eval(context)

        desired_q = self.input_port_desired_q.Eval(context)
        desired_v = self.input_port_desired_v.Eval(context)
        force = self.input_port_force.Eval(context)
        is_stance = self.input_port_stance.Eval(context)

        self.plant.SetPositionsAndVelocities(self.context, curr_state)

        # output torque
        torque = np.zeros(self.n_v)

        
    
        
        # Extract joint parts
        q_curr_joints = curr_state[:self.n_q][-self.n_u:]
        v_curr_joints = curr_state[self.n_q:][-self.n_u:]
        q_desired_joints = desired_q[-self.n_u:]
        v_desired_joints = desired_v[-self.n_u:]
        
        # inital stabilization on the ground
        if context.get_time() < self.init_stabilization_time:
            torque[-self.n_u:] = 1000 * (q_desired_joints - q_curr_joints) + 100 * (v_desired_joints - v_curr_joints)
        else:
            torque[-self.n_u:] = self.Kp_stab * (q_desired_joints - q_curr_joints) + self.Kd_stab * (v_desired_joints - v_curr_joints)
            # go over each leg
            for i in range(4):
                J = self.plant.CalcJacobianTranslationalVelocity(
                    self.context,
                    JacobianWrtVariable.kV,
                    self.foot_frames[i],
                    self.p_FootFootTip,
                    self.plant.world_frame(),
                    self.plant.world_frame(),
                )

                if is_stance[i]:
                    # stance feet torque = J^Tf
                    F_foot = force[3*i:3*i+3]
                    torque += -J.T @ F_foot
                
                else:
                    # swing foot use Cartesian PD controller

                    # CURRENT
                    # position in cartesian
                    p_foot = self.plant.CalcPointsPositions(
                        self.context, 
                        self.foot_frames[i],     
                        self.p_FootFootTip,        
                        self.plant.world_frame() 
                    ).flatten()

                    # current velocity in cartesian
                    v_foot = J @ curr_state[-self.n_v:]

                    # DESIRED
                    self.plant.SetPositions(self.context_desired, desired_q)
                    self.plant.SetVelocities(self.context_desired, desired_v)

                    p_foot_desired = self.plant.CalcPointsPositions(
                        self.context_desired, 
                        self.foot_frames[i], 
                        self.p_FootFootTip, 
                        self.plant.world_frame()
                    ).flatten()

                    # jacobian for desired
                    J_desired = self.plant.CalcJacobianTranslationalVelocity(
                        self.context_desired, 
                        JacobianWrtVariable.kV,
                        self.foot_frames[i], 
                        self.p_FootFootTip, 
                        self.plant.world_frame(), 
                        self.plant.world_frame()
                    )
                    # translational velocity
                    v_foot_desired = J_desired @ desired_v

                    # PD controller force
                    F_pd = self.Kp @ (p_foot_desired - p_foot) + self.Kd @ (v_foot_desired - v_foot)

                    # map to torque
                    # torque[-self.n_u+i*3:-self.n_u+i*3+3] = 0
                    torque += J.T @ F_pd
            # get last n_us since body is floating
            tor = torque[-self.n_u:]
        output.SetFromVector(torque[-self.n_u:])
        



class DynamicTrajectorySource(LeafSystem):
    """
    A custom system that outputs trajectory values.
    Can be updated during simulation (unlike TrajectorySource).
    """
    
    def __init__(self, output_size=24):
        """
        Constructor - called when you do:
        my_system = DynamicTrajectorySource(24)
        
        Args:
            output_size: How many numbers this system outputs (e.g., 24 for 12 positions + 12 velocities)
        """
        
        # REQUIRED: Call parent class constructor
        # This sets up all the Drake infrastructure
        LeafSystem.__init__(self)
        
        # Store the output size as an instance variable
        # self._xxx is a Python convention for "private" variables
        self._output_size = output_size
        
        # Initialize trajectory storage (starts as None = no trajectory)
        self._trajectory = None
        
        # Time offset - useful when trajectory starts at t=0 but simulation is at t=5
        self._t_offset = 0.0
        
        # DECLARE AN OUTPUT PORT
        # This tells Drake: "This system has an output port called 'desired_state'"
        # 
        # Arguments:
        #   "desired_state" - name of the port (for debugging)
        #   output_size     - how many numbers (e.g., 24)
        #   self.CalcOutput - function to call when someone reads this port
        #
        # Think of it like: "When someone asks for my output, run CalcOutput()"
        self.DeclareVectorOutputPort(
            "desired_state",      # port name
            output_size,          # size of output vector
            self.CalcOutput       # callback function
        )
    
    def set_trajectory(self, trajectory, t_offset=0.0):
        """
        Update the trajectory - can be called anytime, even during simulation!
        
        Args:
            trajectory: A PiecewisePolynomial trajectory
            t_offset: When does this trajectory "start" in simulation time
        
        Example:
            # At simulation time t=5, set a new trajectory that starts at t=0
            my_system.set_trajectory(new_traj, t_offset=5.0)
        """
        self._trajectory = trajectory
        self._t_offset = t_offset
    
    def CalcOutput(self, context, output):
        """
        CALLBACK FUNCTION - Drake calls this automatically when output is needed.
        
        YOU DON'T CALL THIS DIRECTLY! Drake calls it.
        
        Args:
            context: Contains the current time and state
                     Think of it as "what time is it and what's happening"
            
            output:  A vector to fill with values
                     You MUST set values in this, don't create a new one
        
        This function runs every time step (or whenever output is requested).
        """
        
        # Get current simulation time from context
        # context.get_time() returns a float like 5.32
        t = context.get_time() - self._t_offset
        
        # Check if we have a trajectory set
        if self._trajectory is not None:
            # Clamp time to trajectory bounds
            # If trajectory is defined from t=0 to t=2, and we ask for t=5,
            # this returns the value at t=2 (end of trajectory)
            t_clamped = np.clip(
                t, 
                self._trajectory.start_time(),  # e.g., 0.0
                self._trajectory.end_time()      # e.g., 2.0
            )
            
            # Get trajectory value at this time
            # .value(t) returns a 2D array like [[x1], [x2], ...]
            # .flatten() converts to 1D array [x1, x2, ...]
            value = self._trajectory.value(t_clamped).flatten()
            
            # SET the output (don't create new, use SetFromVector)
            output.SetFromVector(value)
        else:
            # No trajectory set - output zeros
            output.SetFromVector(np.zeros(self._output_size))
    


    

