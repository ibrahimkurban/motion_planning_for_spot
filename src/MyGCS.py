import time
import numpy as np
from scipy.linalg import block_diag
from pydrake.all import GraphOfConvexSets, GraphOfConvexSetsOptions, HPolyhedron, Point, MosekSolver
from pydrake.solvers import CommonSolverOption
from src.MyTerrain import Terrain
from utils.utils_plot import animate_footstep_plan_gcs

class GCS(object):
    def __init__(
            self, 
            terrain,
            ideal_pose,
            step_span_prunning, 
            vertical_limits, 
            horizontal_limits, 
            gcs_solver,
            max_step_len=0.4,
            swing_order=[0,3,1,2],
            bridge_stone_wh=[0.01,0.01],
            bridge_x_shift=0, 
            copy_vertex_count=6,
            convex_relaxation=True, 
            gcs_preprocessing=False,
            gcs_max_rounded_paths=100,
            gcs_max_rounding_trials=100,
            mosek_interpnt_max_iter=1000,
            gcs_print_console=False,
            mosek_tol_pfeas=1e-4,
            mosek_tol_dfeas=1e-4,
            mosek_tol_rel_gap=1e-3, 
        ):

        self.terrain = terrain
        self.step_span_prunning = step_span_prunning
        self.vertical_limits = vertical_limits
        self.horizontal_limits = horizontal_limits
        self.gcs_solver = gcs_solver
        self.max_step_len=max_step_len
        self.swing_order=swing_order
        self.bridge_stone_wh=bridge_stone_wh
        self.bridge_x_shift = bridge_x_shift
        self.copy_vertex_count=copy_vertex_count
        self.convex_relaxation = convex_relaxation
        self.gcs_preprocessing = gcs_preprocessing
        self.gcs_max_rounded_paths = gcs_max_rounded_paths
        self.gcs_max_rounding_trials = gcs_max_rounding_trials
        self.mosek_interpnt_max_iter = mosek_interpnt_max_iter
        self.gcs_print_console = gcs_print_console
        self.mosek_tol_pfeas = mosek_tol_pfeas
        self.mosek_tol_dfeas = mosek_tol_dfeas
        self.mosek_tol_rel_gap = mosek_tol_rel_gap


        # create a terrain for planning
        # shring the stepping stones for safe foot placement
        self.terrain_gcs = Terrain(
            init_center_xy=terrain.init_center_xy,
            target_center_xy=terrain.target_center_xy,
            init_wh=terrain.init_wh,
            target_wh=terrain.target_wh,
            num_bridges_y=terrain.num_bridges_y,
            num_stones_in_bridge=terrain.num_stones_in_bridge,
            bridge_stone_wh=bridge_stone_wh,
            bridges_dist_y=terrain.bridges_dist_y,
            bridges_centerline_y=terrain.bridges_centerline_y,
            rand_radius=terrain.rand_radius,
            bridge_x_shift=self.bridge_x_shift,
            rand_seed=terrain.rand_seed,
        )

        # ideal poses
        # inital stone
        self.start_pos = ideal_pose.flatten()
        # targe stone
        self.end_pos = ( ideal_pose + terrain.target.center).flatten()

    def GCS_footstep_planner(self):

        GCS = GraphOfConvexSets()

        GCS, V, E_list, vertex_map = self.add_vertex_and_edges_for_set_copies(
            GCS, 
            self.terrain_gcs, 
            self.step_span_prunning, 
            self.start_pos,
            self.end_pos
        )

        E_list = self.add_edges_towards_parent_vertices(
            E_list, 
            vertex_map, 
            self.swing_order
        )
        
        GCS, E = self.add_edge_constraints_costs(
            GCS, 
            E_list, 
            self.vertical_limits, 
            self.horizontal_limits, 
            self.max_step_len
        )

        options = GraphOfConvexSetsOptions()
        options.preprocessing = self.gcs_preprocessing
        options.max_rounded_paths = self.gcs_max_rounded_paths
        options.max_rounding_trials = self.gcs_max_rounding_trials
        options.convex_relaxation = self.convex_relaxation
        options.solver = self.gcs_solver
        options.solver_options.SetOption(CommonSolverOption.kPrintToConsole, self.gcs_print_console)

        options.solver_options.SetOption(
            MosekSolver.id(), 
            "MSK_IPAR_INTPNT_MAX_ITERATIONS", 
            self.mosek_interpnt_max_iter
            )

        # relax Feasibility Tolerance (Allow constraints to be off by 1e-4 instead of 1e-8)
        options.solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_INTPNT_CO_TOL_PFEAS", self.mosek_tol_pfeas)
        options.solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_INTPNT_CO_TOL_DFEAS", self.mosek_tol_dfeas)

        # relax Gap Tolerance (Accept solution when Primal/Dual gap is < 1e-4)
        options.solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", self.mosek_tol_rel_gap)


        start_time = time.time()
        print("GCS runs...")
        result = GCS.SolveShortestPath(V[0], V[1], options)
        print("num stones = ", len(self.terrain_gcs.stepping_stones), ", time = ", time.time() - start_time)
        

        if not result.is_success():
            print("GCS failed to find a path.")
            return None, None, None, None, 0

        print("GCS successful. Extracting path...")

        # 7. Extract the Path and Positions
        path, path_position = self.extract_path(V, E, result)
        
        # Calculate Step Cost (number of transitions)
        path_cost = len(path) - 1
        
        # 8. Convert 8D path into individual foot trajectories
        pos_fl, pos_fr, pos_rl, pos_rr = self.footstep2position(path_position)

        animate_footstep_plan_gcs(
            self.terrain_gcs, 
            pos_fl, 
            pos_fr, 
            pos_rl, 
            pos_rr, 
            self.vertical_limits, 
            self.horizontal_limits
        )

        return pos_fl, pos_fr, pos_rl, pos_rr, path_cost
    
    
    def get_valid_stone_combinations(self, stones,step_span):
        """
        Returns a list of tuples (fl, fr, rl, rr) representing valid stone assignments.
        Prunes based on robot Body Length and Body Width limits.
        """

        valid_combinations = []
        num_stones = len(stones)
        x_max, x_min, y_max, y_min = step_span
        
        for fl in range(num_stones):
            for fr in range(num_stones):
                for rl in range(num_stones):
                    for rr in range(num_stones):
                        
                        # single foot on bridge stones
                        if fl == fr and (fr > 0 and fr <num_stones-1):
                            continue
                        if fl == rl and (fl > 0 and fl <num_stones-1):
                            continue
                        if fr == rr and (fr > 0 and fr <num_stones-1):
                            continue
                        if rl == rr and (rr > 0 and rr <num_stones-1):
                            continue
                        
                        # long axis
                        # fl - rl
                        if stones[fl].min_dist(stones[rl]) >= x_max:
                            continue
                        # fr - rr
                        if stones[fr].min_dist(stones[rr]) >= x_max:
                            continue

                        # short axis
                        # fl - fr
                        if stones[fl].min_dist(stones[fr]) >= y_max:
                            continue
                        # rl - rr
                        if stones[rl].min_dist(stones[rr]) >= y_max:
                            continue
                            
                        # leg crossing front
                        if stones[fl].center[1] < stones[fr].center[1]: 
                            continue
                        # leg crossing rear
                        if stones[rl].center[1] < stones[rr].center[1]: 
                            continue

                        # fron legs are front left
                        if stones[fl].center[0] < stones[rl].center[0]:
                            continue
                        # fron legs are front right
                        if stones[fr].center[0] < stones[rr].center[0]:
                            continue
                        
                        # FOR BRIDGE STONES ONLY
                        # long axis
                        # fl - rl
                        if 0 < fl < num_stones-1 and 0 < rl < num_stones-1 and stones[fl].center[0] - stones[rl].center[0] <= x_min:
                            continue
                        # fr - rr
                        if 0 < fr < num_stones-1 and 0 < rr < num_stones-1 and stones[fr].center[0] - stones[rr].center[0] <= x_min:
                            continue
                        
                        # short axis
                        # fl - fr
                        if 0 < fl < num_stones-1 and 0 < fr < num_stones-1 and stones[fl].center[1] - stones[fr].center[1] <= y_min:
                            continue
                        # rl - rr
                        if 0 < rl < num_stones-1 and 0 < rr < num_stones-1 and stones[rl].center[1] - stones[rr].center[1] <= y_min:
                            continue


                        # If we pass all checks, it's a valid stance
                        valid_combinations.append((fl, fr, rl, rr))

        print(f"Pruning Complete: Reduced {num_stones**4} combinations to {len(valid_combinations)}.")
        print("Valid foot configurations: (fl, fr, rl, rr):")
        print(valid_combinations)
        return valid_combinations



    def add_vertex_and_edges_for_set_copies(self, GCS, terrain, step_span, start_pos, end_pos):
        
        # stones
        stones = terrain.stepping_stones
        num_stones = len(stones)
        
        copy_count_max = self.copy_vertex_count #for start and end
        copy_count_low = 1 #for bridges

        V = []              # List of GCS Vertex objects
        E_list = []         # List of pairs [u, v] for INTERNAL edges (copy i -> copy i+1)
        vertex_map = {}     # Key: (fl_idx, fr_idx, rl_idx, rr_idx, swing_id, copy_id) -> Vertex Object

        # source target nodes
        V.append(GCS.AddVertex(Point(start_pos), "source"))
        V.append(GCS.AddVertex(Point(end_pos), "target"))
        # encode source and goal nodes
        vertex_map[(0,0,0,0,0,-1)] = V[0]
        vertex_map[(0,0,0,0,0,-2)] = V[1]
        
        # generate_vertices 
        for (fl, fr, rl, rr) in self.get_valid_stone_combinations(stones, step_span):
        
            # Cartesian product of the 4 stones
            H_poly = HPolyhedron(
                block_diag(stones[fl].A, stones[fr].A, stones[rl].A, stones[rr].A),
                np.concatenate([stones[fl].b, stones[fr].b, stones[rl].b, stones[rr].b])
            )

            # If any foot is on "initial" or "goal", use High Copies. Else 1.
            # stone 0 is initial, stone N-1 is goal
            is_start_or_end = (fl == 0 or fl == num_stones-1 or 
                            fr == 0 or fr == num_stones-1 or 
                            rl == 0 or rl == num_stones-1 or 
                            rr == 0 or rr == num_stones-1)
                            
            current_num_copies = copy_count_max if is_start_or_end else copy_count_low

            # Generate Nodes
            for swing_id in range(4):             
                for copy in range(current_num_copies):
                    
                    # create vertex add it to GCS and copy
                    V.append(GCS.AddVertex(H_poly, f"{fl}_{fr}_{rl}_{rr}_{swing_id}_{copy}"))
                    vertex_map[(fl, fr, rl, rr, swing_id, copy)] = V[-1]
                    # if copy > 0:
                    #     E_list.append([V[-2], V[-1]])


        # MANUAL SOURCE NAD TARGET ADDITION
        # add source to initial stone manually
        E_list.append([vertex_map[(0,0,0,0,0,-1)], vertex_map[(0,0,0,0,0,0)]])
        # add goal stone to target  manually node
        # from all copies
        for copy in range(copy_count_max):
            E_list.append([vertex_map[(
                    num_stones-1,num_stones-1,num_stones-1,num_stones-1,0,copy
                )], vertex_map[(0,0,0,0,0,-2)]])


        return GCS, V, E_list, vertex_map



    def add_edges_towards_parent_vertices(self, E_list, vertex_map, swing_order):

        print('Start creating an edge list...')
        # Define your gait order: 0=FL, 1=BR, 2=FR, 3=BL
        next_swing_map = {swing_order[i] : swing_order[i+1] for i in range(len(swing_order)-1)}
        next_swing_map[swing_order[-1]] = swing_order[0]

        # Iterate over all created vertices to find valid connections
        # use list(vertex_map.keys()) to avoid modifying the dict while iterating
        all_keys = list(vertex_map.keys())
        
        for key in all_keys:
            (fl, fr, rl, rr, swing_id, copy_id) = key
            # print('-'*20)
            # print('------outgoing key:', key)
            # ignore source and target nodes
            if copy_id < 0:
                continue
            
            # valid destinations
            for next_key in all_keys:
                (fl_next, fr_next, rl_next, rr_next, swing_id_next, copy_next) = next_key
                
                # ignore source and target nodes
                if copy_next < 0:
                    continue

                # next vertex must have the correct next swing_id
                if swing_id_next != next_swing_map[swing_id]:
                    continue
                    
                # if stones are same, copy should be increment by one
                if key[:4] == next_key[:4] and (copy_next != copy_id + 1):
                    continue
                
                # only one leg should differ
                if  sum(np.array(key[:4]) == np.array(next_key[:4])) < 3:
                    continue

                # at least one stone different, next_copy_id = 0
                if (key[:4] != next_key[:4]) and copy_next != 0:
                    continue

                # stance Feet/stone should remain same
                if any(key[id]!=next_key[id] for id in range(4) if id != swing_id):
                    continue
        
                # add the Edge
                # print('incoming key:', next_key)
                E_list.append([vertex_map[key], vertex_map[next_key]])
                
        return E_list


    def add_edge_constraints_costs(self, G, E_list, vertical_limits,horizontal_limits, step_len):
        
        print('Edge constraints and costs are being added...')
        E = []
        x_pair_map = {0:2, 1:3, 2:0, 3:1}
        y_pair_map = {0:1, 1:0, 2:3, 3:2}
        vertical_x_max, vertical_x_min, vertical_y_max, vertical_y_min = vertical_limits
        horizontal_x_max, horizontal_x_min, horizontal_y_max, horizontal_y_min = horizontal_limits

        for u, v in E_list:
            e = G.AddEdge(u, v)

            # Add kinematic constraints to the edges
            xu = e.xu()
            xv = e.xv()
            diff = xv - xu


            # source to stone
            if u.name() == 'source':
                for i in range(8):
                    e.AddConstraint(diff[i] == 0)
                E.append(e)
                continue
            
            # stone to targer
            if v.name() == 'target':
                for i in range(8):
                    e.AddConstraint(diff[i] == 0)
                E.append(e)
                continue

            swing = int(u.name().split("_")[4])
            # fl_next, fr_next, rl_next, rr_next, swing_next, copy_next = np.array(u.name().split("_"), dtype=int)
            
            for foot in range(4):
                # constraint swing foot
                if foot == swing:
                    x_foot = x_pair_map[foot]
                    y_foot = y_pair_map[foot]
                    horizontal_diff = xv[foot*2:foot*2+2] - xu[x_foot*2:x_foot*2+2]
                    vertical_diff = xv[foot*2:foot*2+2] - xu[y_foot*2:y_foot*2+2]
                    if foot > x_foot:
                        horizontal_diff *= -1
                    if foot > y_foot:
                        vertical_diff *= -1
                    e.AddConstraint(vertical_diff[0] <= vertical_x_max)
                    e.AddConstraint(vertical_diff[0] >= vertical_x_min)
                    e.AddConstraint(vertical_diff[1] <= vertical_y_max)
                    e.AddConstraint(vertical_diff[1] >= vertical_y_min)
                    e.AddConstraint(horizontal_diff[0] <= horizontal_x_max)
                    e.AddConstraint(horizontal_diff[0] >= horizontal_x_min)
                    e.AddConstraint(horizontal_diff[1] <= horizontal_y_max)
                    e.AddConstraint(horizontal_diff[1] >= horizontal_y_min)
                    e.AddConstraint(diff[foot*2] <= step_len)


                # fix stance feet
                else:
                    e.AddConstraint(diff[foot*2] == 0)
                    e.AddConstraint(diff[foot*2+1] == 0)

                    

            # Add constant edge cost
            if swing == 1 or swing == 2:
                e.AddCost(1 + 100*vertical_diff[0]**2 + horizontal_diff[1]**2)
            else:
                e.AddCost(1 + horizontal_diff[1]**2)
            # e.AddCost(1)
            E.append(e)

        return G, E


    def extract_path(self, V, E, result):
        
        V_dict = {}
        for i, v in enumerate(V):
            V_dict[v.name()] = i

        V_adj = np.zeros((len(V), len(V)))

        for e in E:
            # Check if edge is active
            if result.GetSolution(e.phi()) > 0.9:
                u_index = V_dict[e.u().name()]
                v_index = V_dict[e.v().name()]
                V_adj[u_index, v_index] = 1

        path = ["source"]
        path_count = 0
        
        while path[-1] != "target":
            u_name = path[-1]
            
            # Find the index of the active next step
            v_index = np.where(V_adj[V_dict[u_name], :] == 1)[0][0]
            path.append(V[v_index].name())
            path_count += 1
            if path_count > 100: # Increased limit slightly
                print("Abort path extraction: possible loops")
                break

        # Extract positions 8D vector
        path_position = np.zeros((len(path), 8))
        for i in range(len(path)):
            path_position[i, :] = result.GetSolution(V[V_dict[path[i]]].x())
        
        return path, path_position


    def footstep2position(self, planner_footstep):
        """
        Splits the (N, 8) path_position matrix into individual feet trajectories.
        Assumes order: FL, FR, RL, RR (based on your block_diag construction)
        """
        n_steps = planner_footstep.shape[0]

        # Initialize arrays
        pos_fl = np.zeros((n_steps, 2))
        pos_fr = np.zeros((n_steps, 2))
        pos_rl = np.zeros((n_steps, 2))
        pos_rr = np.zeros((n_steps, 2))

        for i in range(n_steps):
            # Slice the 8D vector [FLx, FLy, FRx, FRy, RLx, RLy, RRx, RRy]
            pos_fl[i, :] = planner_footstep[i, 0:2]
            pos_fr[i, :] = planner_footstep[i, 2:4]
            pos_rl[i, :] = planner_footstep[i, 4:6]
            pos_rr[i, :] = planner_footstep[i, 6:8]
        
        return pos_fl, pos_fr, pos_rl, pos_rr

