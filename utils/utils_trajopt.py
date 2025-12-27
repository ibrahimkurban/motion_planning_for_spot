import numpy as np
from pydrake.all import (
    AutoDiffXd,
    ExtractGradient,
)

def AD_equal(a, b):
    return np.array_equal(a, b) and np.array_equal(
        ExtractGradient(a), ExtractGradient(b)
    )



def velocity_matching_constraints(vars, plant, plant_ad, context, context_ad, time_index):
    # extract variables
    n_q, n_v = plant.num_positions(), plant.num_velocities()
    h, q, v, qn = np.split(vars, [1,1+n_q, 1+n_q+n_v])
    
    # if demands derivatives
    if isinstance(vars[0], AutoDiffXd):
        # check if up to date
        if not AD_equal(q, plant_ad.GetPositions(context_ad[time_index])):
            plant_ad.SetPositions(context_ad[time_index], q)
        # get velocity from context 
        v_context = plant_ad.MapQDotToVelocity(context_ad[time_index], (qn - q)/h)
    else:
        # check if up to date
        if not np.array_equal(q, plant.GetPositions(context[time_index])):
            plant.SetPositions(context[time_index])
        # get veloctiy
        v_context = plant.MapQDotToVelocity(context[time_index], (qn - q)/h)

    # return error
    return v  - v_context


def momentum_matching_constraints(vars, plant, plant_ad, context, context_ad, time_index, model):
    n_qv = plant.num_multibody_states()
    qv, com, H = np.split(vars, [n_qv,n_qv+3])
    # if demands derivatives
    if isinstance(vars[0], AutoDiffXd):
        # check if up to date
        if not AD_equal(qv, plant_ad.GetPositionsAndVelocities(context_ad[time_index])):
            plant_ad.SetPositionsAndVelocities(context_ad[time_index], qv)
        # get com and momentum from context 
        com_context = plant_ad.CalcCenterOfMassPositionInWorld(context_ad[time_index], [model])
        # TODO: com_context or com
        H_context = plant_ad.CalcSpatialMomentumInWorldAboutPoint(
                                    context_ad[time_index], 
                                    [model], 
                                    com_context
                                ).rotational()
    else:
        # check if up to date
        if not np.array_equal(qv, plant.GetPositionsAndVelocities(context[time_index])):
            plant.SetPositionsAndVelocities(context[time_index], qv)
        # get com and H
        com_context = plant.CalcCenterOfMassPositionInWorld(context[time_index], [model])
        H_context = plant.CalcSpatialMomentumInWorldAboutPoint(
                                    context[time_index], 
                                    [model], 
                                    com_context
                                ).rotational()
    
    # return error
    return np.concatenate([com - com_context, H - H_context])





def angular_momentum_constraints(vars, plant, plant_ad, context, context_ad, time_index, foot_frames, foot_frames_ad, p_foot):
    n_q = plant.num_positions()
    q, com, Hdot, F = np.split(vars, [n_q,n_q+3, n_q+6])
    F = F.reshape(3,4,order='F')
    
    # if demands derivatives
    if isinstance(vars[0], AutoDiffXd):
        # check if up to date
        if not AD_equal(q, plant_ad.GetPositions(context_ad[time_index])):
            plant_ad.SetPositions(context_ad[time_index], q)
        # compute total torque on com from each foot
        torque = np.zeros(3, dtype=AutoDiffXd)
        for i in range(4):
            p_WF = plant_ad.CalcPointsPositions(
                    context_ad[time_index], 
                    foot_frames_ad[i],
                    p_foot,
                    plant_ad.world_frame(),
                )
            torque += np.cross(p_WF.flatten() - com, F[:,i])

            
    else:
        ## check if up to date
        if not np.array_equal(q, plant.GetPositions(context[time_index])):
            plant.SetPositions(context[time_index], q)
        # compute total torque on com from each foot
        torque = np.zeros(3)
        for i in range(4):
            p_WF = plant.CalcPointPositions(
                    context[time_index], 
                    foot_frames[i],
                    p_foot,
                    plant.world_frame(),
                )
            torque += np.cross(p_WF.flatten() - com, F[:,i])
    
    # return error
    return Hdot - torque



def contact_constraints(vars, plant, plant_ad, context, context_ad, time_index, frame, frame_ad, p_foot):
    n_q = plant.num_positions()
    q, qn = np.split(vars, [n_q])

    # if demands derivatives
    if isinstance(vars[0], AutoDiffXd):
        # check if up to date
        if not AD_equal(q, plant_ad.GetPositions(context_ad[time_index])):
            plant_ad.SetPositions(context_ad[time_index], q)
        if not AD_equal(qn, plant_ad.GetPositions(context_ad[time_index+1])):
            plant_ad.SetPositions(context_ad[time_index+1], qn)

        # compute position of foot
        p_WF = plant_ad.CalcPointsPositions(
                    context_ad[time_index], 
                    frame_ad, 
                    p_foot,
                    plant_ad.world_frame(),
                )
        
        p_WF_n = plant_ad.CalcPointsPositions(
                    context_ad[time_index+1], 
                    frame_ad, 
                    p_foot,
                    plant_ad.world_frame(),
                )
            
    else:
        ## check if up to date
        if not np.array_equal(q, plant.GetPositions(context[time_index])):
            plant.SetPositions(context[time_index], q)
        if not np.array_equal(qn, plant.GetPositions(context[time_index+1])):
            plant.SetPositions(context[time_index+1], qn)
            
        # compute position of foot
        p_WF = plant.CalcPointsPositions(
                    context[time_index], 
                    frame, 
                    p_foot,
                    plant.world_frame(),
                )
        
        p_WF_n = plant.CalcPointsPositions(
                    context[time_index+1], 
                    frame, 
                    p_foot,
                    plant.world_frame(),
                )
    
    # return error
    return p_WF_n - p_WF