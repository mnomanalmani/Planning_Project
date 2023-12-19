import mujoco_py
import numpy as np 
import matplotlib.pyplot as plt 



## ------- Functions related to finite differencing -------------
delta_step= 5e-1
def gradient_fd(f, x):
  num_variables = len(x)
  out = [ ]

  f_0 = f(x)
  delta_x = delta_step

  for i in range(num_variables):
    x_copy = np.copy(x)
    x_copy[i] = x_copy[i] + delta_x
    f_delta = f(x_copy)
    out.append( (f_delta - f_0) / delta_x )

  return np.array(out)

def hessian_fd(f, x):
  num_variables = len(x)
  out = [ ]

  f_0 = gradient_fd(f, x)
  delta_x = delta_step

  for i in range(num_variables):
    x_copy = np.copy(x)
    x_copy[i] = x_copy[i] + delta_x
    f_delta = gradient_fd(f, x_copy)
    out.append( (f_delta - f_0) / delta_x )

  return np.array(out)



model= mujoco_py.load_model_from_path("./monkeyArm_current_scaled.xml")
sim= mujoco_py.MjSim(model)
data= sim.data 


#
#Define convenience functions for visualization
#

#

DEFAULT_SIZE= 500

viewer= None 
_viewers= {}
dt= 0.0001

metadata = {
    'render.modes': ['human', 'rgb_array', 'depth_array'],
    'video.frames_per_second': int(np.round(1.0 / dt))
}

def render(mode='human',
           width=DEFAULT_SIZE,
           height=DEFAULT_SIZE,
           camera_id=None,
           camera_name=None):
    if mode == 'rgb_array' or mode == 'depth_array':
        if camera_id is not None and camera_name is not None:
            raise ValueError("Both `camera_id` and `camera_name` cannot be"
                             " specified at the same time.")

        no_camera_specified = camera_name is None and camera_id is None
        if no_camera_specified:
            camera_name = 'track'

        if camera_id is None and camera_name in model._camera_name2id:
            camera_id = model.camera_name2id(camera_name)

        _get_viewer(mode).render(width, height, camera_id=camera_id)

    if mode == 'rgb_array':
        # window size used for old mujoco-py:
        data = _get_viewer(mode).read_pixels(width, height, depth=False)
        # original image is upside-down, so flip it
        return data[::-1, :, :]
    elif mode == 'depth_array':
        _get_viewer(mode).render(width, height)
        # window size used for old mujoco-py:
        # Extract depth part of the read_pixels() tuple
        data = _get_viewer(mode).read_pixels(width, height, depth=True)[1]
        # original image is upside-down, so flip it
        return data[::-1, :]
    elif mode == 'human':
        _get_viewer(mode).render()

def close():
    if viewer is not None:
        # self.viewer.finish()
        viewer = None
        _viewers = {}

def _get_viewer(mode):
    viewer = _viewers.get(mode)
    if viewer is None:
        if mode == 'human':
            viewer = mujoco_py.MjViewer(sim)
        elif mode == 'rgb_array' or mode == 'depth_array':
            viewer = mujoco_py.MjRenderContextOffscreen(sim, -1)

        viewer_setup()
        _viewers[mode] = viewer
    return viewer

def viewer_setup():
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass


def get_body_com(body_name):
    return data.get_body_xpos(body_name).copy()


#
#Function for setting the initial state of the model
#

init_qpos= np.load('./qpos_1.npy')
init_qvel= np.load('./qvel_1.npy')

def reset_model():
    qpos= init_qpos
    qvel= init_qvel

    set_state(qpos, qvel)

def reset():
        sim.reset()
        reset_model()

def set_state(qpos, qvel):
    assert qpos.shape == (model.nq, ) and qvel.shape == (model.nv, )
    old_state= sim.get_state()

    new_state= mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                    old_state.act, old_state.udd_state)

    sim.set_state(new_state)
    sim.forward()

#
#Define a function for simulation
#set the target position 

target_pos_g = [0.1, 0.05]
def do_simulation(ctrl, n_frames): 
    for _ in range(n_frames):
        sim.data.ctrl[:]= ctrl
        sim.step()
        sim.forward()

        #Set the target position to a specified value
        upd_theta(target_pos_g[0], target_pos_g[1])

#Set up the position for the target in the environment
def upd_theta(x_coord, y_coord):
        
        target_x = x_coord
        target_y = y_coord

        x_joint_i= model.get_joint_qpos_addr("box:x")
        y_joint_i= model.get_joint_qpos_addr("box:y")

        crnt_state= sim.get_state()

        crnt_state.qpos[x_joint_i]= target_x 
        crnt_state.qpos[y_joint_i]= target_y 

        set_state(crnt_state.qpos, crnt_state.qvel)

upd_theta(target_pos_g[0], target_pos_g[1])

#Now let us set up our objective function
def obj_func(action):

    #Get the initial state of the simulator; Reinitialize the state after calculating the obj_function
    prev_state= sim.get_state()

    #Take a step in the environment using the action specified
    do_simulation(action, 1)

    #Get the hand position
    hand_xpos = sim.data.get_body_xpos("hand")

    #Get the target position 
    target_xpos = sim.data.get_body_xpos("target")

    #Get the model's qpos
    model_qpos = sim.get_state().qpos

    #Find the L2 error between the model qpos and the target qpos given the action
    model_qpos = model_qpos[:-2]
    target_qpos_l = target_qpos[:-2]

    l2_error = np.sum((model_qpos - target_qpos_l)**2)

    #Reinitialize the state after the loss function has been calculated
    set_state(prev_state.qpos, prev_state.qvel)


    return l2_error 

#Now let us define these functions in the context of the planning course
#Note that here x is the action not the state 

def f(x):
  return obj_func(x)

def Df(x):
  return gradient_fd(f, x)

def Hf(x):
  return hessian_fd(f, x)

# #
# # random simulation of the movement
# #

# # reset()

# # for tsteps in range(1000):
# #     action = np.random.rand(5)
# #     control = np.zeros(39)
# #     control[[2,3,4,5, 37]] = action
    
# #     do_simulation(control, 1)
# #     # upd_theta(0.1, 0.02)

# #     render() 


# Now let us test the objective function

# reset()

# for tsteps in range(100):
#     action = np.random.rand(5)
#     control = np.zeros(39)
#     control[[2,3,4,5, 37]] = action
    
#     print(f(control), Df(control), Hf(control))

#     render() 

obj_fun = []
trajectory = []

#Set the state to the initial state
reset()
upd_theta(target_pos_g[0], target_pos_g[1])

tol= 5e-02
r0= 0.5

global target_qpos
target_qpos = sim.get_state().qpos  

while(1):

    print("target_pos_updated")
    x0 = np.zeros(39)+0.02

    while(1):
        x0[x0 < 0.02] = 0.02
        print(x0)
        pks = -1*r0*Df(x0).T/np.linalg.norm(Df(x0).T)

        if Df(x0)@Hf(x0)@Df(x0).T <= 0:
            tk = 1
        else:
            thresh = np.linalg.norm(Df(x0))**3 / (  r0*Df(x0)@Hf(x0)@Df(x0).T )
            tk = min(1, thresh)

        pkc = tk*pks

        x1 = x0 + pkc

        print(np.linalg.norm(Df(x0)), f(x0))
        if np.linalg.norm(Df(x0)) < tol and np.linalg.norm(f(x0)) < tol:
            xl = x0
            break


        render()

        xl = x0
        x0 = x1
        r0 = r0

    xl[xl < 0.02] = 0.02
    # print(xl)
    #Advance the simulator by one step as well
    do_simulation(xl, 1)
    render()
        

    obj_fun.append(f(x0))
    trajectory.append(x0)