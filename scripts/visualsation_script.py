import mujoco
import mujoco.viewer
import time

path = '../assets/urdf/Bipedal_visualisation.xml'
model = mujoco.MjModel.from_xml_path(path)
data = mujoco.MjData(model)

print('default gravity', model.opt.gravity)
print('flipped gravity', model.opt.gravity)

print('Total number of DoFs in the model:', model.nv)
print('Generalized positions:', data.qpos)
print('Generalized velocities:', data.qvel)

# CHANGED: Reset the state before launching the interactive viewer.
mujoco.mj_resetDataKeyframe(model, data, 0)  # Reset the state to keyframe 0

with mujoco.viewer.launch_passive(model, data) as viewer:
  while viewer.is_running():
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(model.opt.timestep)
