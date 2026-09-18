import mujoco
import mujoco.viewer
import time

path = '../assets/urdf/bipedal_pi.xml'
model = mujoco.MjModel.from_xml_path(path)
data = mujoco.MjData(model)
show_left_ui = False
show_right_ui = False

print('default gravity:', model.opt.gravity)

print('Total number of DoFs in the model:', model.nv)
print('Generalized positions:', data.qpos)
print('Generalized velocities:', data.qvel)

mujoco.mj_resetDataKeyframe(model, data, 0)  # Reset the state to keyframe 0

#runsim
step_counter = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
  while viewer.is_running():
    mujoco.mj_step(model, data)
    step_counter += 1

    if step_counter % 100 == 0:
            print(f'mjstep {step_counter}, Accel data: {data.sensor('base_link_accelerometer').data.copy()}, Gyro data: {data.sensor('base_link_gyroscope').data.copy()}')

    viewer.sync()
    time.sleep(model.opt.timestep)
