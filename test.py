import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm

panda = rtb.models.Panda()

target = sg.Sphere(radius=0.02, pose=sm.SE3(0.6, -0.2, 0.0))

Tep = panda.fkine(panda.q)
Tep.A[:3, 3] = target.T[:3, -1]

print("hi")