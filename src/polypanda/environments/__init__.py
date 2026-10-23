import gymnasium as gym

gym.register(id="PolyPanda/Reach-v0", entry_point="polypanda.environments.reach:ReachEnv")
gym.register(id="PolyPanda/Grasp-v0", entry_point="polypanda.environments.grasp:GraspEnv")
gym.register(id="PolyPanda/Transport-v0",
             entry_point="polypanda.environments.transport:TransportEnv")
gym.register(id="PolyPanda/TransportResidual-v0",
             entry_point="polypanda.environments.transport:TransportResidualEnv")
