import gymnasium as gym

gym.register(id="PolyPanda/Reach-v0", entry_point="polypanda.environments.reach:ReachEnv")
gym.register(id="PolyPanda/Grasp-v0", entry_point="polypanda.environments.grasp:GraspEnv")
