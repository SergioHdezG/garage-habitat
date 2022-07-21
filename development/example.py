from environments import habitat_envs
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
import os

# env = habitat_envs.SimpleRLEnvRGB(config_paths="configs/tasks/pointnav.yaml",
#                                result_path=os.path.join("development", "images"))
env = habitat_envs.HM3DRLEnv()

goal_radius = env.episodes[0].goals[0].radius
if goal_radius is None:
    goal_radius = env.config.SIMULATOR.FORWARD_STEP_SIZE
follower = ShortestPathFollower(
    env.habitat_env.sim, goal_radius, False
)

for episode in range(4):
    observations = env.reset()
    img = env.render(print_on_screen=True)

    images = []
    while not env.habitat_env.episode_over:
        best_action = follower.get_next_action(
            env.habitat_env.current_episode.goals[0].position
        )
        if best_action is None:
            break
        observations, reward, done, info = env.step(best_action)
        img = env.render(print_on_screen=True)

    print("Episode finished")