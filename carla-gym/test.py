# ==============================================================================
# -- Imports -------------------------------------------------------------------
# ==============================================================================

import gym
import carla_gym


# ==============================================================================
# -- Main ----------------------------------------------------------------------
# ==============================================================================

def main():
    
    # parameters for the carla_gym environment
    params = {
        'max_time_episode': 1000,  # maximum timesteps per episode
        'obs_range': 32,  # observation range (meter)
        'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
        'min_speed': 20, # desired minimum eg vehicle speed (Km/Hr)
        'max_speed': 30, # desired maximum eg vehicle speed (Km/Hr)
        'discrete': False,  # whether to use discrete control space
        'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
        'continuous_steer_range': [-0.5, 0.5],  # continuous steering angle range
    }
  
    try:
        # Set carla-gym environment
        env = gym.make('Carla-v0', params=params)
        obs = env.reset()

        done = False
        while True:
            action = env.action_space.sample() # random action selection
            obs, reward, done, _, _ = env.step(action)

            if done:
                obs = env.reset()
            
        # episodes = 3
        # for episode in range (1, episodes+1):

            
        #     # Reset environment for new epoch
        #     env.reset()
        #     done = False
        #     score = 0
            
        #     while not done:
        #         action = env.action_space.sample()
        #         obs, reward, done, info = env.step(action)
        #         score += reward

        #     print(f"Episode:{episode}  Score:{score}")


    except KeyboardInterrupt:
        env.destroy_actors()
    
    finally:
        env.destroy_actors()


if __name__ == '__main__':

    main()