if __name__ == "__main__":
    # Env Params
    GRAYSCALE = 0  # 0 for RGB, 1 for Grayscale
    SHOW_INFO_PANEL = 1  # 0 for no info panel, 1 for info panel
    DISCRITIZED_ACTIONS = None
    NUM_TRACKS = 1  # 1 for simple track, 2 for complex track
    NUM_LANES = 1  # 1 for no lane changes, 2 for lane changes
    NUM_LANES_CHANGES = 4  # DO NOT CHANGE
    MAX_TIME_OUT = 2  # number of seconds out of track before episode ends
    FRAMES_PER_STATE = 4  # number of frames to stack for each state
    NVIDIA = True  # True for NVIDIA model, False for normal model with 96,96,3 input
    BUFFER_SIZE = 60_000
    TOTAL_TIMESTEPS = 500_000

    EVAL_FREQ = 100_000  # number of timesteps between each evaluation
    RENDER_EVAL = False  # True if you want to render the evaluation
    OUTPUT_PATH = "./training_folder"  # path to save the training folder
    MODEL_SAVE_FREQ = 100_000  # number of timesteps between each model save

    today = date.today()
    date_str = today.strftime("%b-%d-%Y")
    train_folder_output_path = f"{OUTPUT_PATH}/{date_str}_{time()}"
    os.makedirs(train_folder_output_path)

    env = CarRacing(
        allow_reverse=False,
        grayscale=GRAYSCALE,
        show_info_panel=SHOW_INFO_PANEL,
        discretize_actions=DISCRITIZED_ACTIONS,
        num_tracks=NUM_TRACKS,
        num_lanes=NUM_LANES,
        num_lanes_changes=NUM_LANES_CHANGES,
        max_time_out=MAX_TIME_OUT,
        frames_per_state=FRAMES_PER_STATE,
        nvidia=NVIDIA,
    )

    policy_kwargs = dict(
        features_extractor_class=CNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=1152),
    )

    model = PPO(CustomActorCriticPolicy, env, verbose=1, policy_kwargs=policy_kwargs)

    save_best_model_callback = SaveBestModelCallback(
        eval_env=env,
        n_eval_episodes=1,
        logpath=f"{train_folder_output_path}/logs",
        savepath=f"{train_folder_output_path}/best_model",
        eval_frequency=EVAL_FREQ,
        verbose=1,
        render=RENDER_EVAL,
    )
    save_model_callback = PeriodicSaveModelCallback(
        save_frequency=MODEL_SAVE_FREQ,
        save_path=f"{train_folder_output_path}/models",
    )
    callbacks = CallbackList([save_best_model_callback, save_model_callback])
    comment = input(
        "If you like to add a comment for the training add it here please: \n"
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)

    model.save("actor_critic_car_racing")
