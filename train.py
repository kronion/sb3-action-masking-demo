import datetime
from pathlib import Path

import click
from gym_quarto import OnePlayerWrapper, QuartoEnvV0, RandomPlayer
from stable_baselines3.common import logger

from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.policies import MaskedActorCriticPolicy
from sb3_contrib.ppo_mask import MaskedPPO

SEED = 721

EVAL_FREQ = 10000
EVAL_EPISODES = 200


@click.command()
@click.argument("output_folder", type=click.Path())
@click.option("--load", "-l", "load_path")
@click.option("--timesteps", default=100000)
@click.option("--mask/--no-mask", "use_masking", default=False)
def train(output_folder, load_path, timesteps, use_masking):
    base_output = Path(output_folder)
    full_output = base_output / datetime.datetime.now().isoformat(timespec="seconds")

    logger.configure(folder=str(full_output))

    env = QuartoEnvV0()
    env = OnePlayerWrapper(env, RandomPlayer(env))
    env.seed(SEED)

    # take mujoco hyperparams (but doubled timesteps_per_actorbatch to cover more steps.)
    # model = PPO(MlpPolicy, env, timesteps_per_actorbatch=4096, clip_param=0.2, entcoeff=0.0, optim_epochs=10,
    #             optim_stepsize=3e-4, optim_batchsize=64, gamma=0.99, lam=0.95, schedule='linear', verbose=2)
    if load_path:
        model = MaskedPPO.load(load_path, env, use_masking=use_masking)
    else:
        model = MaskedPPO(
            MaskedActorCriticPolicy,
            env,
            verbose=1,
            use_masking=use_masking,
            policy_kwargs={"net_arch": [dict(pi=[16, 16], vf=[16, 16])]},
            tensorboard_log=str(full_output),
        )

    eval_callback = MaskableEvalCallback(
        env,
        best_model_save_path=str(full_output),
        log_path=str(full_output),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
    )

    try:
        model.learn(total_timesteps=timesteps, callback=eval_callback)
    except Exception:
        import pdb

        pdb.post_mortem()

    model.save(str(full_output / "final_model"))

    env.close()

    latest = base_output / "latest"
    latest.unlink(missing_ok=True)
    relative_path = full_output.relative_to(latest.parent)
    latest.symlink_to(relative_path)


if __name__ == "__main__":
    train()
