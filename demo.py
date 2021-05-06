import datetime
import random
from pathlib import Path

import click
import gym
from gym_quarto import RandomPlayer
from gym_quarto.wrapper import OnePlayerWrapper
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common import logger


SEED = 721
STEP_LIMIT = 20


EVAL_FREQ = 10000
EVAL_EPISODES = 200


@click.group()
def cli():
    pass


@cli.command()
@click.argument("output_folder", type=click.Path())
@click.option("--load", "-l", "load_path")
@click.option("--timesteps", default=100000)
@click.option("--mask/--no-mask", "use_masking", default=False)
def train(output_folder, load_path, timesteps, use_masking):
    base_output = Path(output_folder)
    full_output = base_output / datetime.datetime.now().isoformat(timespec="seconds")

    # logger.configure(folder=str(full_output))

    env = gym.make("quarto-multidiscrete-v1")
    env = OnePlayerWrapper(env, RandomPlayer(env))
    env.seed(SEED)

    if load_path:
        # Note that you may override use_masking value loaded from previous model
        model = MaskablePPO.load(load_path, env, use_masking=use_masking)
    else:
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            verbose=1,
            use_masking=use_masking,
            tensorboard_log=str(full_output),
        )

    eval_callback = MaskableEvalCallback(
        env,
        best_model_save_path=str(full_output),
        log_path=str(full_output),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
    )

    model.learn(total_timesteps=timesteps, callback=eval_callback)
    model.save(str(full_output / "final_model"))
    env.close()

    latest = base_output / "latest"
    latest.unlink(missing_ok=True)
    relative_path = full_output.relative_to(latest.parent)
    latest.symlink_to(relative_path)


@cli.command()
@click.argument(
    "model_1_path", type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.argument(
    "model_2_path", type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.option("--mask-1/--no-mask-1", default=False)
@click.option("--mask-2/--no-mask-2", default=False)
@click.option("--num-games", default=100)
def test(
    model_1_path: str, model_2_path: str, mask_1: bool, mask_2: bool, num_games: int
):
    env = gym.make("quarto-multidiscrete-v1")

    agent_1 = {
        "name": "model_1",
        "model": MaskablePPO.load(model_1_path),
        "mask": mask_1,
    }
    agent_2 = {
        "name": "model_2",
        "model": MaskablePPO.load(model_2_path),
        "mask": mask_2,
    }
    agents = [agent_1, agent_2]

    wins = {agent["name"]: 0 for agent in agents}
    wins["tie"] = 0
    wins["timeout"] = 0

    # Randomize agent order
    random.shuffle(agents)

    for i in range(num_games):
        obs = env.reset()
        done = False
        steps = 0

        while not done and steps < STEP_LIMIT:
            for agent in agents:
                if agent["mask"]:
                    masks = env.action_masks()
                    action, _ = agent["model"].predict(obs, action_masks=masks)
                else:
                    action, _ = agent["model"].predict(obs)
                obs, _, done, info = env.step(action)

                if done:
                    if info["invalid"]:
                        for agent2 in agents:
                            if agent2 != agent:
                                info["winner"] = agent2["name"]
                                break
                    elif info["draw"]:
                        info["winner"] = "tie"
                    else:
                        info["winner"] = agent["name"]
                    break

            steps += 1

        winner = info["winner"]
        if steps >= STEP_LIMIT:
            winner = "timeout"
        wins[winner] += 1

    for k, v in wins.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    cli()
