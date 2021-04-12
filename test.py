import random

import click
from gym_quarto import QuartoEnvV0, RandomPlayer
from sb3_contrib.ppo_mask import MaskedPPO


SEED = 721
STEP_LIMIT = 20


@click.command()
@click.argument(
    "model_1_path", type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.argument(
    "model_2_path", type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.option("--num-games", default=100)
@click.option("--mask-1/--no-mask-1", default=False)
@click.option("--mask-2/--no-mask-2", default=False)
def test(model_1_path: str, model_2_path: str, num_games: int, mask_1, mask_2):
    agent_1 = {
        "name": "model_1",
        "model": MaskedPPO.load(model_1_path),
        "mask": mask_1,
    }
    agent_2 = {
        "name": "model_2",
        "model": MaskedPPO.load(model_2_path),
        "mask": mask_2,
    }
    agents = [agent_1, agent_2]

    wins = {agent["name"]: 0 for agent in agents}
    wins["tie"] = 0
    wins["timeout"] = 0

    env = QuartoEnvV0()

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
                        other_agent = None
                        for agent2 in agents:
                            if agent != agent2:
                                other_agent = agent2
                        info["winner"] = other_agent["name"]
                    elif info['draw']:
                        info['winner'] = 'tie'
                    else:
                        info['winner'] = agent["name"]
                    break

            steps += 1

        winner = info["winner"]
        if steps >= STEP_LIMIT:
            winner = "timeout"
        wins[winner] += 1

    for k, v in wins.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    test()
