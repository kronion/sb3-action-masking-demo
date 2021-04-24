# SB3-Contrib Invalid Action Masking Demo

## Installation

`pip install -r requirements.txt`

## Training

`python train.py [--mask / --no-mask] OUTPUT_FOLDER`

## Testing

`python test.py [--mask-1 / --no-mask-1] [--mask-2 / --no-mask-2] MODEL_1_PATH MODEL_2_PATH`

## Experiments

### Masked (training and testing) vs unmasked

Train an agent with masking: `python train.py --mask zoo/mask`

Train an agent without masking: `python train.py --no-mask zoo/no_mask`

Play the two agents against each other, with masking still enabled for the agent that was trained with them:
```
python test.py zoo/mask/latest/final_model.zip zoo/no_mask/latest/final_model.zip --mask-1
```

### Masked (training only) vs unmasked

Train an agent with masking: `python train.py --mask zoo/mask`

Train an agent without masking: `python train.py --no-mask zoo/no_mask`

Play the two agents against each other, with masking disabled for the agent that was trained with them:
```
python test.py zoo/mask/latest/final_model.zip zoo/no_mask/latest/final_model.zip
```

### Masked (training and testing) vs masked (testing only)

Train an agent with masking: `python train.py --mask zoo/mask`

Train an agent without masking: `python train.py --no-mask zoo/no_mask`

Play the two agents against each other, with masking enabled for both agents:
```
python test.py zoo/mask/latest/final_model.zip zoo/no_mask/latest/final_model.zip --mask-1 --mask-2
```

## Visualizations

We provide TensorBoard run data for two models trained for 2M+ timesteps, one with masking and one without.
To view the data:
```
tensorboard --logdir_spec long:zoo/long/latest/PPO_1,prev:zoo/long_mask/latest/PPO_1
```
