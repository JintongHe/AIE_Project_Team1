# AIE_Project_Team1

## Environment Setup
In the terminal
```angular2html
cd loco-mujoco
pip install -e .
cd ..
```

If wanna do imitation learning

```
loco-mujoco-download      # For downloading the dataset
git clone https://github.com/robfiras/ls-iq.git
cd ls-iq
pip install -e .
cd ..
```
test the environment using
```
python loco-mujoco/examples/simple_mushroom_env/example_unitree_a1.py
```
Starting code is at ```/pipeline```

When choosing the environment in loco-mujoco, make sure it's ```HumanoidTorque.walk.perfect```

## Citation
```
@inproceedings{alhafez2023b,
title={LocoMuJoCo: A Comprehensive Imitation Learning Benchmark for Locomotion},
author={Firas Al-Hafez and Guoping Zhao and Jan Peters and Davide Tateo},
booktitle={6th Robot Learning Workshop, NeurIPS},
year={2023}
}
```

---
## Credits 
Both Unitree models were taken from the [MuJoCo menagerie](https://github.com/google-deepmind/mujoco_menagerie)
