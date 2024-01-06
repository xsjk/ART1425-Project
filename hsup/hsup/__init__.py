from gymnasium.envs.registration import register

register(
    id="hsup/HeatSupply",
    entry_point="hsup.envs:HeatSupplyEnv",
)
