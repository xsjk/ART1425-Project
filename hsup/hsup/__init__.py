from gymnasium.envs.registration import register

register(
    id="hsup/HeatSupply-v0",
    entry_point="hsup.envs:HeatSupplyEnv",
)
