from gymnasium.envs.registration import register

register(
    id="hsup/HeatSupply-v0",
    entry_point="hsup.envs:HeatSupplyEnvV0",
)

register(
    id="hsup/HeatSupply-v1",
    entry_point="hsup.envs:HeatSupplyEnvV1",
)
