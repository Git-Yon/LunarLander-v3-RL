import gymnasium as gym
import time
from stable_baselines3 import DQN  # <-- changement PPO -> DQN

# Charger l’environnement avec rendu visuel
env = gym.make("LunarLander-v3", render_mode="human")

# Charger le modèle entraîné DQN (adapter le chemin)
model = DQN.load("models/dqn_LunarLander/dqn_lunarlander_final", env=env)

# Nombre d’épisodes à exécuter
episodes = 5

for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        time.sleep(0.02)  # pour ralentir un peu le rendu si besoin

    print(f"Episode {ep+1}: Reward = {total_reward:.2f}")

env.close()
