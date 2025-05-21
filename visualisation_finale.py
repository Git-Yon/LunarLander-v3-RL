import gymnasium as gym
from stable_baselines3 import PPO
import time

# Charger l’environnement avec rendu visuel
env = gym.make("LunarLander-v3", render_mode="human")

# Charger le modèle entraîné
model = PPO.load("models/big_ppo_lunarlander_final", env=env)

# Nombre d’épisodes à exécuter pour l’observation
episodes = 5

for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Attendre un petit peu pour voir l'animation (réglable)
        time.sleep(0.02)

    print(f"🎯 Épisode {ep+1}: Reward = {total_reward:.2f}")

env.close()