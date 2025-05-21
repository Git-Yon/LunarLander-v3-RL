import gymnasium as gym
import time
from stable_baselines3 import PPO

# Charger l’environnement avec rendu visuel
env = gym.make("LunarLander-v3", render_mode="human")

# Charger le modèle entraîné (adapté selon ton chemin)
model = PPO.load("models/PPO_LunarLander/ppo_lunarlander_final", env=env)

# Nombre d’épisodes à exécuter
episodes = 5

for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
          # ajuster la vitesse d'affichage

    print(f"Episode {ep+1}: Reward = {total_reward:.2f}")

env.close()
