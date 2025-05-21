import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from tqdm import tqdm
import os



# Création de l'environnement LunarLander-v3
env = gym.make("LunarLander-v3")

# Dossier pour sauvegarder les modèles et logs tensorboard
log_dir = "big_ppo_tensorboard/"
os.makedirs(log_dir, exist_ok=True)

# Callback pour sauvegarder le modèle toutes les 50 000 étapes
checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=log_dir, name_prefix="big_ppo_lander")


# Création du modèle PPO optimisé pour CPU
model = PPO(
    "MlpPolicy",
    env,
    verbose=0,  # On désactive le verbose pour éviter les doublons avec la barre
    tensorboard_log=log_dir,
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
    ent_coef=0.01,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2
)

# Entraînement avec barre de progression et sauvegarde périodique
model.learn(total_timesteps=850000, callback=[checkpoint_callback], tb_log_name="big_PPO_LunarLander")

# Sauvegarde finale du modèle
model.save("models/big_ppo_lunarlander_final")

print("Entraînement terminé. Modèle sauvegardé dans:", log_dir)
