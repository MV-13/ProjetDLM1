# app.py (fichier principal)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import skimage.data
import importlib

import torch.nn as nn
import scipy.ndimage
import skimage
import numpy as np
import matplotlib.pyplot as plt
import utils
import models
import differential_operators as diff
import loss_functions as loss_fn


# Configuration de la page
st.set_page_config(
    page_title="Application SIREN Image Fitting",
    page_icon="🖼️",
    layout="wide"
)

# Vérification de la disponibilité du GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.sidebar.info(f"Calculs exécutés sur: {device}")

# Navigation
pages = {
    "Présentation": "presentation_page",
    "Image Fitting": "image_fitting_page"
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Aller à", list(pages.keys()))

# Fonction pour la page de présentation
def presentation_page():
    st.title("Bienvenue sur l'Application SIREN pour Image Fitting")
    
    st.markdown("""
    ## À propos de cette application
    
    Cette application démontre l'utilisation des réseaux SIREN (Sinusoidal Representation Networks) 
    pour l'ajustement d'images. Le réseau apprend à représenter une image en associant des coordonnées 
    spatiales (x,y) à des valeurs de pixels.
    
    Utilisez la barre latérale pour accéder à la page d'expérimentation "Image Fitting".
    """)
    
    st.image("https://via.placeholder.com/800x400.png?text=SIREN+Neural+Networks", 
             caption="Illustration de réseaux SIREN", use_column_width=True)
    
    st.markdown("""
    ## Les réseaux SIREN
    
    Les SIREN (Sinusoidal Representation Networks) sont des réseaux de neurones qui utilisent 
    des fonctions d'activation sinusoïdales, ce qui leur confère des propriétés particulièrement 
    intéressantes pour représenter des signaux complexes.
    
    ### Avantages des SIREN pour le traitement d'images:
    - Représentation continue des images
    - Capacité à capturer efficacement les hautes fréquences et les détails fins
    - Super-résolution implicite (interpolation entre les pixels)
    - Représentation compacte (parfois plus petite que l'image originale)
    
    ### Technologies utilisées dans cette application:
    * Python
    * PyTorch
    * Streamlit
    * NumPy
    * Matplotlib
    * skimage
    """)
    
    st.markdown("""
    ## Comment utiliser cette application
    
    1. Naviguez vers la page "Image Fitting" en utilisant le menu dans la barre latérale
    2. Choisissez une image source parmi les options disponibles
    3. Ajustez les paramètres du modèle et de l'entraînement selon vos préférences
    4. Lancez l'entraînement et observez comment le réseau SIREN apprend à représenter l'image
    5. Examinez les résultats et l'évolution de la perte pendant l'entraînement
    """)

# Fonction pour la page Image Fitting
def image_fitting_page():
    st.title("Ajustement d'Image (Image Fitting)")
    
    st.markdown("""
    Cette section utilise un réseau SIREN pour apprendre à représenter une image.
    Le réseau apprend à mapper des coordonnées spatiales (x,y) vers des valeurs de pixels.
    """)
    
    # Options de configuration
    st.sidebar.header("Paramètres")
    
    # Choix de l'image
    image_option = st.sidebar.selectbox(
        "Image source:",
        ("camera", "cat", "astronaut", "immunohistochemistry", "brick", "coffee", "rocket"),
        index=1
    )
    
    image_size = st.sidebar.slider("Taille de l'image:", min_value=64, max_value=512, value=256, step=64)
    
    # Paramètres du modèle
    st.sidebar.subheader("Paramètres du modèle")
    hidden_features = st.sidebar.slider("Nombre de neurones cachés:", min_value=64, max_value=512, value=256, step=64)
    hidden_layers = st.sidebar.slider("Nombre de couches cachées:", min_value=1, max_value=5, value=3)
    
    # Paramètres d'entraînement
    st.sidebar.subheader("Entraînement")
    learning_rate = st.sidebar.select_slider(
        "Taux d'apprentissage:",
        options=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
        format_func=lambda x: f"{x:.5f}",
        value=1e-4
    )
    num_epochs = st.sidebar.slider("Nombre d'époques:", min_value=100, max_value=1000, value=500, step=100)
    
    # Lancement de l'entraînement
    if st.sidebar.button("Lancer l'entraînement"):
        progress_bar = st.progress(0)
        epoch_status = st.empty()
        loss_status = st.empty()
        
        # Zone pour afficher l'image
        result_container = st.empty()
        
        # Code d'expérimentation modifié pour Streamlit
        with st.spinner("Chargement de l'image..."):
            # Déterminer quelle image charger
            if image_option == "camera":
                img_data = skimage.data.camera()
            elif image_option == "cat":
                img_data = skimage.data.cat()
            elif image_option == "astronaut":
                img_data = skimage.data.astronaut()
            elif image_option == "immunohistochemistry":
                img_data = skimage.data.immunohistochemistry()
            elif image_option == "brick":
                img_data = skimage.data.brick()
            elif image_option == "coffee":
                img_data = skimage.data.coffee()
            elif image_option == "rocket":
                img_data = skimage.data.rocket()
            
            # Préparation des données
            cameraman = utils.ImageFitting(image_size, img_data)
            dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True, num_workers=0)
            X, y = next(iter(dataloader))
            X, y = X.to(device), y.to(device)
            
            # Instanciation du modèle et de l'optimiseur
            siren = models.Siren(in_features=2, out_features=1, hidden_features=hidden_features,
                                hidden_layers=hidden_layers, outermost_linear=True).to(device)
            optimizer = optim.Adam(siren.parameters(), lr=learning_rate)
            
        # Boucle d'entraînement adaptée pour Streamlit
        losses = []
        col1, col2 = st.columns([2, 1])
        
        with col1:
            result_container = st.empty()
        
        with col2:
            metrics_container = st.empty()
            
        for epoch in range(num_epochs):
            output, coords = siren(X)
            
            loss = loss_fn.MSE(output, y)
            losses.append(loss.item())
            
            # Mise à jour de l'interface Streamlit
            progress = (epoch + 1) / num_epochs
            progress_bar.progress(progress)
            epoch_status.text(f"Époque: {epoch+1}/{num_epochs}")
            loss_status.text(f"Perte: {loss.item():.6f}")
            
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                # Affichage de l'image
                fig, ax = plt.subplots(figsize=(8, 8))
                
                # Récupérer les prédictions et les reformer en image
                img_pred = output.detach().reshape(image_size, image_size).cpu().numpy()
                
                ax.imshow(img_pred, cmap='viridis')
                ax.set_title(f"Époque {epoch+1}")
                ax.axis('off')
                
                # Mettre à jour l'image
                result_container.pyplot(fig)
                
                # Mettre à jour les métriques
                metrics_fig, metrics_ax = plt.subplots(figsize=(4, 3))
                metrics_ax.plot(losses)
                metrics_ax.set_xlabel("Époque")
                metrics_ax.set_ylabel("Perte MSE")
                metrics_ax.set_title("Évolution de la perte")
                metrics_ax.grid(True)
                metrics_container.pyplot(metrics_fig)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        st.success("Entraînement terminé!")
        
        # Affichage du graphique de perte final
        st.subheader("Évolution de la perte")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(losses)
        ax.set_xlabel("Époque")
        ax.set_ylabel("Perte MSE")
        ax.set_title("Évolution de la perte pendant l'entraînement")
        ax.grid(True)
        st.pyplot(fig)
        
        # Comparaison image originale vs prédiction
        st.subheader("Comparaison: Original vs Prédiction")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Image originale")
            original_img = y.detach().reshape(image_size, image_size).cpu().numpy()
            st.image(original_img, caption="Image originale", use_column_width=True)
            
        with col2:
            st.subheader("Image prédite par SIREN")
            predicted_img = output.detach().reshape(image_size, image_size).cpu().numpy()
            st.image(predicted_img, caption="Prédiction du réseau SIREN", use_column_width=True)
    
    else:
        st.info("Configurez les paramètres dans la barre latérale et cliquez sur 'Lancer l'entraînement' pour commencer.")
        
        # Afficher un code d'exemple
        st.subheader("Code pour l'expérience 'Image Fitting'")
        st.code("""
# Get coordinates from the image (X) and the pixel values (y).
# Images possible : camera, cat, astronaut, immunohistochemistry, brick, coffee, rocket
cameraman = utils.ImageFitting(256, skimage.data.cat())
dataloader = DataLoader(cameraman, batch_size = 1, pin_memory = True, num_workers = 0)
X, y = next(iter(dataloader))
X, y = X.to(device), y.to(device)

# Instantiate model, optimizer and number of epochs.
siren = models.Siren(in_features = 2, out_features = 1, hidden_features = 256,
                     hidden_layers = 3, outermost_linear = True).to(device)
optimizer = optim.Adam(siren.parameters(), lr = 1e-4)
num_epochs = 500

# Training loop.
for epoch in range(num_epochs):
    output, coords = siren(X)

    loss = loss_fn.MSE(output, y)
    utils.display_img(epoch, 10, output, coords, loss, 256, device)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        """)
        
        # Afficher une explication du code
        st.subheader("Explication du code")
        st.markdown("""
        Le code ci-dessus:
        
        1. **Charge une image** et prépare les données d'entrée (coordonnées X) et les valeurs cibles y (intensités des pixels)
        2. **Initialise un réseau SIREN** avec:
           - 2 entrées (coordonnées x,y)
           - 1 sortie (intensité du pixel)
           - 3 couches cachées avec 256 neurones chacune
        3. **Configure l'optimiseur Adam** avec un taux d'apprentissage de 0.0001
        4. **Exécute une boucle d'entraînement** sur 500 époques où:
           - Le modèle génère des prédictions pour chaque coordonnée
           - L'erreur quadratique moyenne (MSE) est calculée
           - L'image prédite est affichée périodiquement
           - Les poids du réseau sont mis à jour par rétropropagation
        
        Dans l'interface Streamlit, vous pouvez personnaliser ces paramètres et observer en temps réel
        comment le réseau apprend à représenter l'image sélectionnée.
        """)

# Exécution de la page sélectionnée
if selection == "Présentation":
    presentation_page()
else:
    image_fitting_page()