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
import streamlit_utils as st_utils


# Configuration de la page
st.set_page_config(
    page_title="Application SIREN",
    page_icon="🖼️",
    layout="wide")

# Vérification de la disponibilité du GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.sidebar.info(f"Calculs exécutés sur {device}")

# Navigation
pages = {
    "Présentation": "presentation_page",
    "Image Fitting": "image_fitting_page",
    "Equation de Poisson": "poisson_page",
    "Inpainting": "inpainting_page"}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Aller à", list(pages.keys()))

# Fonction pour la page de présentation
def presentation_page():
    st.title("Bienvenue sur l'Application SIREN")
    
    st.markdown("""
        ## À propos de cette application
        
        Cette application démontre l'utilisation des réseaux SIREN
        (Sinusoidal Representation Networks) pour diverse tâches de traitement d'image.
        Le réseau apprend à représenter une image en associant des des valeurs de pixels
        à des coordonnées spatiales (x,y).
        
        Utilisez la barre latérale pour accéder aux 3 expérimentations:
        - **Image Fitting**: apprendre à représenter une image ;
        - **Équation de Poisson**: représenter une image à partir de son gradient ;
        - **Inpainting implicite**: compléter une image masquée.
                
        Chaque expérimentation compare en direct les performances d'un réseau SIREN
        et d'un réseau standard.
                
        Les réseaux reçoivent comme input une grille de coordonnées de résolution choisie
        par l'utilisateur normalisée dans [-1, 1] et retournent les valeurs de pixels
        pour chaque coordonnée.
    """)
    
    st.markdown("""
        ## Les réseaux SIREN
        
        Les SIREN (Sinusoidal Representation Networks) sont des réseaux de neurones
        qui utilisent des fonctions d'activation sinusoïdales, ce qui leur confère
        des propriétés particulièrement intéressantes pour représenter des signaux complexes.
        
        ### Avantages des SIREN pour le traitement d'images:
        - Représentation continue des images ;
        - Capacité à capturer efficacement les hautes fréquences et les détails fins ;
        - Super-résolution implicite (interpolation entre les pixels) ;
        - Représentation compacte (parfois plus petite que l'image originale).
        
        ### Technologies utilisées dans cette application:
        * Python
        * PyTorch
        * Streamlit
        * NumPy
        * Matplotlib
        * skimage
    """)
    

# Fonction pour la page Image Fitting
def image_fitting_page():
    st.title("Ajustement d'Image (Image Fitting)")
    
    st.markdown("""
        Dans cette section, les modèles sont supervisés par les vraies valeurs de pixels
        de l'image source. 
        """)
    
    # Options de configuration
    st.sidebar.header("Paramètres")
    
    # Choix de l'image
    image_option = st.sidebar.selectbox(
        "Image source:",
        ("camera", "cat", "astronaut", "immunohistochemistry", "brick", "coffee", "rocket"),
        index=1)
    
    image_size = st.sidebar.slider(
        "Résolution de l'image:",
        min_value=64, max_value=512,
        value=256, step=32)
    
    # Paramètres du modèle SIREN
    st.sidebar.subheader("Paramètres du modèle SIREN")

    hidden_features_siren = st.sidebar.slider(
        "Nombre de neurones cachés du modèle SIREN:",
        min_value=64, max_value=512,
        value=256, step=32)
    
    hidden_layers_siren = st.sidebar.slider(
        "Nombre de couches cachées du modèle SIREN:",
        min_value=1, max_value=5,
        value=3, step=1)
    
    # Paramètres du modèle standard
    st.sidebar.subheader("Paramètres du modèle standard")

    activation_option = st.sidebar.selectbox(
        "Fonction d'activation:",
        ("ReLU", "Sigmoid", "Tanh", "LeakyReLU"),
        index = 1)
    
    hidden_features_std = st.sidebar.slider(
        "Nombre de neurones cachés du modèle standard:",
        min_value=64, max_value=512,
        value=256, step=32)

    hidden_layers_std = st.sidebar.slider(
        "Nombre de couches cachées du modèle standard:",
        min_value=1, max_value=5,
        value=3, step=1)

    # Paramètres d'entraînement
    st.sidebar.subheader("Entraînement")

    learning_rate = st.sidebar.select_slider(
        "Taux d'apprentissage:",
        options=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
        format_func=lambda x: f"{x:.5f}",
        value=1e-4)
    
    num_epochs = st.sidebar.slider(
        "Nombre d'époques:",
        min_value=5, max_value=500,
        value=10, step=5)
    
    # Lancement de l'entraînement
    if st.sidebar.button("Lancer l'entraînement"):
        progress_bar = st.progress(0)
        epoch_status = st.empty()
        
        # Zone pour afficher l'image
        result_container = st.empty()
        
        # Code d'expérimentation modifié pour Streamlit
        with st.spinner("Chargement de l'image..."):
            # Déterminer l'image à charger et la fonction d'activation choisie
            img_choice = st_utils.choose_image(image_option)
            chosen_fct = st_utils.choose_function(activation_option)
            
            # Préparation des données
            img = utils.ImageFitting(image_size, img_choice)
            dataloader = DataLoader(img, batch_size=1, pin_memory=True, num_workers=0)
            X, y = next(iter(dataloader))
            X, y = X.to(device), y.to(device)
            
            # Instanciation des modèles et de l'optimiseur.
            siren = models.Siren(
                in_features=2, out_features=1,
                hidden_features=hidden_features_siren,
                hidden_layers=hidden_layers_siren,
                outermost_linear=True).to(device)
            
            standard = models.standard_network(
                activation = chosen_fct,
                in_features=2, out_features=1,
                hidden_features=hidden_features_std,
                hidden_layers=hidden_layers_std,
                outermost_linear=True).to(device)
            
            optim_siren = optim.Adam(siren.parameters(), lr=learning_rate)
            optim_std = optim.Adam(standard.parameters(), lr = learning_rate)
        
        # Boucle d'entraînement adaptée pour Streamlit
        losses_siren = []
        losses_std = []
        col1, col2 = st.columns([2, 1])
        
        with col1:
            siren_container = st.empty()
        
        with col2:
            metrics_container = st.empty()
        
        for epoch in range(num_epochs):
            output_siren, coords_siren = siren(X)
            output_std, coords_std = standard(X)
            
            loss_siren = loss_fn.MSE(output_siren, y)
            loss_std = loss_fn.MSE(output_std, y)
            
            losses_siren.append(loss_siren.item())
            losses_std.append(loss_std.item())
            
            # Mise à jour de l'interface Streamlit
            progress = (epoch + 1) / num_epochs
            progress_bar.progress(progress)
            epoch_status.text(f"Époque: {epoch+1}/{num_epochs}")

            if epoch % 5 == 0 or epoch == num_epochs - 1:
                # Affichage de l'image
                fig, ax = plt.subplots(1, 2, figsize=(8,8))
                
                # Récupérer les prédictions et les reformer en image
                pred_siren = output_siren.detach().reshape(image_size, image_size).cpu().numpy()
                pred_std = output_std.detach().reshape(image_size, image_size).cpu().numpy()
                
                ax[0].imshow(pred_siren, cmap="viridis")
                ax[0].set_title(f"SIREN")
                ax[1].imshow(pred_std, cmap="viridis")
                ax[1].set_title(activation_option)
                ax[0].axis("off")
                ax[1].axis("off")
                
                # Mettre à jour l'image
                siren_container.pyplot(fig)
                
                # Mettre à jour les métriques
                metrics_fig, metrics_ax = plt.subplots(figsize=(4,4))
                metrics_ax.plot(losses_siren, label="SIREN")
                metrics_ax.plot(losses_std, label=activation_option)
                metrics_ax.set_xlabel("Époque")
                metrics_ax.set_ylabel("Perte MSE")
                metrics_ax.set_title("Évolution de la perte")
                metrics_ax.grid(True)
                metrics_ax.legend()
                metrics_container.pyplot(metrics_fig)
            
            optim_siren.zero_grad()
            optim_std.zero_grad()

            loss_siren.backward()
            loss_std.backward()

            optim_siren.step()
            optim_std.step()
        
        st.success("Entraînement terminé!")
        
        # Affichage du graphique de perte final
        st.subheader("Évolution de la perte")
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(losses_siren, label="SIREN")
        ax.plot(losses_std, label=activation_option)
        ax.set_xlabel("Époque")
        ax.set_ylabel("Perte MSE")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
        # Comparaison image originale vs prédiction
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Originale")
            original_img = y.detach().reshape(image_size, image_size).cpu().numpy()
            original_img = np.clip(original_img, .0, 1.0)
            st.image(original_img, use_container_width=True)

        with col2:
            st.subheader("SIREN")
            siren_img = output_siren.detach().reshape(image_size, image_size).cpu().numpy()
            siren_img = np.clip(siren_img, .0, 1.0)
            st.image(siren_img, use_container_width=True)

        with col3:
            st.subheader(activation_option)
            std_image = output_std.detach().reshape(image_size, image_size).cpu().numpy()
            std_image = np.clip(std_image, .0, 1.0)
            st.image(std_image, use_container_width=True)

    else:
        st.info("Configurez les paramètres dans la barre latérale et cliquez sur 'Lancer l'entraînement' pour commencer.")
        
        # Afficher un code d'exemple
        st.subheader("Code pour l'expérience 'Image Fitting'")
        st.code("""
            # Choose an image from skimage.data in the following list :
            # camera, cat, astronaut, immunohistochemistry, brick, coffee, rocket.
            img = utils.ImageFitting(256, skimage.data.cat())
            dataloader = DataLoader(img, batch_size=1, pin_memory=True, num_workers=0)
            X, y = next(iter(dataloader))
            X, y = X.to(device), y.to(device)

            # Instantiate siren model, ReLU model, optimizer and number of epochs.
            siren = models.Siren(
                in_features=2, out_features=1,
                hidden_features=256, hidden_layers=3,
                outermost_linear=True).to(device)

            standard = models.standard_network(
                activation=nn.ReLU(),
                in_features=2, out_features=1,
                hidden_features=256, hidden_layers=3,
                outermost_linear=True).to(device)

            optim_siren = optim.Adam(siren.parameters(), lr=1e-4)
            optim_std = optim.Adam(standard.parameters(), lr=1e-4)
            num_epochs = 500

            # Training loop.
            for epoch in range(num_epochs):
                output_siren, coords_siren = siren(X)
                output_std, coords_std = standard(X)

                loss_siren = loss_fn.MSE(output_siren, y)
                loss_std = loss_fn.MSE(output_std, y)

                optim_siren.zero_grad()
                optim_std.zero_grad()
                loss_siren.backward()
                loss_std.backward()
                optim_siren.step()
                optim_std.step()
            """)
        
        # Afficher une explication du code
        st.subheader("Explication du code")
        st.markdown("""
            Le code ci-dessus:
            
            1. **Charge une image** et prépare les données d'entrée (coordonnées X)
                et les valeurs cibles y (intensités des pixels).
            
            2. **Initialise un réseau SIREN** et un réseau ReLU avec:
                - 2 entrées (coordonnées x, y) ;
                - 1 sortie (intensité du pixel) ;
                - 3 couches cachées avec 256 neurones chacune.
            
            3. **Configure l'optimiseur Adam** avec un taux d'apprentissage de 0.0001.
            
            4. **Exécute une boucle d'entraînement** sur 500 époques où:
                - Le modèle génère des prédictions pour chaque coordonnée ;
                - L'erreur quadratique moyenne (MSE) est calculée par rapport
                  à la véritable image ;
                - L'image prédite est affichée périodiquement ;
                - Les poids du réseau sont mis à jour par rétropropagation.
            
            Dans l'interface Streamlit, vous pouvez personnaliser ces paramètres
            et observer en temps réel comment le réseau apprend à représenter
            l'image sélectionnée.
        """)

# Fonction pour la page Equation de Poisson
def poisson_page():
    st.title("Résoudre l'équation de Poisson")
    
    st.markdown(r"""
        Dans cette section, les réseaux ne sont supervisés que par la dérivée spatiale 
        de l'image (son gradient), ce qui équivaut à résoudre l'équation de Poisson:
        $$
                \nabla \phi = \nabla f
        $$
        où $\phi$ est un réseau de neurones et $f$ l'image originale.
        """,
        unsafe_allow_html=True)
    
    # Options de configuration
    st.sidebar.header("Paramètres")
    
    # Choix de l'image
    image_option = st.sidebar.selectbox(
        "Image source:",
        ("camera", "cat", "astronaut", "immunohistochemistry", "brick", "coffee", "rocket"),
        index=1)
    
    image_size = st.sidebar.slider(
        "Résolution de l'image:",
        min_value=64, max_value=512,
        value=256, step=32)
    
    # Paramètres du modèle SIREN
    st.sidebar.subheader("Paramètres du modèle SIREN")

    hidden_features_siren = st.sidebar.slider(
        "Nombre de neurones cachés du modèle SIREN:",
        min_value=64, max_value=512,
        value=256, step=32)
    
    hidden_layers_siren = st.sidebar.slider(
        "Nombre de couches cachées du modèle SIREN:",
        min_value=1, max_value=5,
        value=3, step=1)
    
    # Paramètres du modèle standard
    st.sidebar.subheader("Paramètres du modèle standard")

    activation_option = st.sidebar.selectbox(
        "Fonction d'activation:",
        ("ReLU", "Sigmoid", "Tanh", "LeakyReLU"),
        index = 1)
    
    hidden_features_std = st.sidebar.slider(
        "Nombre de neurones cachés du modèle standard:",
        min_value=64, max_value=512,
        value=256, step=32)

    hidden_layers_std = st.sidebar.slider(
        "Nombre de couches cachées du modèle standard:",
        min_value=1, max_value=5,
        value=3, step=1)

    # Paramètres d'entraînement
    st.sidebar.subheader("Entraînement")

    learning_rate = st.sidebar.select_slider(
        "Taux d'apprentissage:",
        options=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
        format_func=lambda x: f"{x:.5f}",
        value=1e-4)
    
    num_epochs = st.sidebar.slider(
        "Nombre d'époques:",
        min_value=5, max_value=500,
        value=10, step=5)
    
    # Lancement de l'entraînement
    if st.sidebar.button("Lancer l'entraînement"):
        progress_bar = st.progress(0)
        epoch_status = st.empty()
        loss_siren_status = st.empty()
        loss_std_status = st.empty()
        
        # Zone pour afficher l'image
        result_container = st.empty()
        
        # Code d'expérimentation modifié pour Streamlit
        with st.spinner("Chargement de l'image..."):
            # Déterminer l'image à charger et la fonction d'activation choisie
            img_data = st_utils.choose_image(image_option)
            chosen_fct = st_utils.choose_function(activation_option)
            
            # Préparation des données
            img = utils.PoissonEqn(image_size, img_data)
            dataloader = DataLoader(img, batch_size=1, pin_memory=True, num_workers=0)
            X, y = next(iter(dataloader))
            X = X.to(device)
            ysave = y.copy()
            y = {key: value.to(device) for key, value in y.items()}
            
            # Instanciation des modèles, de l'optimiseur et du masque.
            siren = models.Siren(
                in_features=2, out_features=1,
                hidden_features=hidden_features_siren,
                hidden_layers=hidden_layers_siren,
                outermost_linear=True).to(device)
            
            standard = models.standard_network(
                activation = chosen_fct,
                in_features=2, out_features=1,
                hidden_features=hidden_features_std,
                hidden_layers=hidden_layers_std,
                outermost_linear=True).to(device)
            
            optim_siren = optim.Adam(siren.parameters(), lr=learning_rate)
            optim_std = optim.Adam(standard.parameters(), lr=learning_rate)
        
        # Boucle d'entraînement adaptée pour Streamlit
        losses_siren = []
        losses_std = []
        col1, col2 = st.columns([2, 1])
        
        with col1:
            siren_container = st.empty()
        
        with col2:
            metrics_container = st.empty()
        
        for epoch in range(num_epochs):
            output_siren, coords_siren = siren(X)
            output_std, coords_std = standard(X)
            
            loss_siren = loss_fn.gradients_mse(output_siren, coords_siren, y['grads'])
            loss_std = loss_fn.gradients_mse(output_std, coords_std, y['grads'])
            
            losses_siren.append(loss_siren.item())
            losses_std.append(loss_std.item())
            
            # Mise à jour de l'interface Streamlit
            progress = (epoch + 1) / num_epochs
            progress_bar.progress(progress)
            epoch_status.text(f"Époque: {epoch+1}/{num_epochs}")

            if epoch % 5 == 0 or epoch == num_epochs - 1:
                # Affichage de l'image
                fig, ax = plt.subplots(1, 2, figsize=(8,8))
                
                # Récupérer les prédictions et les reformer en image
                pred_siren = output_siren.detach().reshape(image_size, image_size).cpu().numpy()
                pred_std = output_std.detach().reshape(image_size, image_size).cpu().numpy()
                
                ax[0].imshow(pred_siren, cmap="viridis")
                ax[0].set_title(f"SIREN")
                ax[1].imshow(pred_std, cmap="viridis")
                ax[1].set_title(activation_option)
                ax[0].axis("off")
                ax[1].axis("off")
                
                # Mettre à jour l'image
                siren_container.pyplot(fig)
                
                # Mettre à jour les métriques
                metrics_fig, metrics_ax = plt.subplots(figsize=(4,4))
                metrics_ax.plot(losses_siren, label="SIREN")
                metrics_ax.plot(losses_std, label=activation_option)
                metrics_ax.set_xlabel("Époque")
                metrics_ax.set_ylabel("Perte MSE")
                metrics_ax.set_title("Évolution de la perte")
                metrics_ax.grid(True)
                metrics_ax.legend()
                metrics_container.pyplot(metrics_fig)
            
            optim_siren.zero_grad()
            optim_std.zero_grad()

            loss_siren.backward()
            loss_std.backward()

            optim_siren.step()
            optim_std.step()
        
        st.success("Entraînement terminé!")
        
        # Affichage du graphique de perte final
        st.subheader("Évolution de la perte")
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(losses_siren, label="SIREN")
        ax.plot(losses_std, label=activation_option)
        ax.set_xlabel("Époque")
        ax.set_ylabel("Perte MSE")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
        
        # Comparaison image originale vs prédiction
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Originale")
            original_img = y['pixels'].reshape(image_size, image_size).cpu().numpy()
            original_img = np.clip(original_img, .0, 1.0)
            st.image(original_img, use_container_width=True)

        with col2:
            st.subheader("SIREN")
            siren_img = output_siren.detach().reshape(image_size, image_size).cpu().numpy()
            siren_img = np.clip(siren_img, .0, 1.0)
            st.image(siren_img, use_container_width=True)
                
        with col3:
            st.subheader(activation_option)
            std_image = output_std.detach().reshape(image_size, image_size).cpu().numpy()
            std_image = np.clip(std_image, .0, 1.0)
            st.image(std_image, use_container_width=True)

    else:
        st.info("Configurez les paramètres dans la barre latérale et cliquez sur 'Lancer l'entraînement' pour commencer.")
        
        # Afficher un code d'exemple
        st.subheader("Code pour l'expérience 'Equation de Poisson'")
        st.code("""
            # Choose your image as previously.
            img = utils.PoissonEqn(128, skimage.data.cat())
            dataloader = DataLoader(img, batch_size=1, pin_memory=True, num_workers=0)
            X, y = next(iter(dataloader))
            X = X.to(device)
            y = {key: value.to(device) for key, value in y.items()}

            # Instantiate model, optimizer and number of epochs.
            siren = models.Siren(in_features=2, out_features=1,
                                hidden_features=256, hidden_layers=3,
                                outermost_linear=True).to(device)

            standard = models.standard_network(
                activation=nn.ReLU(),
                in_features=2, out_features=1,
                hidden_features=256, hidden_layers=3,
                outermost_linear=True).to(device)

            optim_siren = optim.Adam(siren.parameters(), lr=1e-4)
            optim_std = optim.Adam(standard.parameters(), lr=1e-4)
            num_epochs = 1000

            # Training loop.
            for epoch in range(num_epochs):
                output_siren, coords_siren = siren(X)
                output_std, coords_std = standard(X)

                # The networks are supervised using the gradient of the image.
                loss_siren = loss_fn.gradients_mse(output_siren, coords_siren, y['grads'])
                loss_std = loss_fn.gradients_mse(output_std, coords_std, y['grads'])

                optim_siren.zero_grad()
                optim_std.zero_grad()
                loss_siren.backward()
                loss_std.backward()
                optim_siren.step()
                optim_std.step()
            """)
        
        # Afficher une explication du code
        st.subheader("Explication du code")
        st.markdown("""
            Le code ci-dessus:
            
            1. **Charge une image** et prépare les données d'entrée (coordonnées X)
               et les valeurs cibles y (intensités des pixels).
            2. **Initialise un réseau SIREN** et un réseau ReLU avec:
                - 2 entrées (coordonnées x,y) ;
                - 1 sortie (intensité du pixel) ;
                - 3 couches cachées avec 256 neurones chacune.
            3. **Configure l'optimiseur Adam** avec un taux d'apprentissage de 0.0001.
            4. **Exécute une boucle d'entraînement** sur 500 époques où:
                - Le modèle génère des prédictions pour chaque coordonnée ;
                - L'erreur quadratique moyenne (MSE) est calculée par rapport au
                  gradient de l'image ;
                - L'image prédite est affichée périodiquement ;
                - Les poids du réseau sont mis à jour par rétropropagation.
            
            Dans l'interface Streamlit, vous pouvez personnaliser ces paramètres
            et observer en temps réel comment le réseau apprend à représenter
            l'image sélectionnée.
        """)

# Fonction pour la page Inpainting implicite
def inpainting_page():
    st.title("Inpainting implicite")
    
    st.markdown("""
        Dans cette section, les réseaux sont supervisés par une version masquée de l'image.
        La perte optimisée par les réseaux corresponds à la MSE calculée uniquement sur
        les pixels non-masqués.
        En revanche, la perte affichée sur les graphiques correspond à la MSE calculée
        entre la prédiction des modèles et l'image originale. 
    """)
    
    # Options de configuration
    st.sidebar.header("Paramètres")
    
    # Choix de l'image
    image_option = st.sidebar.selectbox(
        "Image source:",
        ("camera", "cat", "astronaut", "immunohistochemistry", "brick", "coffee", "rocket"),
        index=1)
    
    image_size = st.sidebar.slider(
        "Résolution de l'image:",
        min_value=64, max_value=512,
        value=256, step=32)
    
    # Paramètres du modèle SIREN
    st.sidebar.subheader("Paramètres du modèle SIREN")

    hidden_features_siren = st.sidebar.slider(
        "Nombre de neurones cachés du modèle SIREN:",
        min_value=64, max_value=512,
        value=256, step=32)
    
    hidden_layers_siren = st.sidebar.slider(
        "Nombre de couches cachées du modèle SIREN:",
        min_value=1, max_value=5,
        value=3, step=1)
    
    # Paramètres du modèle standard
    st.sidebar.subheader("Paramètres du modèle standard")

    activation_option = st.sidebar.selectbox(
        "Fonction d'activation:",
        ("ReLU", "Sigmoid", "Tanh", "LeakyReLU"),
        index = 1)
    
    hidden_features_std = st.sidebar.slider(
        "Nombre de neurones cachés du modèle standard:",
        min_value=64, max_value=512,
        value=256, step=32)

    hidden_layers_std = st.sidebar.slider(
        "Nombre de couches cachées du modèle standard:",
        min_value=1, max_value=5,
        value=3, step=1)

    # Paramètres du masque
    st.sidebar.subheader("Paramètres du masque")

    context_ratio = st.sidebar.slider(
        "Pourcentage de pixels conservés:",
        min_value=0, max_value=100,
        value=10, step=1)

    # Paramètres d'entraînement
    st.sidebar.subheader("Entraînement")

    learning_rate = st.sidebar.select_slider(
        "Taux d'apprentissage:",
        options=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
        format_func=lambda x: f"{x:.5f}",
        value=1e-4)
    
    num_epochs = st.sidebar.slider(
        "Nombre d'époques:",
        min_value=5, max_value=500,
        value=10, step=5)
    
    # Lancement de l'entraînement
    if st.sidebar.button("Lancer l'entraînement"):
        progress_bar = st.progress(0)
        epoch_status = st.empty()
        loss_siren_status = st.empty()
        loss_std_status = st.empty()
        
        # Zone pour afficher l'image
        result_container = st.empty()
        
        # Code d'expérimentation modifié pour Streamlit
        with st.spinner("Chargement de l'image..."):
            # Déterminer l'image à charger et la fonction d'activation choisie
            img_choice = st_utils.choose_image(image_option)
            chosen_fct = st_utils.choose_function(activation_option)
            
            # Préparation des données
            img = utils.ImageFitting(image_size, img_choice)
            dataloader = DataLoader(img, batch_size=1, pin_memory=True, num_workers=0)
            X, y = next(iter(dataloader))
            X, y = X.to(device), y.to(device)
            
            # Instanciation des modèles, de l'optimiseur et du masque.
            siren = models.Siren(
                in_features=2, out_features=1,
                hidden_features=hidden_features_siren,
                hidden_layers=hidden_layers_siren,
                outermost_linear=True).to(device)
            
            standard = models.standard_network(
                activation=chosen_fct,
                in_features=2, out_features=1,
                hidden_features=hidden_features_std,
                hidden_layers=hidden_layers_std,
                outermost_linear=True).to(device)
            
            optim_siren = optim.Adam(siren.parameters(), lr=learning_rate)
            optim_std = optim.Adam(standard.parameters(), lr = learning_rate)

            mask = utils.mask(1 - context_ratio/100, image_size)
            
        # Boucle d'entraînement adaptée pour Streamlit
        losses_siren = []
        losses_std = []
        col1, col2 = st.columns([2, 1])
        
        with col1:
            siren_container = st.empty()
        
        with col2:
            metrics_container = st.empty()
        
        for epoch in range(num_epochs):
            output_siren, coords_siren = siren(X)
            output_std, coords_std = standard(X)
            
            loss_siren = loss_fn.MSE(mask*output_siren, mask*y)
            loss_std = loss_fn.MSE(mask*output_std, mask*y)

            loss_total_siren = loss_fn.MSE(output_siren, y)
            loss_total_std = loss_fn.MSE(output_std, y)
            
            losses_siren.append(loss_total_siren.item())
            losses_std.append(loss_total_std.item())
            
            # Mise à jour de l'interface Streamlit
            progress = (epoch + 1) / num_epochs
            progress_bar.progress(progress)
            epoch_status.text(f"Époque: {epoch+1}/{num_epochs}")

            if epoch % 5 == 0 or epoch == num_epochs - 1:
                # Affichage de l'image
                fig, ax = plt.subplots(1, 2, figsize=(8,8))
                
                # Récupérer les prédictions et les reformer en image
                pred_siren = output_siren.detach().reshape(image_size, image_size).cpu().numpy()
                pred_std = output_std.detach().reshape(image_size, image_size).cpu().numpy()
                
                ax[0].imshow(pred_siren, cmap="viridis")
                ax[0].set_title(f"SIREN")
                ax[1].imshow(pred_std, cmap="viridis")
                ax[1].set_title(activation_option)
                ax[0].axis("off")
                ax[1].axis("off")
                
                # Mettre à jour l'image
                siren_container.pyplot(fig)
                
                # Mettre à jour les métriques
                metrics_fig, metrics_ax = plt.subplots(figsize=(4,4))
                metrics_ax.plot(losses_siren, label="SIREN")
                metrics_ax.plot(losses_std, label=activation_option)
                metrics_ax.set_xlabel("Époque")
                metrics_ax.set_ylabel("Perte MSE")
                metrics_ax.set_title("Évolution de la perte")
                metrics_ax.grid(True)
                metrics_ax.legend()
                metrics_container.pyplot(metrics_fig)
            
            optim_siren.zero_grad()
            optim_std.zero_grad()

            loss_siren.backward()
            loss_std.backward()

            optim_siren.step()
            optim_std.step()
        
        st.success("Entraînement terminé!")
        
        # Affichage du graphique de perte final
        st.subheader("Évolution de la perte")
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(losses_siren)
        ax.plot(losses_std)
        ax.set_xlabel("Époque")
        ax.set_ylabel("Perte MSE")
        ax.grid(True)
        st.pyplot(fig)
        
        # Comparaison image originale vs prédiction
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Originale")
            original_img = y.detach().reshape(image_size, image_size).cpu().numpy()
            original_img = np.clip(original_img, .0, 1.0)
            st.image(original_img, use_container_width=True)
        
        with col2:
            st.subheader("Masquée")
            masked_img = (mask * y).detach().reshape(image_size, image_size).cpu().numpy()
            masked_img = np.clip(masked_img, .0, 1.0)
            st.image(masked_img, use_container_width=True)
        
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("SIREN")
            predicted_img = output_siren.detach().reshape(image_size, image_size).cpu().numpy()
            predicted_img = np.clip(predicted_img, .0, 1.0)
            st.image(predicted_img, use_container_width=True)
        
        with col4:
            st.subheader(activation_option)
            predicted_img = output_std.detach().reshape(image_size, image_size).cpu().numpy()
            predicted_img = np.clip(predicted_img, .0, 1.0)
            st.image(predicted_img, use_container_width=True)
    
    else:
        st.info("Configurez les paramètres dans la barre latérale et cliquez sur 'Lancer l'entraînement' pour commencer.")
        
        # Afficher un code d'exemple
        st.subheader("Code pour l'expérience 'Image Fitting'")
        st.code("""
            # Get coordinates from the cameraman image (X) and the pixel values (y).
            img = utils.ImageFitting(256, skimage.data.cat())
            dataloader = DataLoader(img, batch_size=1, pin_memory=True, num_workers=0)
            X, y = next(iter(dataloader))
            X, y = X.to(device), y.to(device)
            y.requires_grad_(True)

            # Create mask.
            mask = utils.mask(.99, 256)

            # Instantiate model, optimizer and number of epochs.
            siren = models.Siren(
                in_features=2, out_features=1,
                hidden_features=256, hidden_layers=3,
                outermost_linear=True).to(device)

            standard = models.standard_network(
                activation=nn.ReLU(),
                in_features=2, out_features=1,
                hidden_features=256, hidden_layers=3,
                outermost_linear=True).to(device)

            optim_siren = optim.Adam(siren.parameters(), lr=1e-4)
            optim_std = optim.Adam(standard.parameters(), lr=1e-4)
            num_epochs = 1000

            # Training loop.
            for epoch in range(num_epochs):
                output_siren, coords_siren = siren(X)
                output_std, coords_std = standard(X)

                # The networks are supervised using only non-masked pixels.
                loss_siren = loss_fn.MSE(mask * output_siren, mask * y)
                loss_std = loss_fn.MSE(mask * output_std, mask * y)

                # Display results every 10 epochs.
                utils.display_img(epoch, 10, output_siren, coords_siren, loss_siren, 256, device)
                utils.display_img(epoch, 10, output_std, coords_std, loss_siren, 256, device)

                optim_siren.zero_grad()
                optim_std.zero_grad()
                loss_siren.backward()
                loss_std.backward()
                optim_siren.step()
                optim_std.step()
            """)
        
        # Afficher une explication du code
        st.subheader("Explication du code")
        st.markdown("""
            Le code ci-dessus:
            
            1. **Charge une image** et prépare les données d'entrée (coordonnées X) et les valeurs cibles y (intensités des pixels).
            2. **Initialise un réseau SIREN** et un réseau ReLU avec:
            - 2 entrées (coordonnées x,y) ;
            - 1 sortie (intensité du pixel) ;
            - 3 couches cachées avec 256 neurones chacune.
            3. **Crée un masque** pour masquer des pixels aléatoires de l'image.
            4. **Configure l'optimiseur Adam** avec un taux d'apprentissage de 0.0001.
            5. **Exécute une boucle d'entraînement** sur 500 époques où:
            - Le modèle génère des prédictions pour chaque coordonnée ;
            - L'erreur quadratique moyenne (MSE) est calculée uniquement sur les pixels non-masqués ;
            - L'image prédite est affichée périodiquement ;
            - Les poids du réseau sont mis à jour par rétropropagation.
            
            Dans l'interface Streamlit, vous pouvez personnaliser ces paramètres et observer en temps réel
            comment le réseau apprend à représenter l'image sélectionnée.
        """)


# Exécution de la page sélectionnée
if selection == "Présentation":
    presentation_page()
elif selection == "Image Fitting":
    image_fitting_page()
elif selection == "Equation de Poisson":
    poisson_page()
else:
    inpainting_page()