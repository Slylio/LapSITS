# Fonctionnalités ajoutées au système Watershed

## 1. Contrôle du niveau de détail
- **Slider de niveau de détail (0.0 à 1.0)** : Filtre les composantes par aire
  - 0.0 = très simplifié (garde seulement les grandes composantes)
  - 1.0 = tous les détails (garde presque toutes les composantes)
- **Filtrage automatique** : Supprime les petites composantes en modifiant leurs altitudes
- **Recalcul optimisé** : Le calcul ne se déclenche que quand le clic de la souris est relâché (évite le calcul continu)

## 2. Contrôle du niveau de coupe horizontale
- **Champ de texte** : Permet de spécifier manuellement le niveau de coupe (0 à max)
- **Validation** : Vérifie que la valeur est dans la plage valide
- **Affichage du maximum** : Montre "niveau / max" pour guider l'utilisateur
- **Reconstruction RGB** : Génère une image RGB simplifiée au niveau spécifié

## 3. Affichage RGB pour le watershed
- **Image en couleur** : Affiche l'image RGB originale au lieu du niveau de gris
- **Reconstruction interactive** : Met à jour l'image quand on change le niveau de coupe
- **Couleurs moyennes** : Utilise les couleurs moyennes de chaque région segmentée

## 4. Interface utilisateur
- **Contrôles conditionnels** : Les contrôles watershed n'apparaissent que pour ce type d'arbre
- **Messages informatifs** : Explique les nouvelles fonctionnalités lors du changement de type
- **Gestion d'erreurs** : Valide les entrées utilisateur et affiche des messages d'erreur

## 5. Fonctions ajoutées

### Dans `watershed.py`:
- `apply_detail_level_filter(tree, altitudes, detail_level)` : Filtre par niveau de détail
- `reconstruct_rgb_at_level(tree, altitudes, cube_rgb, level)` : Reconstruit l'image RGB
- `get_max_hierarchy_level(tree, altitudes)` : Calcule le niveau maximum valide

### Dans `core/tree.py`:
- Paramètre `detail_level` ajouté à `compute_tree_with_type()`
- Paramètre `detail_level` ajouté à `compute_watershed_tree()`

### Dans `gui/main_window.py`:
- Variables : `current_detail_level`, `current_cut_level`, `max_cut_level`, `cube_rgb`
- Contrôles : `detail_level_slider`, `cut_level_input`, `cut_level_max_label`
- Fonctions : `on_detail_level_changed()`, `on_detail_level_value_display()`, `on_cut_level_changed()`, `update_watershed_image_display()`

### Dans `core/pattern_spectra.py`:
- Paramètre `detail_level` ajouté à `compute_global_ps()`

## 6. Utilisation

1. **Sélectionner le watershed** : Choisir "Watershed" dans la liste des types d'arbres
2. **Ajuster le niveau de détail** : Utiliser le slider (0.0 = simplifié, 1.0 = détaillé)
3. **Ajuster le niveau de coupe** : Entrer un nombre dans le champ (0 à max)
4. **Voir l'image RGB** : L'image se met à jour automatiquement avec les couleurs moyennes des régions

## 7. Exemple d'utilisation

```python
# Calculer l'arbre watershed avec niveau de détail
tree, altitudes = compute_tree_with_type(cube_rgb, 'watershed', detail_level=0.3)

# Obtenir le niveau maximum
max_level = get_max_hierarchy_level(tree, altitudes)

# Reconstruire l'image à un niveau spécifique
recon_image = reconstruct_rgb_at_level(tree, altitudes, cube_rgb, level=max_level//2)
```

## 8. Avantages

- **Contrôle précis** : Permet de filtrer les détails non désirés
- **Visualisation intuitive** : L'image RGB aide à comprendre la segmentation
- **Simplification interactive** : Voir l'effet du niveau de coupe en temps réel
- **Interface cohérente** : Les contrôles apparaissent seulement quand appropriés
- **Performance optimisée** : Le recalcul du Pattern Spectra ne se déclenche que lors du relâchement du slider
