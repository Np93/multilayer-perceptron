import pandas as pd
import os

# Définir le chemin du dossier data
data_folder = 'data'
data_file = os.path.join(data_folder, 'data.csv')

# 1. Charger les données depuis data.csv sans noms de colonnes
data = pd.read_csv(data_file, header=None)

# 2. Supprimer la colonne id (première colonne)
data = data.drop(columns=[0])

# 3. Vérifier qu'il y a 31 colonnes restantes (diagnosis + 30 features)
assert data.shape[1] == 31, "Il manque des colonnes dans les données."

# 4. Séparer les données en deux groupes : 'B' et 'M'
data_benign = data[data[1] == 'B']
data_malignant = data[data[1] == 'M']

# 5. Calculer le nombre total de lignes
total_benign = len(data_benign)
total_malignant = len(data_malignant)
total_data = len(data)

# 6. Calculer les tailles des ensembles en maintenant les proportions spécifiées
def calculate_class_distribution(total_lines, total_benign, total_malignant):
    num_benign = int((total_benign / (total_benign + total_malignant)) * total_lines)
    num_malignant = total_lines - num_benign
    return num_benign, num_malignant

test_size = int(total_data * 0.25)
valid_size = int(total_data * 0.15)
train_size = total_data - test_size - valid_size

num_benign_test, num_malignant_test = calculate_class_distribution(test_size, total_benign, total_malignant)
num_benign_valid, num_malignant_valid = calculate_class_distribution(valid_size, total_benign, total_malignant)

# 7. Échantillonner les ensembles de test et validation
test_benign = data_benign.sample(n=num_benign_test, random_state=0)
remaining_benign = data_benign.drop(test_benign.index)

test_malignant = data_malignant.sample(n=num_malignant_test, random_state=0)
remaining_malignant = data_malignant.drop(test_malignant.index)

valid_benign = remaining_benign.sample(n=num_benign_valid, random_state=0)
remaining_benign = remaining_benign.drop(valid_benign.index)

valid_malignant = remaining_malignant.sample(n=num_malignant_valid, random_state=0)
remaining_malignant = remaining_malignant.drop(valid_malignant.index)

# 8. Utiliser le reste des données pour l'ensemble d'entraînement
train_benign = remaining_benign
train_malignant = remaining_malignant

# 9. Combiner les ensembles et mélanger
train = pd.concat([train_benign, train_malignant]).sample(frac=1, random_state=0).reset_index(drop=True)
test = pd.concat([test_benign, test_malignant]).sample(frac=1, random_state=0).reset_index(drop=True)
valid = pd.concat([valid_benign, valid_malignant]).sample(frac=1, random_state=0).reset_index(drop=True)

# 10. Vérifier que les proportions sont correctes
def check_proportions(df, expected_total, expected_benign, expected_malignant):
    benign_count = df[1].value_counts().get('B', 0)
    malignant_count = df[1].value_counts().get('M', 0)
    assert benign_count + malignant_count == expected_total, f"Le total des lignes n'est pas correct : {benign_count + malignant_count} != {expected_total}"
    assert abs((benign_count / (benign_count + malignant_count)) - (expected_benign / (expected_benign + expected_malignant))) < 0.01, "Les proportions de bénins ne sont pas correctes"
    assert abs((malignant_count / (benign_count + malignant_count)) - (expected_malignant / (expected_benign + expected_malignant))) < 0.01, "Les proportions de malins ne sont pas correctes"

check_proportions(train, len(train), len(train_benign), len(train_malignant))
check_proportions(test, len(test), num_benign_test, num_malignant_test)
check_proportions(valid, len(valid), num_benign_valid, num_malignant_valid)

# 11. Sauvegarder les ensembles dans des fichiers CSV dans le dossier data sans en-têtes
train.to_csv(os.path.join(data_folder, 'train.csv'), index=False, header=False)
valid.to_csv(os.path.join(data_folder, 'valid.csv'), index=False, header=False)
test.to_csv(os.path.join(data_folder, 'test.csv'), index=False, header=False)

print("Les ensembles de données ont été divisés et enregistrés avec succès dans le dossier 'data'.")