import pandas as pd
import random
import os

# Configuracion
DATA_DIR = r"D:\Agustin\Facultad\ProyectoFinal\archive"
CSV_FILE = os.path.join(DATA_DIR, "Data_Entry_2017.csv")
OUTPUT_FILE = "holdout_test_set.csv"
SAMPLES_PER_PATHOLOGY = 20
RANDOM_SEED = 42  # Para reproducibilidad

ALL_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
    'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

def main():
    print(f"Cargando dataset original desde {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    
    random.seed(RANDOM_SEED)
    
    holdout_indices = set()
    holdout_records = []
    
    # Pre-calcular que imagenes tienen cada patologia para busqueda rapida
    LABELS_TO_EXTRACT = ALL_LABELS + ['No Finding']
    pathology_to_images = {label: [] for label in LABELS_TO_EXTRACT}
    
    for idx, row in df.iterrows():
        labels_str = row['Finding Labels']
        if labels_str == 'No Finding':
            pathology_to_images['No Finding'].append(idx)
        else:
            for label in ALL_LABELS:
                if label in labels_str:
                    pathology_to_images[label].append(idx)
                    
    print("\nExtrayendo imagenes para el conjunto de validacion...")
    for label in LABELS_TO_EXTRACT:
        available_indices = pathology_to_images[label]
        # Filtrar aquellos que ya elegimos por otra patologia (evitar duplicados en el test set aunque tengan multiples etiquetas)
        available_indices = [idx for idx in available_indices if idx not in holdout_indices]
        
        if len(available_indices) >= SAMPLES_PER_PATHOLOGY:
            # Seleccionar N imagenes estricta y aleatoriamente
            selected_indices = random.sample(available_indices, SAMPLES_PER_PATHOLOGY)
            holdout_indices.update(selected_indices)
            print(f"- {label}: Seleccionadas {SAMPLES_PER_PATHOLOGY} muestras.")
            
            for idx in selected_indices:
                holdout_records.append({
                    'Target_Pathology_For_Extraction': label,
                    'Image Index': df.loc[idx, 'Image Index'],
                    'Original_Finding_Labels': df.loc[idx, 'Finding Labels']
                })
        else:
            print(f"- [AVISO] {label}: Solo se encontraron {len(available_indices)} muestras disponibles no solapadas. Se tomaran todas.")
            holdout_indices.update(available_indices)
            for idx in available_indices:
                holdout_records.append({
                    'Target_Pathology_For_Extraction': label,
                    'Image Index': df.loc[idx, 'Image Index'],
                    'Original_Finding_Labels': df.loc[idx, 'Finding Labels']
                })

    print(f"\nTotal de imagenes unicas extraidas: {len(holdout_indices)}")
    
    # Guardar a CSV
    out_df = pd.DataFrame(holdout_records)
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Conjunto de validacion guardado exitosamente en: {OUTPUT_FILE}")
    print("RECUERDE ACTUALIZAR dataset.py PARA EXCLUIR ESTAS IMAGENES DEL ENTRENAMIENTO.")

if __name__ == "__main__":
    main()
