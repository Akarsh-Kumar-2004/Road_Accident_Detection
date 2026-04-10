import json

def fix_notebook():
    file_path = 'd:/Accident-Detection-System/accident-classification.ipynb'
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            
            if "layers.BatchNormalization()" in source and "layers.Conv2D" in source:
                new_source = """## Defining Cnn with Regularization to prevent Overfitting
model = tf.keras.models.Sequential([
  # Data Augmentation to introduce noise
  layers.RandomFlip("horizontal_and_vertical", input_shape=(img_height, img_width, 3)),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.2),
  
  layers.BatchNormalization(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.25),
  
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.25),
  
  layers.Conv2D(128, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.25),
  
  layers.Conv2D(256, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.25),
  
  layers.Flatten(),
  layers.Dense(512, activation='relu'),
  layers.Dropout(0.5), # High dropout for dense layer
  layers.Dense(len(class_names), activation= 'softmax')
])

# Using adam optimizer with a slightly lower learning rate or default
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])"""
                cell['source'] = [line + '\n' for line in new_source.split('\n')]
                # remove trailing newline on the last element to avoid double space
                if cell['source']:
                    cell['source'][-1] = cell['source'][-1].rstrip('\n')

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
if __name__ == "__main__":
    fix_notebook()
