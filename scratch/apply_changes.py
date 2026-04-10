import json

def apply_changes():
    file_path = 'd:/Accident-Detection-System/accident-classification.ipynb'
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
            
        source = "".join(cell.get('source', []))
        
        # 1. Update Imports
        if 'from keras.callbacks import ModelCheckpoint' in source:
            new_source = source.replace('from keras.callbacks import ModelCheckpoint', 
                                        'from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau')
            cell['source'] = [line + '\n' if i < len(new_source.split('\n'))-1 else line for i, line in enumerate(new_source.split('\n'))]
        
        # 2. Update Training Dataset loading
        elif '## loading training set' in source:
            new_source = """## loading training set
training_data = tf.keras.preprocessing.image_dataset_from_directory(
    'data/train',
    validation_split=0.05,
    subset="training",
    seed=42,
    image_size= (img_height, img_width),
    batch_size=batch_size,
    color_mode='rgb'
)"""
            cell['source'] = [line + '\n' if i < len(new_source.split('\n'))-1 else line for i, line in enumerate(new_source.split('\n'))]
            
        # 3. Update Validation Dataset loading
        elif '## loading validation dataset' in source:
            new_source = """## loading validation dataset
validation_data =  tf.keras.preprocessing.image_dataset_from_directory(
    'data/train',
    validation_split=0.05,
    subset="validation",
    seed=42,
    image_size= (img_height, img_width),
    batch_size=batch_size,
    color_mode='rgb'
)"""
            cell['source'] = [line + '\n' if i < len(new_source.split('\n'))-1 else line for i, line in enumerate(new_source.split('\n'))]
            
        # 4. Remove Dropout from Model
        elif '## Defining Cnn' in source:
            new_source = """## Defining Cnn with Regularization to prevent Object Overfitting
model = tf.keras.models.Sequential([
  layers.RandomFlip("horizontal_and_vertical", input_shape=(img_height, img_width, 3)),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.2),
  layers.BatchNormalization(),
  layers.Conv2D(32, 3, activation='relu'), 
  layers.MaxPooling2D(), 
  
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),
  
  layers.Conv2D(128, 3, activation='relu'),
  layers.MaxPooling2D(),
  
  layers.Conv2D(256, 3, activation='relu'),
  layers.MaxPooling2D(),
  
  layers.Flatten(),
  layers.Dense(512, activation='relu'),
  layers.Dense(len(class_names), activation= 'softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])"""
            cell['source'] = [line + '\n' if i < len(new_source.split('\n'))-1 else line for i, line in enumerate(new_source.split('\n'))]

        # 5. Add ReduceLROnPlateau to callbacks
        elif '## lets train our CNN' in source:
            new_source = """## lets train our CNN
checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
callbacks_list = [checkpoint, reduce_lr]
history = model.fit(training_data, validation_data=validation_data, epochs = 25, callbacks=callbacks_list)"""
            cell['source'] = [line + '\n' if i < len(new_source.split('\n'))-1 else line for i, line in enumerate(new_source.split('\n'))]

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    apply_changes()
    print("Notebook updated.")
