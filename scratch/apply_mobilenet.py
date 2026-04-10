import json

def apply_mobilenet():
    file_path = 'd:/Accident-Detection-System/accident-classification.ipynb'
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
            
        source = "".join(cell.get('source', []))
        
        # 1. Update batch_size
        if 'batch_size = 100' in source:
            new_source = source.replace('batch_size = 100', 'batch_size = 32')
            cell['source'] = [line + '\n' if i < len(new_source.split('\n'))-1 else line for i, line in enumerate(new_source.split('\n'))]

        # 2. Update the Model
        elif '## Defining Cnn' in source:
            new_source = """## Defining Pre-trained MobileNetV2 Model
base_model = tf.keras.applications.MobileNetV2(input_shape=(img_height, img_width, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

model = tf.keras.models.Sequential([
  layers.RandomFlip("horizontal_and_vertical", input_shape=(img_height, img_width, 3)),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.2),
  
  # Rescaling pixel values for MobileNetV2 which expects [-1, 1] inputs
  layers.Rescaling(1./127.5, offset=-1),
  
  base_model,
  layers.GlobalAveragePooling2D(),
  layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])"""
            cell['source'] = [line + '\n' if i < len(new_source.split('\n'))-1 else line for i, line in enumerate(new_source.split('\n'))]

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    apply_mobilenet()
    print("Notebook updated to MobileNetV2.")
