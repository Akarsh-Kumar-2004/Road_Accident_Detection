import json

def fix_notebook():
    file_path = 'd:/Accident-Detection-System/accident-classification.ipynb'
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            
            if "plt.plot(history.history['loss']" in source and "plt.plot(history.history['accuracy']" in source:
                cell['source'] = [
                    "## stats on training data\n",
                    "plt.plot(history.history['loss'], label = 'training loss')\n",
                    "plt.plot(history.history['val_loss'], label = 'validation loss')\n",
                    "plt.title('Training and Validation Loss')\n",
                    "plt.grid(True)\n",
                    "plt.legend()\n"
                ]
            elif "plt.plot(history.history['val_loss']" in source and "plt.plot(history.history['val_accuracy']" in source:
                cell['source'] = [
                    "## stats on validation data\n",
                    "plt.plot(history.history['accuracy'], label = 'training accuracy')\n",
                    "plt.plot(history.history['val_accuracy'], label = 'validation accuracy')\n",
                    "plt.title('Training and Validation Accuracy')\n",
                    "plt.grid(True)\n",
                    "plt.legend()\n"
                ]

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print("Notebook updated successfully.")

if __name__ == "__main__":
    fix_notebook()
