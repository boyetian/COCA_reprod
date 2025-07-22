import os
import nltk
from nltk.corpus import wordnet as wn
import json

def create_imagenet_class_index_dict():
    """
    Creates a dictionary mapping ImageNet class names to their indices.

    The indices are based on the sorted order of the class folders in the dataset.
    The class names are derived from the folder names (WordNet synset IDs)
    using nltk.
    """
    # We can use any sub-directory to get the class folders, as they are consistent.
    base_data_dir = 'data/Tiny-ImageNet-C'
    
    # Find a valid directory to scan for class folders
    data_dir = None
    if os.path.exists(base_data_dir):
        corruption_types = sorted([d for d in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, d))])
        if corruption_types:
            severities = sorted([d for d in os.listdir(os.path.join(base_data_dir, corruption_types[0])) if os.path.isdir(os.path.join(base_data_dir, corruption_types[0], d))])
            if severities:
                data_dir = os.path.join(base_data_dir, corruption_types[0], severities[0])

    if not data_dir or not os.path.exists(data_dir):
        print(f"Could not find a suitable data directory in {base_data_dir}")
        print("Please ensure the Tiny-ImageNet-C dataset is correctly extracted.")
        return None

    try:
        # Get sorted list of class folder names (e.g., 'n01443537')
        class_folders = sorted([d for d in os.listdir(data_dir) if d.startswith('n') and os.path.isdir(os.path.join(data_dir, d))])
    except FileNotFoundError:
        print(f"Data directory not found at {data_dir}")
        print("Please make sure you have downloaded and extracted the Tiny-ImageNet-C dataset.")
        return None

    # Download wordnet if not already available
    try:
        wn.synsets('test')
    except nltk.downloader.DownloadError:
        print("Downloading WordNet corpus...")
        nltk.download('wordnet')

    class_to_idx = {}
    for i, folder_name in enumerate(class_folders):
        try:
            synset = wn.synset_from_pos_and_offset('n', int(folder_name[1:]))
            # Use the first lemma as the representative name for the class
            class_name = synset.lemmas()[0].name()
            class_to_idx[class_name] = i
        except Exception as e:
            print(f"Could not process folder {folder_name}: {e}")
            
    return class_to_idx

if __name__ == '__main__':
    imagenet_class_dict = create_imagenet_class_index_dict()
    if imagenet_class_dict:
        print("ImageNet Class to Index Dictionary:")
        # Pretty print the dictionary
        print(json.dumps(imagenet_class_dict, indent=4))

        # Save to a file
        output_filename = 'imagenet_class_index.json'
        with open(output_filename, 'w') as f:
            json.dump(imagenet_class_dict, f, indent=4)
        print(f"\nDictionary saved to {output_filename}")
