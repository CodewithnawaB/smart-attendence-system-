import os
import ssl
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

import cv2

import numpy as np
from keras_facenet import FaceNet

def create_embeddings(dataset_path='C:\\Face_Attendance_Project\\Dataset', 
                     output_embeddings='C:\\Face_Attendance_Project\\embeddings.npy',
                     output_labels='C:\\Face_Attendance_Project\\labels.npy'):
    """Create embeddings from dataset images"""
    
    print("Initializing FaceNet...")
    embedder = FaceNet()
    
    embeddings = []
    labels = []
    
    print(f"Loading images from: {dataset_path}\n")
    
    # Iterate through student folders
    for label in sorted(os.listdir(dataset_path)):
        folder = os.path.join(dataset_path, label)
        
        if not os.path.isdir(folder):
            continue
        
        print(f"Processing: {label}")
        count = 0
        
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            
            # Check if it's an image file
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            
            try:
                # Read image
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"  ⚠ Skipped (cannot read): {img_name}")
                    continue
                
                # Resize and convert to RGB
                img = cv2.resize(img, (160, 160))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Get embedding
                emb = embedder.embeddings([img])[0]
                embeddings.append(emb)
                labels.append(label)
                count += 1
                
            except Exception as e:
                print(f"  ⚠ Error processing {img_name}: {e}")
                continue
        
        print(f"  ✓ Processed {count} images\n")
    
    # Save embeddings and labels
    print(f"\nSaving embeddings...")
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    np.save(output_embeddings, embeddings)
    np.save(output_labels, labels)
    
    print(f"✅ Embeddings created successfully!")
    print(f"📊 Total samples: {len(embeddings)}")
    print(f"👥 Total students: {len(np.unique(labels))}")
    print(f"💾 Saved: {output_embeddings}")
    print(f"💾 Saved: {output_labels}")
    
    return embeddings, labels


if __name__ == "__main__":
    try:
        create_embeddings()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure the dataset path is correct and all dependencies are installed.")