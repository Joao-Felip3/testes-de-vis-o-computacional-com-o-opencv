import cv2 as cv 
import os 
import numpy as np
# liata com os nomes das pessoas as quais cada pasta de imagens pertence 
# é importante colocar os mesmos nomes que estão como título das pastas de imagens
people = ["name1" "name2", "name3"]
#chama o modelo haarcascade para detecção de imagens 
haar_cascade = cv.CascadeClassifier("haar_faces_model.xml")

# Caminho do diretório contendo as pastas com as imagens de treinamento
dir_path = r"/home/joao/Pictures/detection/"

# lista com os caminos de cada pasta com imagens
p = []
for i in os.listdir(dir_path):
    p.append(i)

# Listas contendo as características (faces) e legendas  (identidades correspondentes)
features = []
labels = []


def create_train():
    for person in people:
        path = os.path.join(dir_path, person)
        label = people.index(person)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            
           #lê a imagem 
            img_array = cv.imread(img_path)

            # vesrifica se a leitura da imagem teve Êxito
            if img_array is None:
                print(f"Failed to read image: {img_path}")
                continue

            #converte pra escala cinza 
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            
           
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            # Extrai as faces e nomes correspindentes
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)


create_train()



features= np.array(features, dtype= "object")
labels = np.array(labels)
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features,labels)

face_recognizer.save("face_trained.yml")
np.save("features.npy", features)
np.save("labels.npy", labels)