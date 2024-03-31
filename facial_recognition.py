#Real time recognition with webcam

import cv2 as cv

# chama o modelo haarcascade para detecção de faces 
haar_cascade = cv.CascadeClassifier("haar_faces_model.xml")

# List of people
people = ["Anne Hathaway", "Christian Bale","João Felipe"]

# chama o modelo pré-treinado de reconhecimento de faces LBPH face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")

# Recebe as capturas da camera com o índice especificado, nesse caso 0  corresponde à webcam
my_Captures = cv.VideoCapture(0)

while True:
    # ret é um parâmetro booleano que retorna se o frame foi capturado ou não com sucesso

    ret, frame = my_Captures.read()
    
    if not ret:
        print("Error while reading frames!")
        break

    # converte cada frame capturado pra escad=la de cinza
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Realiza a detecção de faces na imagem 
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    
    for (x, y, width, height) in faces_rect:
        # extrai a região de interesse na imagem 
        face_roi = gray[y:y+height, x:x+width]

        # Faz o reconhecimento de faces
        label, confidence = face_recognizer.predict(face_roi)
       

        # esenha o retângulo em volta da região de interesse
        cv.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 255), thickness=2)

        # legenda com o nome da pessoa e a acurácia 
        cv.putText(frame, f"{people[label]} ({confidence:.2f}%)", (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), thickness=2)

    # mostra os frames resultantes 
    cv.imshow('Webcam video', frame)

    # fecha o programa quando ocorre um atraso maior que 42 milissegundos pór frame 
    if cv.waitKey(42) & 0xFF == ord('q'):
        break

# libera as capturas 
my_Captures.release()

# fecha as janelas do opencv quando o programa encerra 
cv.destroyAllWindows()
