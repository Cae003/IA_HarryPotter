import cv2
import face_recognition as fr

imgHermione = fr.load_image_file('ImagensFR/Hermione.jpg')
#imgHermione = cv2.cvtColor(imgHermione, cv2.COLOR_BGR2RGB) #Ativar só se a imgem estiver com as cores erradas
videoTeste = fr.load_video_file('Video.mp4')
#videoTeste = cv2.cvtColor(videoTeste, cv2.COLOR_BGR2RGB) #Ativar só se o video estiver com as cores erradas

# 1ª etapa: Reconhecer o rosto dentro da imagem com o algoritmo HOG
faceLoc = fr.face_locations(imgHermione)[0]
cv2.rectangle(imgHermione, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0,225,0), 2)
print(faceLoc) #printa a localização do rosto na imagem

# 2ª etapa: Fazer o enconding (codificação) do rosto
encodeHermione = fr.face_encodings(imgHermione)[0]
print(encodeHermione) #printa o vetor numérico do rosto
encodeVideoTeste = fr.face_encodings(videoTeste)[0] #Acho que aqui tem que ser um loop para pegar todos os frames do vídeo

# 3ª etapa: Comparar os encodings (codificações) dos rostos
comparacao = fr.compare_faces([encodeHermione], encodeVideoTeste)
distancia = fr.face_distance([encodeHermione], encodeVideoTeste)
print(comparacao) #printa True ou False dependendo se o rosto do vídeo é igual ao da imagem
print(distancia) #printa a distância entre os rostos (quanto menor a distância, mais parecidos eles são)

cv2.imshow('Hermione Original', imgHermione)
cv2.waitKey(0)