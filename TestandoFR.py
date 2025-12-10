import cv2
import face_recognition as fr

imgBase = fr.load_image_file('ImagensTeste/emma.jpeg')
imgBase = cv2.cvtColor(imgBase, cv2.COLOR_BGR2RGB) #Ativar só se a imgem estiver com as cores erradas
imgTeste = fr.load_image_file('ImagensTeste/margot.jpeg')
imgTeste = cv2.cvtColor(imgTeste, cv2.COLOR_BGR2RGB) #Ativar só se o video estiver com as cores erradas

# 1ª etapa: Reconhecer o rosto dentro da imagem com o algoritmo HOG
faceLoc = fr.face_locations(imgBase)[0]
cv2.rectangle(imgBase, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0,225,0), 2)
cv2.rectangle(imgTeste, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0,225,0), 2)
print(faceLoc) #printa a localização do rosto na imagem

# 2ª etapa: Fazer o enconding (codificação) do rosto
encodeImagem = fr.face_encodings(imgBase)[0]
print(encodeImagem) #printa o vetor numérico do rosto
encodeimgTeste = fr.face_encodings(imgTeste)[0] #Acho que aqui tem que ser um loop para pegar todos os frames do vídeo

# 3ª etapa: Comparar os encodings (codificações) dos rostos
comparacao = fr.compare_faces([encodeImagem], encodeimgTeste)
distancia = fr.face_distance([encodeImagem], encodeimgTeste)
print(comparacao) #printa True ou False dependendo se o rosto do vídeo é igual ao da imagem
print(distancia) #printa a distância entre os rostos (quanto menor a distância, mais parecidos eles são)

cv2.imshow('Imagem Original', imgBase)
cv2.imshow('Imagem Teste', imgTeste)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
