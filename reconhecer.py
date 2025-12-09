import cv2
import face_recognition as fr

# ======== CARREGAR IMAGENS DE TREINO ========
print("Carregando imagens de referência...")

imgHermione = fr.load_image_file('ImagensFR/Hermione.jpg')
encodeHermione = fr.face_encodings(imgHermione)[0]

imgHarryOculos = fr.load_image_file('ImagensFR/Harry_O.jpg')
encodeHarryOculos = fr.face_encodings(imgHarryOculos)[0]
imgHarry = fr.load_image_file('ImagensFR/Harry.jpg')
encodeHarry = fr.face_encodings(imgHarry)[0]

imgRony = fr.load_image_file('ImagensFR/Rony.jpg')
encodeRony = fr.face_encodings(imgRony)[0]

known_encodings = [encodeHermione, encodeHarryOculos, encodeHarry, encodeRony]
known_names = ["Hermione", "Harry", "Harry", "Rony"]

print("Encodings carregados!\n")

# ======== ABRIR VÍDEO ORIGINAL ========
video = cv2.VideoCapture('Video2.mp4')

if not video.isOpened():
    print("Erro ao abrir o vídeo!")
    exit()

# ======== CONFIGURAR VIDEO DE SAÍDA ========
fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # codec
fps = int(video.get(cv2.CAP_PROP_FPS))     # mesmo FPS do original
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

output = cv2.VideoWriter('Video_Processado.mp4', fourcc, fps, (width, height))

print("Gerando novo vídeo processado...\n")

# ======== PROCESSAR FRAME A FRAME ========
while True:
    ret, frame = video.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar rostos
    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    # Para cada rosto detectado
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = fr.compare_faces(known_encodings, face_encoding)
        print(matches)
        distances = fr.face_distance(known_encodings, face_encoding)
        print(distances)

        best_match = distances.argmin()
        print(best_match)
        name = "Desconhecido"

        if matches[best_match]:
            name = known_names[best_match]

        # Desenhar bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(
            frame,
            f"{name} ({distances[best_match]:.2f})",
            (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    # Mostrar vídeo
    cv2.imshow("Reconhecimento - Harry Potter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Escrever o frame no vídeo de saída
    output.write(frame)

# Finalizar
video.release()
output.release()
cv2.destroyAllWindows()

print("Processo concluído! Video salvo como: Video_Processado.mp4")