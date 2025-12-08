import cv2
import face_recognition as fr

# ======== CARREGAR IMAGENS DE TREINO ========
print("Carregando imagens de referência...")

# Hermione
imgHermione = fr.load_image_file('ImagensFR/Hermione.jpg')
encodeHermione = fr.face_encodings(imgHermione)[0]

# Harry
imgHarry = fr.load_image_file('ImagensFR/Harry.jpg')
encodeHarry = fr.face_encodings(imgHarry)[0]

# Rony
imgRony = fr.load_image_file('ImagensFR/Rony.jpg')
encodeRony = fr.face_encodings(imgRony)[0]

known_encodings = [encodeHermione, encodeHarry, encodeRony]
known_names = ["Hermione", "Harry", "Rony"]

print("Encodings carregados!\n")

# ======== ABRIR O VÍDEO ========
video = cv2.VideoCapture('Video2.mp4')

if not video.isOpened():
    print("Erro ao abrir o vídeo!")
    exit()

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Reduz processamento (opcional)
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar rostos no frame
    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    # Para cada rosto detectado
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        # Comparar com os 3 personagens
        matches = fr.compare_faces(known_encodings, face_encoding)
        distances = fr.face_distance(known_encodings, face_encoding)

        # Escolher o mais próximo
        best_match = distances.argmin()
        name = "Desconhecido"

        # Se o rosto for parecido o suficiente
        if matches[best_match]:
            name = known_names[best_match]

        # Desenhar retângulo e nome no vídeo
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, f"{name} ({distances[best_match]:.2f})", (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Mostrar vídeo
    cv2.imshow("Reconhecimento - Harry Potter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()