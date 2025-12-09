import cv2
import face_recognition as fr

# ======== CARREGAR IMAGENS DE TREINO ========
print("Carregando imagens de referência...")

imgCaetano = fr.load_image_file('Integrantes_Projeto/Caetano.jpg')
encodeCaetano = fr.face_encodings(imgCaetano)[0]

known_encodings = [encodeCaetano]
known_names = ["Caetano"]

print("Encodings carregados!\n")

# ======== ABRIR WEBCAM (0 = webcam padrão) ========
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Erro ao acessar a webcam!")
    exit()

print("Webcam aberta! Pressione 'q' para sair.\n")

# ======== LOOP PRINCIPAL - PROCESSAR WEBCAM ========
while True:
    ret, frame = video.read()
    if not ret:
        print("Erro ao capturar frame da webcam.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar rostos
    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    # Para cada rosto detectado
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = fr.compare_faces(known_encodings, face_encoding)
        distances = fr.face_distance(known_encodings, face_encoding)

        best_match = distances.argmin()
        name = "Desconhecido"

        if matches[best_match]:
            name = known_names[best_match]

        # Desenhar caixas e texto
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

    # Mostrar na tela
    cv2.imshow("Reconhecimento Facial - Webcam", frame)

    # Pressionar 'q' para encerrar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Finalizar
video.release()
cv2.destroyAllWindows()

print("Encerrado.")