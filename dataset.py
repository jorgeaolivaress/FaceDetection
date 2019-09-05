import cv2
import os

#Se declara en la libreria la captura de video
web_cam = cv2.VideoCapture(0)

#Utilizaremos el XML que procesa imagenes frontales
cascPath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

count = 0

#El nombre ingresado sera el que tendrá la carpeta (Se utiliza para identificar la persona)
nameFolder = input("Ingrese su nombre: ")
if not os.path.isdir("images/"+nameFolder):
    os.mkdir("images/"+nameFolder) 

#Remplazamos los espacios por guiones bajos para no tener problemas en los nombres de las imagenes
nameImage = nameFolder.replace(" ","_")
while(True):

    #comenzamos la lectura de los datos capturados por la webcam
    _, imagen_marco = web_cam.read()

    #grises = cv2.cvtColor(imagen_marco, cv2.COLOR_BGR2GRAY)
    colors = cv2.cvtColor(imagen_marco, cv2.COLOR_BGR2RGB)

    #detectamos el rostro con colores en rojo, verde y azul
    rostro = faceCascade.detectMultiScale(colors, 1.5, 5)

    for(x,y,w,h) in rostro:
        cv2.rectangle(imagen_marco, (x,y), (x+w, y+h), (255,0,0), 4)
        count += 1
        #guardamos las imagenes en el directorio creado junto con un contador y la matriz de colores detectados
        cv2.imwrite("images/"+nameFolder+"/"+nameImage+"_"+str(count)+".jpg", colors[y:y+h, x:x+w])
        cv2.imshow("Creando Dataset", imagen_marco)

    #el sistema se cierra si se presiona la letra q 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #en caso contrario si llega a las 400 tambien se cierra
    elif count >= 400:
        break


# Cuando todo está hecho, liberamos la captura
web_cam.release()
cv2.destroyAllWindows()