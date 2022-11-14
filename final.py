import cv2

#img_file = "D:\Python\ML_FINAL\car.jpg" araba görseli üzerinden yapacağımız işlemin ön hazırlığı için
mp4_file = "D:\Python\ML_FINAL\cars.mp4" #arabalar videosunun path'i
cars_mp4 = cv2.VideoCapture(mp4_file) #video dosyası olduğu için bu şekilde belirttik
car_cascade = cv2.CascadeClassifier('D:\Python\ML_FINAL\cars.xml') #arabaların tanımlanabilmesi için makinenin öğrendiği veri
#img = cv2.imread(img_file) video üzerinde yapacaüımız işlemin ön hazırlığı için tanımladığımız görseli okumaya yara

#blackWhite = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) görseli siyah beyaz yapar böylelikle işlem yükü azalır
#cars = car_cascade.detectMultiScale(blackWhite) siyah beyaz görselde çoklu ölçek tespiti

#for (x,y,w,h) in cars: ayrılan çoklu bölüm üzerinde oluşturacağımız karenin dört köşesi için kaç araba varsa 
    #cv2.rectangle(blackWhite, (x,y), (x+w,y+h), (0,0,255), 2) 2 birim kalınlıkta bir kare oluşturur
    #cv2.imshow('AI Car Detector', blackWhite) görsel üzerinde arabayı kare içine alan çıktı
    #cv2.waitKey(0)

while True:

    (read_succesful, frames) = cars_mp4.read()  #mp4 dosyasını okutuyoruz ve framelere ayırmak için tanımlıyoruz

    if read_succesful: #okuma başarılıysa

        grayscale_frames = cv2.cvtColor(frames, cv2.COLOR_RGB2GRAY) #her frame siyah beyaz tanımlanıyor işlem yükünü azaltmak için

    else:
        break
    
    cars = car_cascade.detectMultiScale(grayscale_frames) #siyah beyaz framelerde çoklu ölçek tespiti

    for (x,y,w,h) in cars: #video üzerinde araçları kare içine almak için
        cv2.rectangle(frames, (x,y), (x+w,y+h), (0,0,255), 2) # frameler üzerinde 2 birim kalınlığında kareler oluşturuluyor

    cv2.imshow('AI Car Detector', frames) #video üzerinde arabayı kare içine alan çıktı
    cv2.waitKey(1)
