import cv2

def detect(img, sf, mn):
    img = cv2.imread(f'img/{img}')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    result = faces.detectMultiScale(gray, scaleFactor=sf, minNeighbors=mn)

    for (x, y, w, h) in result:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)

    cv2.imshow('img', img)
    cv2.waitKey(3000)

def main():
    detect('1.jpeg', 1.1, 4)
    detect('2.jpeg', 1.1, 4)
    detect('3.jpeg', 1.1, 4)
    
if __name__ == "__main__":
    main()