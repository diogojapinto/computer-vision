import cv2 as cv

resources_folder = "../../../res/"
WEBCAM_URL = "http://172.30.31.244:8080/video?dummy=param.mjpg"

# Ask which sub exercise to display
option = input("Select exercise: (a/b)")

if option == 'a':
    print("Starting Exercise a)")
    video = cv.VideoCapture()
    video.open(WEBCAM_URL)

    while True:
        ret, frame = video.read()

        if ret is True:
            cv.imshow("Video", frame)

        if cv.waitKey(5) != -1:
            break

    video.release()
    cv.destroyAllWindows()

elif option == 'b':
    print("Starting Exercise b)")
else:
    print("That exercise doesn't exist!")
    exit()
