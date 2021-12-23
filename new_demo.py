import imutils
import cv2

from both_tracker import Tracker


def main():
    cap = cv2.VideoCapture('/datav/shared/leon/YOLOX_DeepSort_stu/YOLOX/test_video/tonya-attack.mp4')
    fps = int(cap.get(5))
    videoWriter = None
    tracker = Tracker()
    while True:

        _, im = cap.read()
        if im is None:
            break

        im = imutils.resize(im, height=500)
        result,_ = tracker.update(im)

        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 
            videoWriter = cv2.VideoWriter(
                '/datav/shared/leon/YOLOX_DeepSort_stu/result/tonya_attacked_track_result_newdemo3.mp4', fourcc, fps, (result.shape[1], result.shape[0]))

        videoWriter.write(result)
        # cv2.imshow(name, result)
        # cv2.waitKey(t)

        # if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
        #     # 点x退出
        #     break

    cap.release()
    videoWriter.release()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    
    main()