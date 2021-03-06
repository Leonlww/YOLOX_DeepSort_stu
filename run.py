import imutils
import cv2

from objtracker import Tracker


def main():
    cap = cv2.VideoCapture('你自己准备用来跟踪的视频路径.mp4')
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
                '保存结果的视频路径.mp4', fourcc, fps, (result.shape[1], result.shape[0]))

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