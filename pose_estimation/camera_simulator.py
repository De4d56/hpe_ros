import rospy
import sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


def img_publisher(video):
    publisher = rospy.Publisher("camera", Image, queue_size=1)
    bridge = CvBridge()

    filename = video + "/rgb-000"
    this_filename = "bad_name"
    for i in range(0, 233):
        if i < 10:
            this_filename = filename + "00" + str(i) + ".jpg"
        elif i < 100:
            this_filename = filename + "0" + str(i) + ".jpg"
        elif i < 1000:
            this_filename = filename + str(i) + ".jpg"
        print(this_filename)
        img = cv2.imread(this_filename, flags=cv2.IMREAD_COLOR)
        if img is None:
            print("NULL img")
            return
        image_message = bridge.cv2_to_imgmsg(img, encoding="passthrough")
        publisher.publish(image_message)
        #cv2.imshow("Output", img)
        #cv2.waitKey(0)
        rospy.sleep(0.1)

    #cap = cv2.VideoCapture('chaplin.mp4')
    #if not cap.isOpened():
    #    print("Error opening video stream or file")
    #    return

    #while cap.isOpened():
    #    ret, frame = cap.read()
    #    rospy.sleep(0.04)
    #    image_message = bridge.cv2_to_imgmsg(frame, encoding="passthrough")
    #    publisher.publish(image_message)
    #cap.release()


if __name__ == '__main__':
    rospy.init_node("camera_sim")
    img_publisher(sys.argv[1])