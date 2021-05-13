from scipy.io import loadmat
import cv2
import rospy

# Load files
# annots = loadmat("preds.mat")
# img = cv2.imread("mpii/images/086310343.jpg")

# Draw each point for the prediction
def draw_point(annots, img):
    for point in annots["preds"][6]:
        print(point)
        x = point[0]        # (1280 * point[0]) / norm_value_x
        y = point[1]        # (720 * point[1]) / norm_value_y
        img = cv2.circle(img, (int(x), int(y)), radius=6, color=(0, 0, 255), thickness=6)

    # Show image
    cv2.imshow("Output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
