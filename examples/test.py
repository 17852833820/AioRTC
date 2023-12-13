import cv2

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取视频帧
    ret, frame = cap.read()

    # 检查帧是否成功读取
    if not ret:
        print("Failed to capture frame")
        break

    # 显示图像窗口
    cv2.imshow("Video", frame)

    # 检测键盘输入，如果按下 'q' 键则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和关闭 OpenCV 窗口
cap.release()
cv2.destroyAllWindows()
