from pynput import mouse
from threading import Thread
import time

#左クリックをした位置と離した位置を記録
class MouseMonitor:
    def __init__(self):
        self.coord = [-1, -1, -1, -1]
        self.released = False
        pass

    def coordinate(self):
        print("x1: {0}".format(self.coord[0]))
        print("x2: {0}".format(self.coord[1]))
        print("y1: {0}".format(self.coord[2]))
        print("y2: {0}".format(self.coord[3]))
        return self.coord[0], self.coord[1], self.coord[2], self.coord[3]

    def on_click(self, x, y, button, pressed):
        if pressed:
            self.coord[0] = int(x)
            self.coord[2] = int(y)
            self.released = False
        else:
            self.coord[1] = int(x)
            self.coord[3] = int(y)
            self.released = True

    def start(self):
        with mouse.Listener(
                on_click=self.on_click) as self.listener:
            self.listener.join()


"""mou=MouseMonitor() #マウスイベントを検出
mou_th=Thread(target=mou.start,args=())
mou_th.start()
while True:
    if mou.released:
        print((mou.coordinate()))
    time.sleep(0.1)"""