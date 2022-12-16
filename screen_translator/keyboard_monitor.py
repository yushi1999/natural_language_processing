
from pynput.keyboard import Key, Listener
import threading
import time

# スクリーンショットのキーコマンドが押されたことを検出
class KeyMonitor:
    def __init__(self):
        self.key_list=[]
    
    #指定されたキーが押されているかを返す
    def is_pressed(self,key:Key):
        return key in self.key_list
    
    def on_press(self,key):
        try:
            if key.char not in self.key_list:
                self.key_list.append(key.char)
        except:
            if key not in self.key_list:
                self.key_list.append(key)
        print("key_list: "+str(self.key_list))
            
    def on_release(self,key):
        try:
            if key.char in self.key_list:
                self.key_list.remove(key.char)
        except:
            if key in self.key_list:
                self.key_list.remove(key)
        print("key_list: "+str(self.key_list))
    
    def start(self):
        with Listener(
            on_press=self.on_press,
            on_release=self.on_release) as listener:
            listener.join()
    
"""monitor=KeyMonitor()
th00=threading.Thread(target=monitor.start,args=())
th00.start()

while True:
    if monitor.is_pressed("4") and monitor.is_pressed(Key.cmd):
        print("2 keys are pushed.")
    time.sleep(0.01)"""