"""  THIS IS NOT A ROS-NODE AND SHOULD BE RAN INDEPENDENTLY WITH PYTHON2 WHEN THE POSE DETECTION SERVER IS RUNNING """
import json
import websocket
import threading
import time
import posix_ipc
import mmap
import numpy as np
import cv2


# Option to load already generated features if training classifier
# Don't save to file if empty
settings = {
    "algorithms" : ["actions"],
    "auxiliaryDB" : "generated.lmdb",
    # Can override: densepose, UV_Textures
    "override" : [],
    "saveToFile" : True,
    "loadedLimitMB" : 3000
    # TODO: add model file names to save to
}
# SaveToFile doesn't work atm

ws = None
ws_thread = None

shared_mem = posix_ipc.SharedMemory("/densepose_debug_mem", flags = posix_ipc.O_CREAT, size = 1500*1500*3)
debugMapfile = mmap.mmap(shared_mem.fd, shared_mem.size)
shared_mem.close_fd()

def websocket_init():
    # Websocket communcation client

    def on_open(ws):
        print("Densepose websocket connected.")
        print("Sending settings: "+json.dumps(settings, indent=4))
        message = json.dumps({"type":"sendTrainSettings", "settings": settings})
        ws.send(message)

    def on_message(ws, message):
        # print("On message: "+message)
        message = json.loads(message)
        if message["type"] == "trained":
            print(message["statistics"])

    def on_error(ws, err):
        print("Websocket error: "+str(err))

    def on_close(ws):
        print("Websocket closed")

    ws = websocket.WebSocketApp("ws://localhost:2043",
        keep_running=True,
        on_open = on_open,
        on_message = on_message,
        on_error = on_error,
        on_close = on_close) 

    def socket_wrapper():
        websocket.enableTrace(False)
        ws.run_forever()

    ws_thread = threading.Thread(target=socket_wrapper)
    ws_thread.daemon = True
    ws_thread.start()



websocket_init()

while True:
    time.sleep(2)