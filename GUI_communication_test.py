#Devolped by Dylan to test the GUI and python connection 
#Socket was designed using this tutorial 
#https://realpython.com/python-sockets/

#imports
import socket, time, random

#change the test conditions
mode = 1
# change the number of 0 and 1 to change the bais of the random number 
# equal is 50/50 
bias = [0,1] 

#Socket setup - needs to be the same on matlab 
HOST = "127.0.0.1" #loopback socket
PORT = 65432  #port 

if mode == 0: #preset data for video
    #data 
    #data = [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0]
    data = [1,1,1]
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        for num in data:
            if num == 1:
                s.sendall(b"1")
            else:
                s.sendall(b"0")
            time.sleep(1)
elif mode == 1: #random data with bias continous
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        for t in range(1000):
            num = random.sample(bias,1)
            print(num)
            if num[0] == 1:
                s.sendall(b"1")
            else:
                s.sendall(b"0")
            time.sleep(1)

#s.close()
