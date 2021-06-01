# Person Detection and Tracking (ROS service node) - onging 

### Install requirements
* check [here](https://github.com/DoranLyong/ROS-Person-Detection-and-Tracking/tree/main/python)

### 1. Pedestrain sample capture 

<br/>

### 2. ID enrollment 



### 3. Pedestrain recognition 
```bash 
~$ rosrun emergency_detection_management person_detection_and_tracking.py
```

* This node communicates with ROS service. 
* If you want to check a simple tutorial for ROS service, check ```add_two_ints_client.py``` and ```add_two_ints_server.py```
* ```img_client.py``` and ```img_server.py``` are just for a simple test to implement image transfer with ROS service. - You can ignore it. 
* I defined my own message types to transfer a image sequence. Check ```srv``` directory in the project node. 

<br/>

### Example Usage 
1. Run two service node: 
``` bash 
rosrun emergency_detection_management person_detection_and_tracking.py

rosrun emergency_detection_management img_server.py
```

2. Finally, request the service  running by:
```bash 
rosrun emergency_detection_management test_percep_client.py
```

* ```test_percep_client.py``` request ```person_detection_and_tracking.py``` node detect and track your target person. 
* If there is your target in the camera scene, ```person_detection_and_tracking.py``` captures and stacks an image sequence. 
* This image sequence is transferred to ```img_server.py```, and some processes can be added in ```img_server.py``` node if you want.
    * ```img_server.py``` can consist of detecting human activities of your target only.
    * This function can be changed depending on your application domain.


***
## Reference 
[1] [yolov4-tiny-tflite-for-person-detection](https://github.com/DoranLyong/yolov4-tiny-tflite-for-person-detection) / pretrained person detector <br/>
[2] [person-reid-tiny-baseline](https://github.com/DoranLyong/person-reid-tiny-baseline) / person reid baseline code <br/>
[3] [Python Faiss 사용법 간단 정리, 블로그](https://lsjsj92.tistory.com/605) / Facebook AI의 벡터 유사도 구하는 라이브러리 <br/>

