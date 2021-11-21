# Person Detection and Tracking (Python)

### Install requirements
```bash
~$ pip install --upgrade -r requirements.txt
```

### Get trained weights 
* download [resnet50_person_reid_128x64.pth](https://drive.google.com/file/d/16RmXYOx06tIYfiAF9ej2iFn8SXKyCgbF/view?usp=sharing) and locate it in ```./ReID_data/pretrained_models```.
* download [resnet50_200.pth](https://drive.google.com/file/d/1VRTXhsXInUIRz1EUlCOEuDAxDtPi30Nh/view?usp=sharing) and locate it in ```./ReID_data/model_weights```. 

<br/>

### 1. Pedestrain sample capture 
```bash 
~$ python capture_user_samples.py
```
* Only ```one person``` should be captured during this process.
* If more than one person is captured, it is recommended to retake.
* You can change the target person ID by editing ```FILE_NAME ``` in ```cfgs/capture_cfg.yaml```
    * don't use [underscore](https://en.wikipedia.org/wiki/Underscore) ('_') for naming ID and use capital letters.
* Set ```PID``` without duplication (i.e., uniquely with regard to each person)
* Set ```TYPE``` in what you want, but following the format(e.g., ```t0``` as type0, ```t1``` as type1)
    * (ex) if you want to capture the same person with different clothes, distinguish them with ```TYPE``` values.
        * (NOTICE) Keep in mind that you should set the same ```PID``` in this case.
* You can tune the number of pictures for capturing with ```NUM_IMG```.         
* In ```cfgs/reid_cfg.yaml```, you can tune the number of classes by setting ```CLASS_NUM```. 
    * (ex) if you set ```CLASS_NUM``` := 256, 
        * it means you can settle ```PID``` values in the range [0, 255]
        * or you can register person IDs in max 256 numbers


### 2. ID enrollment 
```bash 
~$ python enrollment_persons.py
``` 


### 3. Pedestrain recognition 
```bash 
~$ python run_person_detector.py     # the main running code 
~$ python reid_test.py               # for checking reid module 
```



***
## Reference 
[1] [yolov4-tiny-tflite-for-person-detection](https://github.com/DoranLyong/yolov4-tiny-tflite-for-person-detection) / pretrained person detector <br/>
[2] [person-reid-tiny-baseline](https://github.com/DoranLyong/person-reid-tiny-baseline) / person reid baseline code <br/>
[3] [Python Faiss 사용법 간단 정리, 블로그](https://lsjsj92.tistory.com/605) / Facebook AI의 벡터 유사도 구하는 라이브러리 <br/>

