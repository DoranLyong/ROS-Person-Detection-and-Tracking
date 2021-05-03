# Person Detection and Tracking (Python) - onging 

### Install requirements
```bash
~$ pip install --upgrade requirements.txt
```


### 1. Pedestrain sample capture 
```bash 
~$ python capture_user_samples.py
```
* Only ```one person``` should be captured during this process.
* If more than one person is captured, it is recommended to retake.
* You can change the target person ID by editing ```FILE_NAME ``` in ```cfgs/capture_cfg.yaml```
    * don't use [underscore](https://en.wikipedia.org/wiki/Underscore) ('_') for naming ID

<br/>

### 2. ID enrollment 
```bash 
~$ enrollment_persons.py
``` 



### 3. Pedestrain recognition 



***
## Reference 
[1] [yolov4-tiny-tflite-for-person-detection](https://github.com/DoranLyong/yolov4-tiny-tflite-for-person-detection) / pretrained person detector <br/>
[2] [person-reid-tiny-baseline](https://github.com/DoranLyong/person-reid-tiny-baseline) / person reid baseline code <br/>
[3] [Python Faiss 사용법 간단 정리, 블로그](https://lsjsj92.tistory.com/605) / Facebook AI의 벡터 유사도 구하는 라이브러리 <br/>

