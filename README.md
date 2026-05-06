本项目使用MNN替换paddle inference、paddle lite进行paddleocr边缘端部署，具有较好推理速度
1、opencv使用4.6.0版本，mnn使用3.3版本
2、opencv中contrib需要进行编译，否则在图片上就是中文乱码
3、训练、转化参考 https://blog.csdn.net/qq_23123181/article/details/159650029?spm=1001.2014.3001.5501
