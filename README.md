# yolo_web
把yolov5模型部署到web端，简单的应用  
yolo版本是yolov5 6.0，请不要随便替换里面的yolov5源代码，只替换你的模型即可
### 环境配置
```
pip intsall -r requirements.txt
```
### 使用方法
把训练好的pt模型放在根目录下，或者自己能找到的目录，反正在app.py里面调  

修改index.html和base.html中自己要识别的东西  

在程序根目录下设置app.py的端口和地址  
如果是在远程服务器上使用，IP请设置为0.0.0.0  
端口（Port）请随意  
打开终端或者命令行，如果是Windows直接运行app.py即可  
Linux用户合理复制下面的命令  
```
cd yolo-web-main
python app.py
```
根据提示点击地址进入网页即可  

所有运行出来的东西都会保存在根目录static文件夹下，注意存储的及时清理
