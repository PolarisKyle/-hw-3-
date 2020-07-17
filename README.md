# -hw-3-
hw_3_食物分类
有三个文件夹——testing，training，validation。均为食物照片。

共有11类，Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, and Vegetable/Fruit.我们要创建一个CNN，用来实现食物的分类。

label为数字0~10别对应以上11种食物类别。training与validation中的食物照片名称第一个数字即为其类别（“_”前的数字）。Testing中的照片名称不包含其类别。

对training进行训练，再对validation进行分类与正确结果进行比较。最后对testing进行分类。

本程序只对trian set进行训练，然后在用得到的，模型对validation set进行分类。

train set的正确率大致为0.9，validation set 的正确率大致在0.5左右。有待改进。并未对testing进行分类。

欢迎大家批评指正。
由于食物照片太多，一共一万多张照片，1.1g左右。照片就不上传了。大家从下方百度云链接下载后，放入data文件夹即可。

链接：https://pan.baidu.com/s/1aCsicRCglfLF-JikL8dN5g 
提取码：pkwq 
复制这段内容后打开百度网盘手机App，操作更方便哦
