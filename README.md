# YOLOv9

## 仓库说明

- yolov9全景分割模型训练仓库
- 包含标签制作 训练 脚本

## 数据集制作

- 相关脚本位于`tools`文件夹下
- 作物架mask掩玛区域不包括实例分割掩玛区域

1. 作物架数据标注
   1. 标注作物架标签为mask格式，参考：http://192.168.4.117:8080/projects/149/data 
2. 数据集制作
   1. 在label-studio项目中分别导出`JSON`、`YOLO` 、 `Brush labels to PNG`三种格式标签
   2. 运行`change_class.py`将实例分割标签class进行转换
   3. 运行`transfom_semantics_label.py`脚本将label-studio导出的png语义标签、img、实例标签提取为正确格式
   4. 运行`add_bg.py`脚本将将语义标签转换为txt格式并添加背景类
   5. 运行`morph_maasks.py`脚本可以将语义分割标签膨胀腐蚀（可选）
   6. 运行`show_stuff.py`脚本可以将语义分割标签可视化出来（可选）
   7. 运行`split_datasets.py`脚本将数据集划分为 train / val 
   8. 最终数据集组成：
    ```bash
    rack_datasets_v1_coco/
         images/
             train/
                 xxx.jpg
             val/
                 xx.jpg
         labels/
             train/
                 xxx.txt
             val/
                 xx.txt
         stuff
             train/
                 xxx.txt
             val/
                 xx.txt
    ```

## 训练说明

- 训练配置参数路径为`data/rack-v4_0.yaml`（`path`代表数据集root路径、`train`代表相对于`path`的相对路径、`names`代表实例分割标签对应类别、`stuff_names`代表全景掩玛类别（当前为作物架与其他））
- 训练脚本为`panoptic/train_ours.py`（需修改该脚本中的预训练模型路径`weights`、训练配置yaml文件路径`cfg`、保存文件名`name`）

## 推理说明

- 普通推理+可视化脚本为 `panoptic/predict_ours.py`

## 其他说明

### 类别代码中固定

- 因为当前实例分割为0-6的七分类，所以作物架类别ID为7
- 位于`utils/coco_utils.py`代码中:
```python
strawberry_instances_ids = [
    1, 2, 3, 4, 5, 6, 7]

strawberry_stuff_ids = [
    8, 0]
```
