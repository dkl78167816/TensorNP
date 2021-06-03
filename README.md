# 毕业设计——TensorNP：一个轻量级张量分解算法库

![scut](./doc/scut.ico)

## 特点

1. numpy、pytorch多计算后端
2. 完善的基准测试
3. pythonic
4. 六种分解算法

## 使用

```python
import tensornp as tnp

tensor = tnp.randn(2, 3, 4)
factors, core = tnp.hosvd(tensor)
```

## 引用

