import numpy as np

# 官方 Demo 数据路径
data_path = "assets/example1/snowboard.npz"

try:
    data = np.load(data_path, allow_pickle=True)
    print(f"\n{'='*50}")
    print(f"解析成功: {data_path}")
    print(f"{'='*50}\n")

    for key in data.files:
        content = data[key]
        
        # 处理可能被 pickling 过的对象（如字典）
        if content.dtype == 'O':
            item = content.item()
            if isinstance(item, dict):
                print(f"🔑 Key: {key} (Dictionary)")
                for sub_key, sub_val in item.items():
                    if hasattr(sub_val, 'shape'):
                        print(f"   └── {sub_key:15} | Shape: {str(sub_val.shape):15} | Dtype: {sub_val.dtype}")
                    else:
                        print(f"   └── {sub_key:15} | Value: {sub_val}")
            else:
                print(f"🔑 Key: {key} | Type: {type(item)}")
        else:
            # 普通数组
            shape_str = str(content.shape)
            print(f"🔑 Key: {key:15} | Shape: {shape_str:15} | Dtype: {content.dtype}")

    print(f"\n{'='*50}")

except Exception as e:
    print(f"解析失败: {e}")