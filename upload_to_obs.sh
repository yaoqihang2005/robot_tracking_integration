#!/bin/bash
# 一键打包并上传到 OBS
# 使用方法: ./upload_to_obs.sh <数据集目录名> <OBS路径>

# ==================== 配置 ====================
# 默认配置（根据你的实际情况修改）
DEFAULT_DATA_DIR="simple_sorting_0409_filtered_v4"
DEFAULT_OBS_PATH="obs://sai.liyl/lihong/"

# ==================== 检查参数 ====================
DATA_DIR=${1:-$DEFAULT_DATA_DIR}
OBS_PATH=${2:-$DEFAULT_OBS_PATH}

echo "=========================================="
echo "🚀 开始打包并上传到 OBS"
echo "=========================================="
echo "📁 数据集目录: $DATA_DIR"
echo "☁️  OBS 路径: $OBS_PATH"
echo ""

# ==================== 检查数据集是否存在 ====================
if [ ! -d "data/$DATA_DIR" ]; then
    echo "❌ 错误: 找不到数据集目录 data/$DATA_DIR"
    echo "请确保数据集在 data/ 目录下"
    exit 1
fi

# ==================== 打包 ====================
echo "📦 正在打包..."
cd data || exit 1
TAR_FILE="${DATA_DIR}.tar.gz"

if [ -f "$TAR_FILE" ]; then
    echo "⚠️  打包文件已存在，删除旧文件..."
    rm -f "$TAR_FILE"
fi

tar -czf "$TAR_FILE" "$DATA_DIR/"

if [ $? -eq 0 ]; then
    echo "✅ 打包成功: $TAR_FILE"
    echo "📊 文件大小: $(du -h "$TAR_FILE" | cut -f1)"
else
    echo "❌ 打包失败"
    exit 1
fi

echo ""

# ==================== 上传到 OBS ====================
echo "☁️  正在上传到 OBS..."

if command -v obsutil &> /dev/null; then
    obsutil cp "$TAR_FILE" "$OBS_PATH"
    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✅ 上传成功！"
        echo "📍 OBS 路径: $OBS_PATH$TAR_FILE"
        echo "=========================================="
    else
        echo "❌ 上传失败"
        exit 1
    fi
else
    echo ""
    echo "=========================================="
    echo "⚠️  未找到 obsutil 命令"
    echo "=========================================="
    echo ""
    echo "请先安装并配置 obsutil："
    echo "1. 下载 obsutil: https://support.huaweicloud.com/utiltg-obs/obs_11_0001.html"
    echo "2. 配置访问密钥"
    echo ""
    echo "或者手动上传打包文件："
    echo "📦 打包文件位置: data/$TAR_FILE"
fi
