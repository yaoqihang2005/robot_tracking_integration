
## ☁️ OBS 使用注意事项

### RHOS 环境配置

在 RHOS 实验室服务器上，`obsutil` 已安装在 `/usr/local/bin/obsutil`。

由于 home 目录是只读文件系统，**必须修改配置文件或使用命令行参数**：

### 方法 1：修改配置文件（推荐，一次修改永久有效）

编辑 `~/.obsutilconfig`，修改以下配置项：

```ini
sdkLogPath=/data/lihong-project/qihang/projects/robot_tracking_integration/.obsutil_log/obssdk.log
utilLogPath=/data/lihong-project/qihang/projects/robot_tracking_integration/.obsutil_log/obsutil.log
defaultTempFileDir=/data/lihong-project/qihang/projects/robot_tracking_integration
```

### 方法 2：使用命令行参数（每次都要加）

下载文件时必须加上 `-cpd` 参数指定检查点目录：

```bash
obsutil cp obs://sai.liyl/lihong/文件名 data/ -cpd ./.obsutil_checkpoint
```

### 上传文件到 OBS

同样需要注意检查点目录问题：

```bash
cd data
tar -czf 你的文件名.tar.gz 你的目录/
obsutil cp 你的文件名.tar.gz obs://sai.liyl/lihong/ -cpd ./.obsutil_checkpoint
```

### OBS 上现有文件列表

| 文件名 | 大小 |
|--------|------|
| `510_erase_board_350_lerobot.zip` | 2.49GB |
| `512_stiring_vision_only_50000.tar.gz` | 11.59GB |
| `513_screw_350_vision_30000.tar.gz` | 11.59GB |
| `513_screw_lerobot_490.zip` | 3.35GB |
| `513_screw_vision_90000.tar.gz` | 11.59GB |
| `514_stiring_350_vision_30000.tar.gz` | 11.59GB |

### 查看 OBS 文件

```bash
obsutil ls obs://sai.liyl/lihong/
```
