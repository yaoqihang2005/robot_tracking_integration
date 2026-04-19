import os
import glob
import cv2
import numpy as np
import argparse
import http.server
import socketserver
import json
import webbrowser
from threading import Thread
from main_pipeline import main_pipeline

import base64
import subprocess

def kill_process_on_port(port):
    """强制关闭占用指定端口的进程"""
    import signal
    try:
        # 1. 尝试使用 lsof 找到 PID
        output = subprocess.check_output(["lsof", "-t", f"-i:{port}"], stderr=subprocess.DEVNULL)
        pids = output.decode().strip().split('\n')
        for pid in pids:
            if pid:
                pid = int(pid)
                print(f"⚠️ 发现端口 {port} 被进程 {pid} 占用，正在清理...")
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    except Exception:
        # 2. 如果 lsof 失败，尝试使用 fuser (部分系统可能需要)
        try:
            subprocess.run(["fuser", "-k", f"{port}/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    
    # 给系统一点时间释放端口
    import time
    time.sleep(0.5)

# 极简 Web 服务器，用于处理鼠标点选坐标
class ROIHandler(http.server.SimpleHTTPRequestHandler):
    selected_roi = None
    selected_points = []
    selected_labels = []
    last_mask = None
    html_content = ""
    sam_helper = None
    video_frames_dir = None
    original_frame = None

    def do_GET(self):
        # ... (保持不变)
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(ROIHandler.html_content.encode('utf-8'))
        elif self.path == '/frame.jpg':
            if os.path.exists("frame.jpg"):
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')
                self.end_headers()
                with open("frame.jpg", "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Image not found")
        else:
            super().do_GET()
    
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data)
        
        if self.path == '/click':
            # 处理点击事件
            point = data['point'] # [x, y]
            label = data['label'] # 1 or 0
            ROIHandler.selected_points.append(point)
            ROIHandler.selected_labels.append(label)
            
            # 生成 Mask 预览
            mask = ROIHandler.sam_helper.get_mask_from_points(
                ROIHandler.video_frames_dir, 
                ROIHandler.selected_points, 
                ROIHandler.selected_labels
            )
            ROIHandler.last_mask = mask # 记录最新的 Mask
            
            # 将 Mask 叠加到原图上
            overlay = ROIHandler.original_frame.copy()
            overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5
            
            # 绘制已有点
            for p, l in zip(ROIHandler.selected_points, ROIHandler.selected_labels):
                color = (0, 255, 0) if l == 1 else (0, 0, 255)
                cv2.circle(overlay, (int(p[0]), int(p[1])), 5, color, -1)

            _, buffer = cv2.imencode('.jpg', overlay)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"mask_img": img_base64}).encode('utf-8'))

        elif self.path == '/confirm':
            # 确认最终结果
            ROIHandler.selected_roi = {
                "points": ROIHandler.selected_points,
                "labels": ROIHandler.selected_labels,
                "mask": ROIHandler.last_mask # 返回 Mask 数组
            }
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        
        elif self.path == '/reset':
            # 重置点
            ROIHandler.selected_points = []
            ROIHandler.selected_labels = []
            ROIHandler.last_mask = None
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

def start_web_selector(image_path, video_frames_dir=None):
    """启动一个简单的网页供用户点选物体"""
    from core.sam_helper import SAM2Helper
    if ROIHandler.sam_helper is None:
        ROIHandler.sam_helper = SAM2Helper()
    
    ROIHandler.video_frames_dir = video_frames_dir
    ROIHandler.original_frame = cv2.imread(image_path)
    ROIHandler.selected_points = []
    ROIHandler.selected_labels = []

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SAM 2 Interactive Selection</title>
        <style>
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                text-align: center; 
                background: #f0f0f0; 
                margin: 0;
                padding: 20px; 
                display: flex;
                flex-direction: column;
                align-items: center;
            }}
            .container {{ 
                background: white; 
                padding: 20px; 
                border-radius: 8px; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
                display: flex;
                flex-direction: column;
                align-items: center;
                /* 锁定最小宽度，防止内容变化导致抖动 */
                min-width: 600px; 
            }}
            .canvas-wrapper {{
                position: relative;
                margin: 10px 0;
                /* 关键：锁定宽高比或使用固定尺寸 */
                border: 2px solid #333;
                line-height: 0;
            }}
            canvas {{ 
                cursor: crosshair; 
                display: block;
            }}
            .controls {{ margin-top: 20px; }}
            button {{ padding: 10px 20px; font-size: 16px; margin: 0 10px; cursor: pointer; border: none; border-radius: 4px; transition: background 0.3s; }}
            #confirm {{ background: #28a745; color: white; }}
            #confirm:hover {{ background: #218838; }}
            #reset {{ background: #dc3545; color: white; }}
            #reset:hover {{ background: #c82333; }}
            .hint {{ color: #666; margin-bottom: 10px; font-size: 14px; }}
            .status {{ 
                margin-top: 10px; 
                font-weight: bold; 
                color: #007bff; 
                height: 20px; /* 固定高度防止撑开容器 */
            }}
        </style>
    </head>
    <body>
        <div class="container" id="main-container">
            <h2>SAM 2 交互式点选</h2>
            <div class="hint">左键点击：正样本 (绿色) | 右键点击：负样本 (红色)</div>
            <div class="canvas-wrapper">
                <canvas id="canvas"></canvas>
            </div>
            <div id="status" class="status">准备就绪，请开始点选...</div>
            <div class="controls">
                <button id="reset">重置所有点</button>
                <button id="confirm">确认并开始全视频追踪</button>
            </div>
        </div>
        <script>
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            const status = document.getElementById('status');
            const container = document.getElementById('main-container');
            const img = new Image();
            img.src = '/frame.jpg';

            img.onload = () => {{
                // 锁定画布和容器尺寸，防止后续更新导致漂移
                canvas.width = img.width;
                canvas.height = img.height;
                container.style.width = (img.width + 40) + 'px'; 
                ctx.drawImage(img, 0, 0);
            }};

            canvas.onmousedown = (e) => {{
                e.preventDefault();
                // 此时 rect.width 应该等于 canvas.width，防止缩放漂移
                const rect = canvas.getBoundingClientRect();
                const x = (e.clientX - rect.left) * (canvas.width / rect.width);
                const y = (e.clientY - rect.top) * (canvas.height / rect.height);
                const label = e.button === 0 ? 1 : 0; 

                status.innerText = "正在计算 Mask...";
                fetch('/click', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ point: [x, y], label: label }})
                }})
                .then(res => res.json())
                .then(data => {{
                    const maskImg = new Image();
                    maskImg.src = 'data:image/jpeg;base64,' + data.mask_img;
                    maskImg.onload = () => {{
                        // 清除旧图并绘制新图（新图已包含 Mask 和点）
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.drawImage(maskImg, 0, 0);
                        status.innerText = "Mask 已更新";
                    }};
                }});
            }};

            // 禁用右键菜单
            canvas.oncontextmenu = (e) => e.preventDefault();

            document.getElementById('reset').onclick = () => {{
                fetch('/reset', {{ method: 'POST' }})
                .then(() => {{
                    ctx.drawImage(img, 0, 0);
                    status.innerText = "已重置";
                }});
            }};

            document.getElementById('confirm').onclick = () => {{
                status.innerText = "正在提交...";
                fetch('/confirm', {{ method: 'POST', body: JSON.stringify({{}}) }})
                .then(() => {{ 
                    alert("分割已确认，请返回终端查看追踪进度。");
                    window.close(); 
                }});
            }};
        </script>
    </body>
    </html>
    """
    
    with open("selector.html", "w") as f: f.write(html_content)
    ROIHandler.html_content = html_content
    
    PORT = 8080
    ROIHandler.selected_roi = None
    
    # 每次启动前强制清理端口占用 (由于权限限制，如果杀不掉则尝试寻找下一个可用端口)
    kill_process_on_port(PORT)
    
    # 临时重命名图片方便 Web 加载
    if os.path.exists("frame.jpg"): os.remove("frame.jpg")
    os.link(image_path, "frame.jpg")

    # 显式监听所有网卡 (0.0.0.0)
    server_started = False
    max_tries = 10
    current_port = PORT
    
    while not server_started and current_port < PORT + max_tries:
        try:
            # 允许端口复用
            socketserver.TCPServer.allow_reuse_address = True
            with socketserver.TCPServer(("0.0.0.0", current_port), ROIHandler) as httpd:
                server_started = True
                print(f"\n" + "="*60)
                print(f">>> Web 选框服务已启动！")
                print(f">>> 1. 如果你是 VS Code 远程，请在下方“端口”面板手动添加 {current_port} 端口转发。")
                print(f">>> 2. 然后在本地浏览器访问: http://127.0.0.1:{current_port}")
                print(f">>> 3. 或者尝试公网访问: http://202.120.47.58:{current_port} (需防火墙开启)")
                print("="*60)
                
                # 只处理一个请求（即提交坐标的 POST 请求）
                while ROIHandler.selected_roi is None:
                    httpd.handle_request()
        except OSError as e:
            if e.errno == 98: # Address already in use
                print(f"⚠️ 端口 {current_port} 仍被占用，尝试下一个端口...")
                current_port += 1
            else:
                print(f"❌ 启动服务失败: {e}")
                return None
        except Exception as e:
            print(f"❌ 启动服务发生意外错误: {e}")
            return None
            
    if not server_started:
        print(f"❌ 尝试了 {max_tries} 个端口均失败，请手动清理进程。")
        return None
            
    return ROIHandler.selected_roi

def get_box_interactively(video_path):
    """通过 Web 界面让用户点选初始框"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret: return None
    
    # 为了交互性能，我们将第一帧缩放到 512p
    h, w = frame.shape[:2]
    scale = 512.0 / max(h, w)
    resized_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    
    temp_img = "data/first_frame_temp.jpg"
    if not os.path.exists("data"): os.makedirs("data")
    cv2.imwrite(temp_img, resized_frame)
    
    # SAM 2 需要一个文件夹
    frames_dir = "data/temp_frames_interactive"
    if os.path.exists(frames_dir):
        import shutil
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)
    cv2.imwrite(os.path.join(frames_dir, "00000.jpg"), resized_frame)
    
    # 返回的是原图分辨率下的坐标，所以我们需要记录 scale
    selection = start_web_selector(temp_img, video_frames_dir=frames_dir)
    
    if selection is None: return None
    
    # 将坐标还原回原图分辨率
    if isinstance(selection, dict):
        selection['points'] = [[p[0] / scale, p[1] / scale] for p in selection['points']]
    else:
        selection = selection / scale
        
    return selection

def run_batch():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True, help="视频所在目录")
    parser.add_argument("--reuse_box", action="store_true", default=True, help="是否复用第一个视频的框")
    args = parser.parse_args()

    # 递归查找所有 mp4 (排除 tactile 视频，因为我们要追踪的是视觉物体)
    video_files = glob.glob(os.path.join(args.video_dir, "**/*.mp4"), recursive=True)
    # 过滤掉触觉视频 (根据路径名)
    video_files = [v for v in video_files if "images.wrist" in v or "images.top" in v]
    video_files = sorted(video_files)
    
    if not video_files:
        print(f"在 {args.video_dir} 下未找到视频。")
        return

    print(f"找到 {len(video_files)} 条视频，准备开始处理...")
    
    shared_box = None
    
    for i, v_path in enumerate(video_files):
        video_id = os.path.basename(v_path).replace(".mp4", "")
        output_subdir = os.path.join("results", video_id)
        
        # 检查是否已处理过
        if os.path.exists(os.path.join(output_subdir, "quality_scores.npz")):
            print(f"[{i+1}/{len(video_files)}] 跳过已完成: {video_id}")
            continue

        print(f"\n" + "="*60)
        print(f"[{i+1}/{len(video_files)}] 正在处理: {v_path}")
        
        # 获取 Box
        if shared_box is None:
            current_box = get_box_interactively(v_path)
            if current_box is None:
                print("未选框，跳过此视频。")
                continue
            if args.reuse_box:
                shared_box = current_box
        else:
            current_box = shared_box
            
        # 运行流水线
        try:
            main_pipeline(v_path, current_box, output_dir=output_subdir)
            print(f"✅ 处理成功: {video_id}")
        except Exception as e:
            print(f"❌ 处理失败 {video_id}: {e}")

if __name__ == "__main__":
    run_batch()