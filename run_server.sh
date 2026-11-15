#!/usr/bin/env sh
# Chạy chatbot web NeoRAG (app/server.py):
#   ./run_server.sh [cổng]        chỉ máy này mở được (127.0.0.1)
#   ./run_server.sh lan [cổng]    cho máy khác cùng mạng LAN truy cập (0.0.0.0)
cd "$(dirname "$0")" || exit 1

HOST=127.0.0.1
if [ "$1" = "lan" ]; then
    HOST=0.0.0.0
    shift
fi
PORT="${1:-8000}"

if [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then   # Git Bash trên Windows
    PY=.venv/Scripts/python.exe
else
    PY=python3
fi

if [ ! -f .env ]; then
    echo "[cảnh báo] Chưa có file .env: chỉ trả lời được các câu đã có trong cache."
    echo "           Tạo bằng: cp .env.example .env  rồi điền OPENAI_API_KEY."
fi

if [ "$HOST" = "0.0.0.0" ]; then
    echo "Chế độ LAN: máy khác trong cùng mạng mở http://<IP-máy-này>:$PORT"
    (hostname -I 2>/dev/null || ipconfig getifaddr en0 2>/dev/null) | tr ' ' '\n' \
        | grep -E '^[0-9]+\.' | sed "s|.*|  http://&:$PORT|"
fi

URL="http://localhost:$PORT"
echo "Đang mở $URL  (Ctrl+C để dừng)"
# Đợi server mở cổng rồi mới mở trình duyệt (NO_BROWSER=1 để bỏ qua)
[ -z "$NO_BROWSER" ] && ( sleep 2; (command -v xdg-open >/dev/null && xdg-open "$URL") \
    || (command -v open >/dev/null && open "$URL") ) >/dev/null 2>&1 &

exec "$PY" app/server.py --host "$HOST" --port "$PORT"
