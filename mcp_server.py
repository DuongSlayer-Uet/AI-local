#!/usr/bin/env python3
"""
MCP server chay tren Raspberry Pi 4: expose cac tool lien quan phan cung cho
AI-server (RTX5060) goi qua MCP.

Them tool moi sau nay: chi can viet 1 ham moi va gan @mcp.tool(), khong can
sua gi o AI-server (server.py) ca - server.py tu lay danh sach tool luc
khoi dong.

Cai dat:
    pip install "mcp[cli]"

Chay:
    python3 mcp_server.py
"""
import subprocess

from mcp.server.fastmcp import FastMCP

# ============================================================
# CAU HINH - sua o day
# ============================================================
HOST = "0.0.0.0"
PORT = 8765

mcp = FastMCP("raspi-tools", host=HOST, port=PORT)


@mcp.tool()
def measure_temp() -> str:
    """Do nhiet do CPU hien tai cua Raspberry Pi (don vi C)."""
    result = subprocess.run(
        ["vcgencmd", "measure_temp"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()  # vd: "temp=48.3'C"


@mcp.tool()
def get_current_time() -> str:
    """Kiem tra gio hien tai (gio:phut) cua Raspberry Pi."""
    result = subprocess.run(
        ["date", "+%H:%M"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()  # vd: "10:32"


# ------------------------------------------------------------
# Them tool moi o day, vi du:
#
# @mcp.tool()
# def tool_moi(param: str) -> str:
#     """Mo ta tool de model biet khi nao nen goi."""
#     ...
# ------------------------------------------------------------


if __name__ == "__main__":
    mcp.run(transport="sse")
