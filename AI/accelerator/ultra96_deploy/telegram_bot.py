#!/usr/bin/env python3
#Telegram bot for Ultra96 operations and health checks.

import os
import sys
import json
import time
import subprocess
import urllib.request
import urllib.error

#Configuration
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ALLOWED_USER_IDS = None
POLL_INTERVAL = 2
SCRIPTS_DIR = os.getenv("PICKLEBALL_SCRIPTS_DIR", "/home/xilinx/pickleball")

COMMANDS = {
    "power": {
        "script": "power_management.py",
        "default_args": [],
        "description": "Power management (e.g. /power --status, /power --mode low_power)",
        "sudo": True,
    },
    "eval_sw": {
        "script": "eval_sw.py",
        "default_args": [],
        "description": "Run software model evaluation",
        "sudo": True,
    },
    "eval_hw": {
        "script": "eval_hw.py",
        "default_args": [],
        "description": "Run hardware (FPGA) model evaluation",
        "sudo": True,
    },
}

#Processes to kill during /cleanup.
PYNQ_PROC_PATTERN = "ai_event_generator|eval_hw|eval_sw|predict_fpga|ps_dma_driver|ai_ps_dma_driver"


def telegram_api(method, params=None):
    #Make a Telegram Bot API call using stdlib only.
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    if params:
        data = json.dumps(params).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
    else:
        req = urllib.request.Request(url)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as err:
        print(f"[ERROR] API call failed: {err}", file=sys.stderr)
        return None


def read_sysfs(path):
    #Read a single sysfs file and return stripped content.
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except Exception:
        return None


def read_board_power():
    #Sum V*I across IRPS5401 rails and return total watts plus per-rail data.
    hwmon_base = "/sys/class/hwmon"
    if not os.path.isdir(hwmon_base):
        return {"total_w": -1, "rails": []}

    rails = []
    total_mw = 0

    for hwmon in sorted(os.listdir(hwmon_base)):
        hwmon_path = os.path.join(hwmon_base, hwmon)
        if not os.path.isdir(hwmon_path):
            continue

        name_path = os.path.join(hwmon_path, "name")
        name = read_sysfs(name_path)
        if not name or "irps5401" not in name:
            continue

        idx = 1
        while True:
            v_path = os.path.join(hwmon_path, f"in{idx}_input")
            i_path = os.path.join(hwmon_path, f"curr{idx}_input")
            if not os.path.exists(v_path) or not os.path.exists(i_path):
                break
            try:
                v_mv = int(read_sysfs(v_path))
                i_ma = int(read_sysfs(i_path))
                p_mw = v_mv * i_ma / 1000.0

                label_path = os.path.join(hwmon_path, f"in{idx}_label")
                label = read_sysfs(label_path) or f"rail{idx}"

                rails.append(
                    {
                        "label": label,
                        "v_mv": v_mv,
                        "i_ma": i_ma,
                        "p_mw": p_mw,
                    }
                )
                total_mw += p_mw
            except (ValueError, TypeError):
                pass
            idx += 1

    return {"total_w": total_mw / 1000.0, "rails": rails}


def get_system_info():
    #Gather board status details used by /ping.
    info = {}

    try:
        with open("/proc/uptime", "r", encoding="utf-8") as handle:
            secs = float(handle.read().split()[0])
            days, rem = divmod(int(secs), 86400)
            hours, rem = divmod(rem, 3600)
            mins, _ = divmod(rem, 60)
            info["uptime"] = f"{days}d {hours}h {mins}m"
    except Exception:
        info["uptime"] = "unknown"

    for temp_path in [
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/hwmon/hwmon0/temp1_input",
    ]:
        try:
            with open(temp_path, "r", encoding="utf-8") as handle:
                temp_mc = int(handle.read().strip())
                info["cpu_temp"] = f"{temp_mc / 1000:.1f} C"
                break
        except Exception:
            continue
    if "cpu_temp" not in info:
        info["cpu_temp"] = "unknown"

    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            meminfo = {}
            for line in handle:
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(":")] = int(parts[1])
            total = meminfo.get("MemTotal", 0)
            avail = meminfo.get("MemAvailable", 0)
            used = total - avail
            if total > 0:
                info["memory"] = (
                    f"{used // 1024} MB / {total // 1024} MB "
                    f"({100 * used // total}%)"
                )
            else:
                info["memory"] = "unknown"
    except Exception:
        info["memory"] = "unknown"

    try:
        result = subprocess.run(
            ["df", "-h", "/"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) >= 2:
            parts = lines[1].split()
            info["disk"] = f"{parts[2]} / {parts[1]} ({parts[4]})"
        else:
            info["disk"] = "unknown"
    except Exception:
        info["disk"] = "unknown"

    try:
        info["hostname"] = subprocess.run(
            ["hostname"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
    except Exception:
        info["hostname"] = "unknown"

    info["power"] = read_board_power()
    return info


def format_status():
    #Format the status response payload for /ping.
    info = get_system_info()
    power = info["power"]

    lines = [
        "FPGA Board Status: ONLINE",
        "",
        f"  Hostname:  {info['hostname']}",
        f"  Uptime:    {info['uptime']}",
        f"  CPU Temp:  {info['cpu_temp']}",
        f"  Memory:    {info['memory']}",
        f"  Disk:      {info['disk']}",
    ]

    if power["total_w"] >= 0 and power["rails"]:
        lines.append("")
        lines.append("  Power Rails:")
        for rail in power["rails"]:
            lines.append(
                f"    {rail['label']:<12} "
                f"{rail['v_mv']:>5} mV x {rail['i_ma']:>5} mA "
                f"= {rail['p_mw']:>7.0f} mW"
            )
        lines.append(f"    {'_' * 44}")
        lines.append(f"    Total: {power['total_w']:.2f} W")
    else:
        lines.append("")
        lines.append("  Power:     unavailable (no IRPS5401 found)")

    lines.append("")
    lines.append(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    return "\n".join(lines)


def run_script(cmd_name, extra_args=None):
    #Run whitelisted scripts and return combined output.
    cmd = COMMANDS[cmd_name]
    script_path = os.path.join(SCRIPTS_DIR, cmd["script"])

    if not os.path.exists(script_path):
        return f"ERROR: {cmd['script']} not found in {SCRIPTS_DIR}"

    args = cmd["default_args"][:]
    if extra_args:
        args = extra_args

    run_cmd = []
    if cmd.get("sudo"):
        run_cmd = ["sudo", "-E"] #Preserve env so XRT can discover the device.
    run_cmd += ["/usr/local/share/pynq-venv/bin/python", script_path] + args

    try:
        proc = subprocess.Popen(
            run_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=SCRIPTS_DIR,
            start_new_session=True,
        )
        pgid = os.getpgid(proc.pid)

        try:
            stdout, stderr = proc.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            try:
                subprocess.run(
                    ["sudo", "kill", "-9", f"-{pgid}"],
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
            except Exception:
                proc.kill()
            stdout, stderr = proc.communicate()
            stderr = (stderr or "") + "\n[killed: 120s timeout]"

        output = stdout or ""
        if stderr:
            output += "\n[stderr]\n" + stderr
        if proc.returncode not in (0, -9, None):
            output += f"\n[exit code: {proc.returncode}]"
    except Exception as err:
        output = f"ERROR: {err}"

    output = output.strip() or "(no output)"
    if len(output) > 4000:
        output = output[:4000] + "\n... (truncated)"

    return output


def handle_update(update):
    #Process one Telegram update event.
    msg = update.get("message")
    if not msg or "text" not in msg:
        return

    chat_id = msg["chat"]["id"]
    user_id = msg["from"]["id"]
    text = msg["text"].strip()

    if ALLOWED_USER_IDS and user_id not in ALLOWED_USER_IDS:
        telegram_api("sendMessage", {"chat_id": chat_id, "text": "Unauthorised."})
        return

    parts = text.split()
    cmd = parts[0].split("@")[0].lstrip("/") if parts else ""
    extra_args = parts[1:] if len(parts) > 1 else None

    if cmd == "ping":
        reply = format_status()
        telegram_api(
            "sendMessage",
            {"chat_id": chat_id, "text": f"```\n{reply}\n```", "parse_mode": "Markdown"},
        )
        return

    if cmd in COMMANDS:
        telegram_api(
            "sendMessage",
            {"chat_id": chat_id, "text": f"Running {COMMANDS[cmd]['script']}..."},
        )
        output = run_script(cmd, extra_args)
        telegram_api(
            "sendMessage",
            {"chat_id": chat_id, "text": f"```\n{output}\n```", "parse_mode": "Markdown"},
        )
        return

    if cmd == "clearmem":
        subprocess.run(
            [
                "sudo",
                "sh",
                "-c",
                "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory",
            ],
            capture_output=True,
            timeout=10,
            check=False,
        )
        info = get_system_info()
        telegram_api(
            "sendMessage",
            {"chat_id": chat_id, "text": f"Caches dropped + memory compacted.\nMemory: {info['memory']}"},
        )
        return

    if cmd == "memtop":
        result = subprocess.run(
            ["ps", "aux", "--sort=-%mem"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        lines = result.stdout.strip().split("\n")
        output = "\n".join(lines[:12])
        telegram_api(
            "sendMessage",
            {"chat_id": chat_id, "text": f"```\n{output}\n```", "parse_mode": "Markdown"},
        )
        return

    if cmd == "cleanup":
        lines = []

        kill = subprocess.run(
            ["sudo", "pkill", "-9", "-f", PYNQ_PROC_PATTERN],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if kill.returncode == 0:
            lines.append("pkill: killed matching processes")
        else:
            lines.append("pkill: no matching processes found")

        subprocess.run(
            [
                "sudo",
                "sh",
                "-c",
                "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory",
            ],
            capture_output=True,
            timeout=10,
            check=False,
        )
        lines.append("caches dropped, memory compacted")

        info = get_system_info()
        lines.append(f"memory now: {info['memory']}")

        telegram_api("sendMessage", {"chat_id": chat_id, "text": "\n".join(lines)})
        return

    if cmd in ("help", "start"):
        help_lines = [
            "FPGA Ping Bot Commands:",
            "",
            "/ping      - Board status + power",
            "/clearmem  - Drop caches + compact memory",
            "/memtop    - Top memory consumers",
            "/cleanup   - Kill stray PYNQ procs + free memory",
            "/help      - This message",
            "",
        ]
        for name, info in COMMANDS.items():
            help_lines.append(f"/{name}  -  {info['description']}")

        telegram_api("sendMessage", {"chat_id": chat_id, "text": "\n".join(help_lines)})


def main():
    #Run the Telegram long-polling loop.
    if not BOT_TOKEN or BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print(
            "Set TELEGRAM_BOT_TOKEN in environment or edit BOT_TOKEN in this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    me = telegram_api("getMe")
    if not me or not me.get("ok"):
        print("[ERROR] Invalid bot token.", file=sys.stderr)
        sys.exit(1)

    bot_name = me["result"]["username"]
    print(f"[OK] Bot @{bot_name} is running. Waiting for /ping...")

    offset = 0
    while True:
        try:
            updates = telegram_api("getUpdates", {"offset": offset, "timeout": 30})
            if updates and updates.get("ok"):
                for update in updates["result"]:
                    handle_update(update)
                    offset = update["update_id"] + 1
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as err:
            print(f"[ERROR] {err}", file=sys.stderr)
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
