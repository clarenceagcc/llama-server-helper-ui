
import json
import shlex
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Generator  

import requests
import streamlit as st
import psutil

from helpers_ import is_windows, safe_kill_process, server_base, check_health, now_ms, approx_token_count, fetch_metrics_text, parse_prometheus_sample, pick_file

@dataclass
class ServerConfig:
    exe_path: str
    model_path: str
    host: str = "127.0.0.1"
    port: int = 8080

    # Startup knobs (restart-required)
    ngl: int = -1
    ctx: int = 32768
    batch: int = 8192
    ubatch: int = 512
    flash_attn: bool = True
    enable_metrics: bool = True

    # Extra raw args (advanced users)
    extra_args: str = ""

    def to_cmd(self) -> List[str]:
        cmd = [
            self.exe_path,
            "-m", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "-ngl", str(self.ngl),
            "-c", str(self.ctx),
            "-b", str(self.batch),
            "-ub", str(self.ubatch),
        ]
        if self.flash_attn:
            cmd += ["-fa", "on"]
        else:
            cmd += ["-fa", "off"]

        if self.enable_metrics:
            cmd += ["--metrics"]

        if self.extra_args.strip():
            cmd += shlex.split(self.extra_args, posix=not is_windows())

        return cmd


# ---------------------------
# Streamed chat (OpenAI-compatible)
# ---------------------------
def stream_chat_completion(
    host: str,
    port: int,
    payload: Dict[str, Any],
    timeout_s: float = 120.0
) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
    """
    Streams /v1/chat/completions and returns (final_text, perf_stats).
    perf_stats includes: ttft_ms, total_ms, out_chars, approx_out_tokens, approx_in_tokens
    """
    url = f"{server_base(host, port)}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    perf: Dict[str, Any] = {}

    t0 = now_ms()
    ttft = None
    out_chunks: List[str] = []
    out_text = ""

    # Approx prompt tokens (rough)
    in_text = ""
    for m in payload.get("messages", []):
        in_text += f"{m.get('role','')}: {m.get('content','')}\n"
    perf["approx_in_tokens"] = approx_token_count(in_text)

    with requests.post(url, headers=headers, json=payload, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        # llama-server OpenAI streaming uses "data: {json}\n\n" lines and ends with data: [DONE]
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break

            try:
                evt = json.loads(data)
            except Exception:
                continue

            # OpenAI format: choices[0].delta.content
            choices = evt.get("choices")

            # Some servers may send non-standard frames or empty choices.
            if not isinstance(choices, list) or len(choices) == 0:
                continue

            choice0 = choices[0] if isinstance(choices[0], dict) else {}
            delta_obj = choice0.get("delta", {}) if isinstance(choice0, dict) else {}

            delta = delta_obj.get("content", "")
            if not isinstance(delta, str):
                continue
            if delta:
                if ttft is None:
                    ttft = now_ms() - t0
                out_chunks.append(delta)
                out_text = "".join(out_chunks)
                yield out_text, {"event": "delta"}  # progressive yield

    t1 = now_ms()
    perf["ttft_ms"] = float(ttft) if ttft is not None else None
    perf["total_ms"] = float(t1 - t0)
    perf["out_chars"] = len(out_text)
    perf["approx_out_tokens"] = approx_token_count(out_text)
    if perf["total_ms"] > 0:
        perf["approx_toks_per_s"] = perf["approx_out_tokens"] / (perf["total_ms"] / 1000.0)
    else:
        perf["approx_toks_per_s"] = None

    yield out_text, {"event": "done", "perf": perf}


# ---------------------------
# UI State
# ---------------------------
if "proc" not in st.session_state:
    st.session_state.proc = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_perf" not in st.session_state:
    st.session_state.last_perf = None

if "last_metrics_text" not in st.session_state:
    st.session_state.last_metrics_text = None


st.set_page_config(page_title="llama-server Analysis UI", layout="wide")
st.title("🦙 llama-server Analysis UI")

# ---------------------------
# Sidebar: Server control plane
# ---------------------------
with st.sidebar:
    st.header("Server Control")

    st.subheader("Paths")

    col1, col2 = st.columns([4, 1])

    exe_path = col1.text_input(
        "llama-server.exe path",
        value=st.session_state.get("exe_path", r".\llama-server.exe")
    )

    if col2.button("Browse", key="browse_exe"):
        path = pick_file(
            filetypes=(("Executable", "*.exe"), ("All files", "*.*"))
        )
        if path:
            st.session_state.exe_path = path
            st.rerun()


    col3, col4 = st.columns([4, 1])

    model_path = col3.text_input(
        "Model (.gguf) path",
        value=st.session_state.get("model_path", r"C:\Models\Qwen3-8B-Q4_K_M.gguf")
    )

    if col4.button("Browse", key="browse_model"):
        path = pick_file(
            filetypes=(("GGUF models", "*.gguf"), ("All files", "*.*"))
        )
        if path:
            st.session_state.model_path = path
            st.rerun()


    host = st.text_input("Host", value="127.0.0.1")
    port = st.number_input("Port", min_value=1, max_value=65535, value=8080, step=1)

    st.subheader("Startup args (restart-required)")
    ngl = st.number_input("GPU layers (-ngl)", value=-1, step=1)
    ctx = st.number_input("Context (-c)", value=32768, step=1024)
    batch = st.number_input("Batch (-b)", value=8192, step=256)
    ubatch = st.number_input("U-batch (-ub)", value=512, step=64)
    flash_attn = st.checkbox("Flash attention (-fa on)", value=True)
    enable_metrics = st.checkbox("Enable /metrics (--metrics)", value=True)

    extra_args = st.text_area(
        "Extra args (advanced)",
        value="--min-p 0.0 --temp 0.6 --top-p 0.95",
        help="These are passed to llama-server at startup. Per-request sliders will override some values when you chat."
    )

    cfg = ServerConfig(
        exe_path=exe_path,
        model_path=model_path,
        host=host,
        port=int(port),
        ngl=int(ngl),
        ctx=int(ctx),
        batch=int(batch),
        ubatch=int(ubatch),
        flash_attn=flash_attn,
        enable_metrics=enable_metrics,
        extra_args=extra_args
    )

    colA, colB, colC = st.columns(3)
    start_clicked = colA.button("▶ Start", use_container_width=True)
    stop_clicked = colB.button("⏹ Stop", use_container_width=True)
    restart_clicked = colC.button("🔁 Restart", use_container_width=True)

    # Actions
    if start_clicked:
        if st.session_state.proc is not None and st.session_state.proc.poll() is None:
            st.warning("Server already running.")
        else:
            cmd = cfg.to_cmd()
            try:
                st.session_state.proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                st.success("Server started.")
            except Exception as e:
                st.session_state.proc = None
                st.error(f"Failed to start server: {e}")

    if stop_clicked:
        if st.session_state.proc is None or st.session_state.proc.poll() is not None:
            st.info("Server not running.")
        else:
            safe_kill_process(st.session_state.proc)
            st.session_state.proc = None
            st.success("Server stopped.")

    if restart_clicked:
        if st.session_state.proc is not None and st.session_state.proc.poll() is None:
            safe_kill_process(st.session_state.proc)
            st.session_state.proc = None
        cmd = cfg.to_cmd()
        try:
            st.session_state.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            st.success("Server restarted.")
        except Exception as e:
            st.session_state.proc = None
            st.error(f"Failed to restart server: {e}")

    running = check_health(cfg.host, cfg.port)
    st.markdown(f"**Status:** {'🟢 reachable' if running else '🔴 not reachable'}")

    if st.session_state.proc is not None and psutil is not None:
        try:
            p = psutil.Process(st.session_state.proc.pid)
            cpu = p.cpu_percent(interval=0.1)
            mem = p.memory_info().rss / (1024**3)
            st.caption(f"PID {p.pid} • CPU {cpu:.1f}% • RAM {mem:.2f} GB")
        except Exception:
            pass


# ---------------------------
# Main layout: Prompt Lab + Telemetry
# ---------------------------
left, right = st.columns([1.6, 1.0], gap="large")

with left:
    st.subheader("Prompt Lab")

    sys_template = st.text_area(
        "System prompt template",
        value="You are a helpful assistant.",
        height=120
    )

    user_template = st.text_area(
        "User prompt template",
        value="{user_message}",
        height=120,
        help="Use {user_message} to inject the chat input. You can add extra wrappers/instructions here."
    )

    st.divider()
    st.subheader("Per-request sampling")

    c1, c2, c3 = st.columns(3)
    temp = c1.slider("temperature", 0.0, 2.0, 0.7, 0.05)
    top_p = c2.slider("top_p", 0.0, 1.0, 0.95, 0.01)
    min_p = c3.slider("min_p", 0.0, 1.0, 0.0, 0.01)

    c4, c5, c6 = st.columns(3)
    max_tokens = c4.number_input("max_tokens", min_value=1, max_value=32768, value=1024, step=64)
    seed = c5.number_input("seed (optional)", min_value=0, max_value=2_147_483_647, value=0, step=1)
    use_seed = c6.checkbox("Use seed", value=False)

    st.divider()
    st.subheader("Chat")

    # render history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_msg = st.chat_input("Type a message…")

    if user_msg:
        # build final user content from template
        final_user = user_template.replace("{user_message}", user_msg)

        # push user message to history (for display)
        st.session_state.messages.append({"role": "user", "content": user_msg})

        # build request messages for server (system + conversation)
        request_messages = [{"role": "system", "content": sys_template}]
        # include conversation history as assistant/user turns
        for m in st.session_state.messages:
            if m["role"] in ("user", "assistant"):
                # (optional) you could store templated user messages separately
                request_messages.append({"role": m["role"], "content": m["content"]})

        # replace last user content with templated version (so server sees wrapper)
        request_messages[-1] = {"role": "user", "content": final_user}

        payload: Dict[str, Any] = {
            "model": "local-model",   # llama-server usually ignores/accepts placeholder
            "messages": request_messages,
            "temperature": float(temp),
            "top_p": float(top_p),
            "stream": True,
            "max_tokens": int(max_tokens),
        }

        # min_p / seed aren’t standard OpenAI fields; some servers accept extra sampler fields.
        # If your llama-server build supports them per-request, it will use them; otherwise it’ll ignore.
        payload["min_p"] = float(min_p)
        if use_seed:
            payload["seed"] = int(seed)

        with st.chat_message("assistant"):
            placeholder = st.empty()

            try:
                final_text = ""
                perf = None
                for text, meta in stream_chat_completion(cfg.host, cfg.port, payload):
                    if meta.get("event") == "delta":
                        placeholder.markdown(text)
                        final_text = text
                    elif meta.get("event") == "done":
                        final_text = text
                        perf = meta.get("perf")
                        placeholder.markdown(final_text)

                st.session_state.messages.append({"role": "assistant", "content": final_text})
                st.session_state.last_perf = perf

            except requests.exceptions.RequestException as e:
                err = f"Request failed: {e}"
                placeholder.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})


with right:
    st.subheader("Telemetry")

    # Client-side perf from last request
    perf = st.session_state.last_perf
    if perf:
        m1, m2 = st.columns(2)
        m1.metric("TTFT (ms)", f"{perf['ttft_ms']:.0f}" if perf["ttft_ms"] is not None else "—")
        m2.metric("Total (ms)", f"{perf['total_ms']:.0f}" if perf["total_ms"] is not None else "—")

        m3, m4 = st.columns(2)
        m3.metric("Approx out tokens", f"{perf['approx_out_tokens']}")
        m4.metric("Approx tok/s", f"{perf['approx_toks_per_s']:.2f}" if perf["approx_toks_per_s"] else "—")

        st.caption(f"Approx prompt tokens: {perf.get('approx_in_tokens', '—')}")

    else:
        st.info("Send a message to populate client-side metrics (TTFT, tok/s, latency).")

    st.divider()
    st.subheader("Server /metrics (if enabled)")

    colM1, colM2 = st.columns([1, 1])
    if colM1.button("🔄 Refresh /metrics", use_container_width=True):
        st.session_state.last_metrics_text = fetch_metrics_text(cfg.host, cfg.port)

    metrics_text = st.session_state.last_metrics_text
    if metrics_text:
        # Optional: attempt to extract a few interesting counters if present
        interesting_keys = [
            "llama_prompt_tokens_total",
            "llama_predicted_tokens_total",
            "llama_prompt_seconds_total",
            "llama_predicted_seconds_total",
        ]
        samples = parse_prometheus_sample(metrics_text, interesting_keys)
        if samples:
            for k, v in samples.items():
                st.write(f"**{k}**: `{v}`")
            st.caption("These keys may differ by build; raw metrics shown below.")

        with st.expander("Raw /metrics output", expanded=False):
            st.code(metrics_text[:20000], language="text")
    else:
        st.warning("No metrics yet. Start llama-server with `--metrics` and hit Refresh.")

    st.divider()
    st.subheader("Server process log (recent)")

    if st.session_state.proc is not None and st.session_state.proc.stdout is not None:
        # show last N lines without blocking
        try:
            # non-blocking-ish: read what's available quickly
            lines = []
            for _ in range(200):
                line = st.session_state.proc.stdout.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n"))
            if lines:
                st.code("\n".join(lines[-60:]), language="text")
            else:
                st.caption("No new log lines.")
        except Exception:
            st.caption("Log unavailable (stdout not readable).")
    else:
        st.caption("Start the server from the sidebar to view logs here.")
