"""Welcome to Reflex! This file outlines the streaming dashboard app."""

import asyncio
import json

import reflex as rx
import uuid

def format_reply(data: dict) -> str:
    """Render a turn for the chat: spoken text first, then the options.

    The API keeps options out of `reply` so TTS never reads a numbered list
    aloud; they still belong on screen, so they get stitched back in here.
    """
    text = data.get("reply", "")
    options = data.get("options") or []
    if not options:
        return text
    listed = "\n".join(f"{i}. {option}" for i, option in enumerate(options, 1))
    return f"{text}\n\n{listed}"


# --- State ---
class AgentState(rx.State):
    show_chat: bool = False
    messages: list[dict[str, str]] = []
    current_input: str = ""
    thread_id: str = ""
    is_loading: bool = False
    is_recording: bool = False

    def set_current_input(self, val: str):
        self.current_input = val

    def handle_voice(self, result: str):
        """Receive the toggle script's outcome: recording started, or a transcript."""
        outcome = json.loads(result)
        self.is_recording = bool(outcome.get("recording"))

        if error := outcome.get("error"):
            self.messages.append({"role": "agent", "content": f"Error de voz: {error}"})
            return

        text = (outcome.get("text") or "").strip()
        if not text:
            return
        self.current_input = text
        return AgentState.send_message

    def toggle_chat(self):
        self.show_chat = not self.show_chat
        if self.show_chat and not self.thread_id:
            self.thread_id = str(uuid.uuid4())
            return AgentState.start_agent

    async def _turn(self, message: str | None):
        """Run one orchestrator turn and show its reply, narration included.

        Called in-process rather than over HTTP: the chat API lives in this
        same server, so a loopback request would only add a serialization
        round trip and break whenever the backend hot-reloads mid-request.
        The call blocks on the LLM and TTS, hence the thread.
        """
        audio_b64 = None
        try:
            data = await asyncio.to_thread(
                chat_with_agent, ChatRequest(thread_id=self.thread_id, message=message)
            )
            self.messages.append({"role": "agent", "content": format_reply(data)})
            audio_b64 = data.get("audio")
        except Exception as e:
            self.messages.append({"role": "agent", "content": f"Error: {str(e)}"})

        self.is_loading = False
        if audio_b64:
            yield rx.call_script(play_audio_script(audio_b64))

    async def start_agent(self):
        self.is_loading = True
        yield
        async for event in self._turn(None):
            yield event

    async def send_message(self):
        if not self.current_input.strip():
            return

        user_msg = self.current_input
        self.messages.append({"role": "user", "content": user_msg})
        self.current_input = ""
        self.is_loading = True
        yield

        async for event in self._turn(user_msg):
            yield event


from .api import ChatRequest, chat_with_agent
from .io_api import api as voice_api


# In dev the page is served by Vite (:3000) while these routes live on the
# Reflex backend (:8000), so a relative fetch would hit Vite and get HTML back.
# `getBackendURL`/`env` are in scope for Reflex's direct eval of call_script;
# the fallback keeps working if that ever stops being true.
BACKEND_URL_JS = """
  const apiURL = (path) => {
    try { return new URL(path, getBackendURL(env.PING).href).href; }
    catch (e) { return `${window.location.protocol}//${window.location.hostname}:8000${path}`; }
  };
"""


def play_audio_script(audio_b64: str) -> str:
    """JS that decodes the base64 wav the orchestrator's reply came with and plays it."""
    return f"""
(async () => {{
  try {{
    const bytes = Uint8Array.from(atob({json.dumps(audio_b64)}), (c) => c.charCodeAt(0));
    const blob = new Blob([bytes], {{ type: "audio/wav" }});
    new Audio(URL.createObjectURL(blob)).play();
  }} catch (e) {{}}
}})()
"""

# One click starts recording, the next stops it and transcribes. Returns a
# JSON string so the handler can tell "started" apart from "here's the text".
TOGGLE_RECORDING_JS = """
(async () => {
""" + BACKEND_URL_JS + """
  const recorder = window.__voiceRecorder;

  if (!recorder || recorder.state === "inactive") {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      window.__voiceChunks = [];
      rec.ondataavailable = (e) => window.__voiceChunks.push(e.data);
      rec.start();
      window.__voiceRecorder = rec;
      window.__voiceStream = stream;
      return JSON.stringify({ recording: true });
    } catch (e) {
      return JSON.stringify({ recording: false, error: "No se pudo abrir el micro: " + e.message });
    }
  }

  const stopped = new Promise((resolve) => { recorder.onstop = resolve; });
  recorder.stop();
  await stopped;
  window.__voiceStream.getTracks().forEach((t) => t.stop());
  window.__voiceRecorder = null;

  const mime = (recorder.mimeType || "audio/webm").split(";")[0];
  const ext = mime.includes("mp4") ? "mp4" : mime.includes("ogg") ? "ogg" : "webm";
  const blob = new Blob(window.__voiceChunks, { type: mime });
  if (!blob.size) return JSON.stringify({ recording: false, error: "No se grabó audio" });

  const form = new FormData();
  form.append("audio", blob, "clip." + ext);
  try {
    const res = await fetch(apiURL("/api/voice"), { method: "POST", body: form });
    if (!res.ok) return JSON.stringify({ recording: false, error: "STT HTTP " + res.status });
    const data = await res.json();
    return JSON.stringify({ recording: false, text: data.text || "" });
  } catch (e) {
    return JSON.stringify({ recording: false, error: e.message });
  }
})()
"""

# --- Styles ---
bg_dark = "#111111"
text_light = "#e5e5e5"
text_muted = "#999999"
titan_red = "#F4434B"
titan_blue = "#176B9C"

style = {
    "background_color": bg_dark,
    "color": text_light,
    "font_family": "Helvetica Neue, Helvetica, Arial, sans-serif",
    "min_height": "100vh",
    "margin": "0",
    "padding": "0",
}

# --- Components ---

def navbar() -> rx.Component:
    return rx.hstack(
        rx.hstack(
            rx.icon("search", color=text_light, size=20, margin_right="1em"),
            rx.vstack(
                rx.text("Home", weight="bold", font_size="1.1em"),
                rx.box(width="100%", height="2px", bg="white"),
                spacing="1",
                align_items="center"
            ),
            rx.text("Channels", color=text_muted, font_size="1.1em", _hover={"color": "white"}),
            rx.text("Apps", color=text_muted, font_size="1.1em", _hover={"color": "white"}),
            spacing="6",
            align_items="center",
        ),
        rx.hstack(
            rx.icon("user", color=text_muted, size=20),
            rx.icon("settings", color=text_muted, size=20),
            rx.text("10:15", color=text_muted, font_size="1.1em"),
            spacing="5",
            align_items="center",
        ),
        width="100%",
        padding_x="4%",
        padding_top="2em",
        padding_bottom="1em",
        position="absolute",
        top="0",
        z_index="999",
        justify="between",
        animation="fadeInDown 0.8s ease-out",
    )

def message_bubble(msg: dict) -> rx.Component:
    is_user = msg["role"] == "user"
    return rx.box(
        rx.markdown(msg["content"]),
        bg=rx.cond(is_user, titan_blue, "#333333"),
        color="white",
        padding="1em",
        border_radius="12px",
        margin_bottom="1em",
        align_self=rx.cond(is_user, "flex-end", "flex-start"),
        max_width="80%",
        animation="fadeInUp 0.3s ease-out",
        box_shadow="0 4px 6px rgba(0, 0, 0, 0.3)",
    )

def chat_modal() -> rx.Component:
    return rx.cond(
        AgentState.show_chat,
        rx.box(
            rx.vstack(
                # Header
                rx.hstack(
                    rx.text("KinoFiles Agent", font_weight="bold", font_size="1.2em"),
                    rx.icon("x", cursor="pointer", on_click=AgentState.toggle_chat, _hover={"color": titan_red}),
                    justify="between",
                    width="100%",
                    padding_bottom="1em",
                    border_bottom="1px solid #444",
                ),
                # Messages Area
                rx.vstack(
                    rx.foreach(AgentState.messages, message_bubble),
                    rx.cond(
                        AgentState.is_loading,
                        rx.spinner(color=titan_red, size="2"),
                    ),
                    width="100%",
                    flex="1",
                    overflow_y="auto",
                    padding_y="1em",
                    spacing="3",
                    align_items="stretch",
                ),
                # Input Area
                rx.hstack(
                    rx.input(
                        placeholder="Escribe tu respuesta...",
                        value=AgentState.current_input,
                        on_change=AgentState.set_current_input,
                        on_key_down=rx.call_script("if(event.key === 'Enter') { document.getElementById('send_btn').click(); }"),
                        bg="#222222",
                        border="1px solid #444",
                        color="white",
                        width="100%",
                        padding="0.8em",
                        border_radius="8px",
                    ),
                    rx.button(
                        rx.icon("send", size=18),
                        id="send_btn",
                        on_click=AgentState.send_message,
                        bg=titan_red,
                        color="white",
                        padding="1em",
                        border_radius="8px",
                        _hover={"bg": "#d32f2f"},
                    ),
                    rx.button(
                        rx.cond(
                            AgentState.is_recording,
                            rx.icon("square", size=18),
                            rx.icon("mic", size=18),
                        ),
                        on_click=rx.call_script(
                            TOGGLE_RECORDING_JS,
                            callback=AgentState.handle_voice,
                        ),
                        bg=rx.cond(AgentState.is_recording, "#d32f2f", "#333333"),
                        color="white",
                        padding="1em",
                        border_radius="8px",
                        _hover={"bg": "#444444"},
                    ),
                    width="100%",
                    padding_top="1em",
                ),
                width="100%",
                height="100%",
                padding="1.5em",
            ),
            position="fixed",
            top="10%",
            right="5%",
            width="400px",
            height="80vh",
            bg="rgba(25, 25, 25, 0.95)",
            backdrop_filter="blur(10px)",
            border="1px solid #333",
            border_radius="16px",
            z_index="1000",
            box_shadow="0 10px 30px rgba(0, 0, 0, 0.5)",
            animation="slideInRight 0.4s ease-out",
        ),
        rx.box()
    )

def hero() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.icon("bot", size=24, color=titan_red, animation="pulse 2s infinite"),
                rx.text("KinoFiles AI", color=text_light, font_weight="bold", font_size="1em"),
                spacing="2",
                align_items="center",
                margin_bottom="0.5em",
                animation="fadeInUp 0.8s ease-out",
            ),
            rx.heading("Kino Files Agent", size="9", weight="bold", letter_spacing="-1px", margin_bottom="0.5em", animation="fadeInUp 1s ease-out"),
            rx.text(
                "Tu asistente inteligente para explorar el universo cinematográfico. "
                "Descubre joyas ocultas, recibe recomendaciones personalizadas y encuentra "
                "la película perfecta basándote en tus gustos y estado de ánimo actual.",
                max_width="60%",
                font_size="1.1em",
                color="#cccccc",
                margin_bottom="2em",
                line_height="1.5",
                animation="fadeInUp 1.2s ease-out",
            ),
            rx.button(
                "Lanzar agente", 
                on_click=AgentState.toggle_chat,
                bg=titan_blue, 
                color="white", 
                font_size="1.1em",
                font_weight="bold",
                padding_x="2em", 
                padding_y="1.5em",
                border_radius="9999px",
                _hover={"bg": "#125a83", "transform": "scale(1.05)", "box_shadow": "0 0 15px rgba(23, 107, 156, 0.6)"},
                transition="all 0.2s ease-in-out",
                animation="fadeInUp 1.4s ease-out",
            ),
            align_items="flex-start",
            width="100%",
        ),
        padding_x="4%",
        width="100%",
        align_items="center",
        position="relative",
        z_index="2"
    )

def app_tile(name: str, color: str, index: int) -> rx.Component:
    return rx.box(
        rx.center(
            rx.text(name, weight="bold", color="white"),
            width="120px",
            height="90px",
            bg=color,
            border_radius="8px",
            transition="all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1)",
            border="2px solid transparent",
            _hover={
                "transform": "scale(1.1) translateY(-5px)", 
                "border": "2px solid white", 
                "z_index": "10",
                "box_shadow": f"0 10px 20px {color}80"
            },
            cursor="pointer",
        ),
        margin_right="10px",
        animation=f"fadeInRight 0.5s ease-out {0.1 * index}s both",
    )

def favourite_apps_row() -> rx.Component:
    apps = [
        {"name": "TV", "color": "#0055A4"},
        {"name": "NETFLIX", "color": "#E50914"},
        {"name": "prime video", "color": "#00A8E1"},
        {"name": "Disney+", "color": "#113CCF"},
        {"name": "HBO", "color": "#663399"},
        {"name": "Apple TV", "color": "#333333"},
        {"name": "YouTube", "color": "#FF0000"},
        {"name": "Twitch", "color": "#9146FF"},
    ]
    return rx.vstack(
        rx.text("Favourite Apps", size="3", color=text_muted, margin_bottom="0.5em", padding_x="4%", animation="fadeInUp 1.6s ease-out both"),
        rx.box(
            rx.hstack(
                *[app_tile(app["name"], app["color"], i) for i, app in enumerate(apps)],
                spacing="2",
                padding_x="4%",
                padding_bottom="1em",
            ),
            width="100%",
            overflow_x="scroll",
            style={"&::-webkit-scrollbar": {"display": "none"}, "msOverflowStyle": "none", "scrollbarWidth": "none"},
        ),
        width="100%",
        align_items="flex-start",
    )

def index() -> rx.Component:
    # Inject CSS keyframes for animations
    keyframes = """
    <style>
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeInRight {
            from { opacity: 0; transform: translateX(20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        @keyframes slideInRight {
            from { opacity: 0; transform: translateX(100%); }
            to { opacity: 1; transform: translateX(0); }
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
    </style>
    """
    
    return rx.box(
        rx.html(keyframes),
        rx.box(
            position="absolute",
            top="0",
            left="0",
            width="100%",
            height="100%",
            background_image="url('/movie_posters_bg.jpg')",
            background_size="cover",
            background_position="center center",
            z_index="0"
        ),
        rx.box(
            position="absolute",
            top="0",
            left="0",
            width="100%",
            height="100%",
            background_image="linear-gradient(to right, rgba(17,17,17,1) 0%, rgba(17,17,17,0.85) 40%, rgba(17,17,17,0.2) 100%)",
            z_index="1"
        ),
        navbar(),
        rx.vstack(
            hero(),
            favourite_apps_row(),
            justify="between",
            width="100%",
            height="100vh",
            padding_top="14vh",
            padding_bottom="4vh",
            z_index="2",
            position="relative"
        ),
        chat_modal(),
        **{**style, "background_color": "transparent"},
        height="100vh",
        overflow="hidden",
        position="relative"
    )

app = rx.App(
    style=style,
    api_transformer=voice_api,
)
app.add_page(index, title="KinoFiles OS")
