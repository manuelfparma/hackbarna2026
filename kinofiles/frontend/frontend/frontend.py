"""Welcome to Reflex! This file outlines the streaming dashboard app."""

import reflex as rx
import csv
import os

# --- Styles ---
bg_dark = "#111111"
text_light = "#e5e5e5"
text_muted = "#999999"
titan_red = "#F4434B" # Extracted from the user's provided logo
titan_blue = "#176B9C" # Blue pill button color from Titan OS mockup

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
        # Left side navigation
        rx.hstack(
            rx.icon("search", color=text_light, size=20, margin_right="1em"),
            rx.vstack(
                rx.text("Home", weight="bold", font_size="1.1em"),
                rx.box(width="100%", height="2px", bg="white"), # Active indicator
                spacing="1",
                align_items="center"
            ),
            rx.text("Channels", color=text_muted, font_size="1.1em", _hover={"color": "white"}),
            rx.text("Apps", color=text_muted, font_size="1.1em", _hover={"color": "white"}),
            spacing="6",
            align_items="center",
        ),
        # Right side icons
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
        position="absolute", # Overlap hero image
        top="0",
        z_index="999",
        justify="between",
    )

def hero() -> rx.Component:
    return rx.box(
        # The bottom gradient overlay to blend into the rest of the page
        rx.box(
            position="absolute",
            bottom="0",
            left="0",
            width="100%",
            height="60%",
            background_image="linear-gradient(to top, rgba(17,17,17,1) 0%, rgba(17,17,17,0.6) 50%, transparent 100%)",
            z_index="1"
        ),
        rx.hstack(
            rx.vstack(
                rx.hstack(
                    rx.icon("bot", size=24, color=titan_red),
                    rx.text("KinoFiles AI", color=text_light, font_weight="bold", font_size="1em"),
                    spacing="2",
                    align_items="center",
                    margin_bottom="0.5em"
                ),
                rx.heading("Kino Files Agent", size="9", weight="bold", letter_spacing="-1px", margin_bottom="0.5em"),
                rx.text(
                    "Tu asistente inteligente para explorar el universo cinematográfico. "
                    "Descubre joyas ocultas, recibe recomendaciones personalizadas y encuentra "
                    "la película perfecta basándote en tus gustos y estado de ánimo actual.",
                    max_width="60%",
                    font_size="1.1em",
                    color="#cccccc",
                    margin_bottom="2em",
                    line_height="1.5",
                ),
                rx.button(
                    "Lanzar agente", 
                    bg=titan_blue, 
                    color="white", 
                    font_size="1.1em",
                    font_weight="bold",
                    padding_x="2em", 
                    padding_y="1.5em",
                    border_radius="9999px", # Pill shape
                    _hover={"bg": "#125a83", "transform": "scale(1.05)"},
                    transition="all 0.2s ease-in-out",
                ),
                align_items="flex-start",
                width="100%",
                padding_top="10%",
            ),
            padding_x="4%",
            padding_bottom="5%",
            width="100%",
            height="100%",
            background_image="linear-gradient(to right, rgba(17,17,17,1) 0%, rgba(17,17,17,0.8) 35%, rgba(17,17,17,0.3) 70%, transparent 100%)",
            align_items="center",
            position="relative",
            z_index="2"
        ),
        width="100%",
        height="85vh",
        background_image="url('/images/agent_bg.jpg')", # User must provide agent_bg.jpg
        background_size="cover",
        background_position="center top",
        position="relative",
        margin_bottom="1em",
    )

def app_tile(name: str, color: str) -> rx.Component:
    return rx.box(
        rx.center(
            rx.text(name, weight="bold", color="white"),
            width="120px",
            height="90px",
            bg=color,
            border_radius="4px",
            transition="transform 0.2s, border 0.2s",
            border="2px solid transparent",
            _hover={"transform": "scale(1.05)", "border": "2px solid white", "z_index": "10"},
            cursor="pointer",
        ),
        margin_right="10px",
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
        rx.text("Favourite Apps", size="3", color=text_muted, margin_bottom="0.5em", padding_x="4%"),
        rx.box(
            rx.hstack(
                *[app_tile(app["name"], app["color"]) for app in apps],
                spacing="2",
                padding_x="4%",
                padding_bottom="1em",
            ),
            width="100%",
            overflow_x="scroll",
            style={"&::-webkit-scrollbar": {"display": "none"}, "msOverflowStyle": "none", "scrollbarWidth": "none"},
        ),
        width="100%",
        margin_bottom="2em",
        align_items="flex-start",
    )

def index() -> rx.Component:
    return rx.box(
        navbar(),
        hero(),
        favourite_apps_row(),
        **style,
    )

app = rx.App(
    style=style,
)
app.add_page(index, title="KinoFiles OS")
