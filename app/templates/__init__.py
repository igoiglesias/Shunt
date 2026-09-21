"""Render Jinja2 templates from the templates directory."""

from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES_DIR = Path(__file__).parent

env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(["html", "xml"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render(template_name: str, request: Request, **context) -> HTMLResponse:
    """Render a template with the given context and return HTMLResponse."""
    template = env.get_template(template_name)
    content = template.render(request=request, **context)
    return HTMLResponse(content)