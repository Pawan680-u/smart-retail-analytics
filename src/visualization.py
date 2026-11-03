import plotly.express as px

PALETTE = ["#0f766e", "#2563eb", "#7c3aed", "#f59e0b", "#dc2626"]

def line_chart(data, x, y, title, color=None):
    return px.line(data, x=x, y=y, color=color, title=title, template="plotly_white", color_discrete_sequence=PALETTE)
