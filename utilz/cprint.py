# ansi_cprint.py

STYLES = {
    "reset": "0",
    "bold": "1",
    "italic": "3",
    "underline": "4",
}

FG_COLORS = {
    "black": "30",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "white": "37",

    "bright_black": "90",
    "bright_red": "91",
    "bright_green": "92",
    "bright_yellow": "93",
    "bright_blue": "94",
    "bright_magenta": "95",
    "bright_cyan": "96",
    "bright_white": "97",
}

BG_COLORS = {
    "red": "41",
    "green": "42",
    "yellow": "43",
    "blue": "44",
    "magenta": "45",
    "cyan": "46",
    "white": "47",
    "bright_black": "100",
    "bright_red": "101",
    "bright_green": "102",
    "bright_yellow": "103",
    "bright_blue": "104",
    "bright_magenta": "105",
    "bright_cyan": "106",
    "bright_white": "107",
}
def cprint(
    text,
    *,
    color=None,
    bg=None,
    bold=False,
    italic=False,
    underline=False,
    end="\n",
):
    codes = []

    if color:
        try:
            codes.append(FG_COLORS[color])
        except KeyError:
            raise ValueError(f"Unknown color: {color}")

    if bg:
        try:
            codes.append(BG_COLORS[bg])
        except KeyError:
            raise ValueError(f"Unknown bg color: {bg}")

    if bold:
        codes.append(STYLES["bold"])
    if italic:
        codes.append(STYLES["italic"])
    if underline:
        codes.append(STYLES["underline"])

    if codes:
        seq = "\033[" + ";".join(codes) + "m"
        reset = "\033[0m"
        print(f"{seq}{text}{reset}", end=end)
    else:
        print(text, end=end)

# %%
if __name__ == '__main__':
    
    cprint("OK", color="green")
    cprint("Warning", color="yellow", bold=True, bg="red")
    cprint("Error", color="red", bold=True, underline=True)
    cprint("Note", color="blue", italic=True)
