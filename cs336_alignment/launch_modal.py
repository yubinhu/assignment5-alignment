from cs336_alignment.modal_utils import app, submit_commands

@app.local_entrypoint()
def main():
    submit_commands([
        ["python", "-u", "-m", "cs336_alignment.scripts"],
    ])