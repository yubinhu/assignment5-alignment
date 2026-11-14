from cs336_alignment.modal_utils import app, submit_commands

@app.local_entrypoint()
def main(*args):
    command = ["python", "-u", "-m", "cs336_alignment.train", *args]
    submit_commands([command])
