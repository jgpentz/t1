from pathlib import Path
from sysconfig import get_paths
from typing import List

import rich_click as click
import uvicorn
from rich.console import Console

from splot_api import __version__

console = Console()
print = console.print

INTERFACE = "0.0.0.0"
PORT = 8080
site_packages = Path(get_paths()["purelib"])
splot_api_install_dir = [str(x) for x in site_packages.rglob("**/splot_api") if x.is_dir]
if splot_api_install_dir == []:
    splot_api_install_dir = None
DEFAULT_RELOAD_PATHS = splot_api_install_dir


@click.command("run", help="Start the local splot Gui API Helper")
@click.version_option(__version__, "--version", "-v")
@click.option(
    "--interface",
    "-i",
    type=str,
    default=INTERFACE,
    show_default=True,
    help="Interface that the API should run on",
)
@click.option(
    "--port",
    "-p",
    type=int,
    default=PORT,
    show_default=True,
    help="Port that the API shoulld run on",
)
@click.option(
    "--reload/--no-reload",
    type=bool,
    is_flag=True,
    default=True,
    show_default=True,
    help="Auto-reload",
)
@click.option(
    "--reload-dirs",
    "-d",
    type=list,
    default=DEFAULT_RELOAD_PATHS,
    show_default=True,
    help="Adjusts directories to reload files from",
)
def cli(
    interface: str,
    port: int,
    reload: bool,
    reload_dirs: List[str],
):
    print(f"[cyan]Starting the splot API v{__version__}[/]")
    print(f"[cyan]Attempting to run on {interface}:{port}[/]")
    print(f"[yellow]Auto-reload is[/] {'[green]Enabled[/]' if reload else '[red]Disabled[/]'}")
    uvicorn.run(
        "splot_api.splot_api:app",
        port=port,
        host=interface,
        reload=reload,
        reload_dirs=reload_dirs,
    )
    print("[red]Terminating splot API![/]")


if __name__ == "__main__":

    cli(["--reload", "-p", "3001"])
